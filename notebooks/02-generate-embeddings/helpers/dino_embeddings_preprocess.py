#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# optional deps
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **k):
        return x


# -----------------------------
# Helpers
# -----------------------------
def infer_input_size_from_preprocess(preprocess, default=224):
    try:
        for t in getattr(preprocess, "transforms", []):
            if hasattr(t, "size"):
                s = t.size
                if isinstance(s, (tuple, list)):
                    return int(s[0])
                return int(s)
    except Exception:
        pass
    return default


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    return sorted(p for p in root.glob("*") if p.suffix.lower() in exts)


def save_webp(img: Image.Image, dst: Path, size: int):
    # Save thumbnail at the model's input size for quick visual checks
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    img.save(dst, "WEBP", quality=90, method=6)


# -----------------------------
# DINOv3 via timm
# -----------------------------
def build_model_and_transform(model_name: str, image_size: int = None):
    try:
        import timm
        from timm.data import create_transform, resolve_data_config
    except Exception:
        print("ERROR: timm is required for DINOv3. Install: pip install timm", file=sys.stderr)
        raise

    # Create model that returns features directly
    # num_classes=0 makes most timm models output pooled features
    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")

    # Build eval transform matching the model
    cfg = resolve_data_config({}, model=model)
    if image_size is not None:
        preprocess = create_transform(**{**cfg, "is_training": False, "img_size": image_size})
    else:
        preprocess = create_transform(**{**cfg, "is_training": False})

    return model, preprocess


# -----------------------------
# Shard writer
# -----------------------------
def flush_shard(
    shard_idx,
    emb_rows,
    meta_rows,
    tens_rows,
    emb_dir,
    tens_dir,
    use_parquet=True,
    save_embeddings=True,
    save_tensors=True,
):
    n = len(meta_rows)
    if save_embeddings and n:
        if use_parquet:
            names = [m[0] for m in meta_rows]
            dims = [m[1] for m in meta_rows]
            paths = [m[2] for m in meta_rows]
            arr_emb = pa.array(emb_rows, type=pa.list_(pa.float32()))
            table = pa.table(
                {
                    "id": pa.array(names, type=pa.string()),
                    "dim": pa.array(dims, type=pa.int32()),
                    "path": pa.array(paths, type=pa.string()),
                    "embedding": arr_emb,
                }
            )
            out_path = emb_dir / f"embeddings_{shard_idx:04d}.parquet"
            pq.write_table(table, out_path, compression="zstd")
        else:
            out_npy = emb_dir / f"embeddings_{shard_idx:04d}.npy"
            out_meta = emb_dir / f"embeddings_{shard_idx:04d}.tsv"
            np.save(out_npy, np.stack(emb_rows, axis=0))
            with open(out_meta, "w") as f:
                f.write("id\tdim\tpath\n")
                for i, d, p in meta_rows:
                    f.write(f"{i}\t{d}\t{p}\n")

    if save_tensors and len(tens_rows):
        tens = torch.stack(tens_rows)  # [N, 3, H, W]
        out_pt = tens_dir / f"tensors_{shard_idx:04d}.pt"
        torch.save(tens, out_pt)

    emb_rows.clear()
    meta_rows.clear()
    tens_rows.clear()
    return shard_idx + 1


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Preprocess images folder for DINOv3 (timm)")
    ap.add_argument("--input", type=str, default="flowers_500", help="Folder with images")
    ap.add_argument("--out", type=str, default="preprocessed", help="Output root folder")

    ap.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_dinov3",
        help="DINOv3 model name from timm (e.g., vit_base_patch16_dinov3, vit_large_patch16_dinov3, etc.)",
    )
    ap.add_argument(
        "--pretrained", type=str, default="timm_default", help="Compatibility flag only (ignored for timm DINOv3)."
    )  # left it for compatibility with CLIP code
    ap.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Force eval size (e.g., 224 or 336). If not set, uses model default.",
    )
    ap.add_argument("--batch", type=int, default=32, help="Batch size for embeddings")
    ap.add_argument("--save_embeddings", action="store_true", help="Write embeddings to Parquet (recommended)")
    ap.add_argument("--save_tensors", action="store_true", help="Save model-ready tensors as .pt shards")
    ap.add_argument("--save_thumbs", action="store_true", help="Save thumbnails (size = model input) as WebP")
    ap.add_argument("--shard_size", type=int, default=100, help="Rows per Parquet/.pt shard")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Inference dtype")
    args = ap.parse_args()

    # Inputs
    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = list_images(in_dir)
    if not imgs:
        print(f"Found 0 images in {in_dir}.", file=sys.stderr)
        sys.exit(1)

    if args.pretrained not in (None, "", "timm_default"):
        print(f"[note] --pretrained='{args.pretrained}' is ignored for DINOv3 via timm.", file=sys.stderr)

    # Model + transform
    model, preprocess = build_model_and_transform(args.model, image_size=args.image_size)

    # Decide pixel size for thumbs and to report tensor shape
    target_size = args.image_size or infer_input_size_from_preprocess(preprocess, default=224)

    # Device & precision
    device = pick_device()
    model.eval().to(device)

    use_half = (args.dtype == "fp16") and (device in ("cuda", "mps"))
    if use_half:
        try:
            model.half()
        except Exception:
            use_half = False

    # Outputs
    if args.save_embeddings and not HAVE_PARQUET:
        print(
            "WARNING: pyarrow not found; installing it enables Parquet. Falling back to .npy shards.", file=sys.stderr
        )
    emb_dir = out_dir / "embeddings"
    tens_dir = out_dir / "tensors"
    thumb_dir = out_dir / "thumbs"
    for d in (emb_dir, tens_dir, thumb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Loader
    def load_one(p: Path):
        try:
            im = Image.open(p).convert("RGB")
            if args.save_thumbs:
                try:
                    save_webp(im, thumb_dir / f"{p.stem}.webp", size=target_size)
                except Exception as e:
                    print(f"[thumb warn] {p.name}: {e}", file=sys.stderr)
            # timm transform -> tensor [3,H,W], normalized as the model expects
            return preprocess(im)
        except Exception as e:
            print(f"Skipping {p.name}: {e}", file=sys.stderr)
            return None

    # Accumulators
    batched_paths: List[Path] = []
    batched_tensors: List[torch.Tensor] = []

    shard_idx = 0
    rows_in_shard = 0

    meta_rows: List[Tuple[str, int, str]] = []
    emb_rows: List[np.ndarray] = []
    tens_rows: List[torch.Tensor] = []

    # Loop
    for p in tqdm(imgs, desc="Preprocessing (DINOv3)"):
        t = load_one(p)
        if t is None:
            continue
        batched_paths.append(p)
        batched_tensors.append(t)
        if len(batched_tensors) >= args.batch:
            imgs_tensor = torch.stack(batched_tensors)  # [B,3,H,W]
            imgs_tensor = imgs_tensor.to(device=device, dtype=(torch.float16 if use_half else torch.float32))
            with torch.inference_mode():
                feats = model(imgs_tensor)  # pooled features (num_classes=0, global_pool='avg')
                feats = torch.nn.functional.normalize(feats, dim=-1)  # cosine-ready
            for spath, feat, tens in zip(batched_paths, feats.detach().cpu(), imgs_tensor.detach().cpu()):
                if args.save_embeddings:
                    emb_rows.append(feat.numpy().astype(np.float32))
                    meta_rows.append((spath.name, int(feat.shape[0]), str(spath)))
                if args.save_tensors:
                    tens_rows.append(tens.to(torch.float16 if use_half else torch.float32).cpu())
                rows_in_shard += 1

            if rows_in_shard >= args.shard_size:
                shard_idx = flush_shard(
                    shard_idx,
                    emb_rows,
                    meta_rows,
                    tens_rows,
                    emb_dir,
                    tens_dir,
                    use_parquet=HAVE_PARQUET and args.save_embeddings,
                    save_embeddings=args.save_embeddings,
                    save_tensors=args.save_tensors,
                )
                rows_in_shard = 0

            batched_paths.clear()
            batched_tensors.clear()

    # Final partial batch
    if batched_tensors:
        imgs_tensor = torch.stack(batched_tensors)
        imgs_tensor = imgs_tensor.to(device=device, dtype=(torch.float16 if use_half else torch.float32))
        with torch.inference_mode():
            feats = model(imgs_tensor)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        for spath, feat, tens in zip(batched_paths, feats.detach().cpu(), imgs_tensor.detach().cpu()):
            if args.save_embeddings:
                emb_rows.append(feat.numpy().astype(np.float32))
                meta_rows.append((spath.name, int(feat.shape[0]), str(spath)))
            if args.save_tensors:
                tens_rows.append(tens.to(torch.float16 if use_half else torch.float32).cpu())
            rows_in_shard += 1

    if rows_in_shard:
        flush_shard(
            shard_idx,
            emb_rows,
            meta_rows,
            tens_rows,
            emb_dir,
            tens_dir,
            use_parquet=HAVE_PARQUET and args.save_embeddings,
            save_embeddings=args.save_embeddings,
            save_tensors=args.save_tensors,
        )

    print("\nDone.")
    print(f"- Embeddings: {emb_dir if args.save_embeddings else '(skipped)'}")
    print(f"- Tensors:    {tens_dir if args.save_tensors else '(skipped)'}")
    print(f"- Thumbs:     {thumb_dir if args.save_thumbs else '(skipped)'}")


if __name__ == "__main__":
    main()
