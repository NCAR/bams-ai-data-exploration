#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import platform
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np

# Required for Parquet output
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_pkg_version(mod) -> str:
    try:
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------
# Config table writer (KV replace)
# -----------------------------
def write_run_config(db, config_table_name: str, kv: List[Tuple[str, str]]) -> None:
    """
    Store exactly one row per key by deleting any existing row(s) for that key
    and inserting the new value. Table schema:
      key: string
      value: string
    """
    try:
        tbl = db.open_table(config_table_name)
    except Exception:
        rows = [{"key": str(k), "value": "" if v is None else str(v)} for k, v in kv]
        db.create_table(config_table_name, data=rows)
        return

    for k, v in kv:
        key_str = str(k)
        val_str = "" if v is None else str(v)

        # SQL string literal escaping: single quote -> doubled
        key_sql = key_str.replace("'", "''")
        where = f"key = '{key_sql}'"

        try:
            tbl.delete(where)
        except Exception:
            # If delete isn't supported, this will append instead.
            # But in your env delete appears to work.
            pass

        tbl.add([{"key": key_str, "value": val_str}])


# -----------------------------
# DINOv3 via timm
# -----------------------------
def build_model_and_transform(model_name: str, image_size: Optional[int] = None):
    """
    Returns: model, preprocess, cfg
    Note: for timm==1.0.20 create_transform expects input_size, not img_size.
    """
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )

    cfg = resolve_data_config({}, model=model)

    # Override expected input size if user forces an image_size
    if image_size is not None:
        cfg = dict(cfg)
        cfg["input_size"] = (3, int(image_size), int(image_size))

    preprocess = create_transform(**{**cfg, "is_training": False})
    return model, preprocess, cfg


# -----------------------------
# Shard writer (Parquet)
# -----------------------------
def flush_shard_parquet(
    shard_idx: int,
    emb_rows: List[np.ndarray],
    meta_rows: List[Tuple[str, int, str, str]],
    emb_dir: Path,
) -> int:
    """
    Writes one Parquet shard:
      id (string), dim (int32), filename (string), dt (string), embedding (list<float32>)
    """
    n = len(meta_rows)
    if n == 0:
        return shard_idx

    ids: List[str] = []
    dims: List[int] = []
    filenames: List[str] = []
    dts: List[str] = []

    for idv, dimv, fnamev, dtv in meta_rows:
        ids.append(idv)
        dims.append(dimv)
        filenames.append(fnamev)
        dts.append(dtv)

    # embeddings as list<float32>
    arr_emb = pa.array(emb_rows, type=pa.list_(pa.float32()))
    table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "dim": pa.array(dims, type=pa.int32()),
            "filename": pa.array(filenames, type=pa.string()),
            "dt": pa.array(dts, type=pa.string()),
            "embedding": arr_emb,
        }
    )

    out_path = emb_dir / f"embeddings_{shard_idx:04d}.parquet"
    pq.write_table(table, out_path, compression="zstd")

    emb_rows.clear()
    meta_rows.clear()
    return shard_idx + 1


# -----------------------------
# Worker pool: decode + preprocess
# -----------------------------
_WORKER_PREPROCESS = None


def _worker_init(model_name: str, image_size: Optional[int]) -> None:
    global _WORKER_PREPROCESS
    # build preprocess in worker process (avoid sending transform object across processes)
    _, preprocess, _ = build_model_and_transform(model_name, image_size=image_size)
    _WORKER_PREPROCESS = preprocess


def _worker_decode_and_preprocess(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    task: {id, filename, dt, blob}
    returns: {id, filename, dt, tensor} or None
    """
    try:
        blob = task["blob"]
        im = Image.open(BytesIO(blob)).convert("RGB")
        t = _WORKER_PREPROCESS(im)  # [3,H,W]
        return {
            "id": task["id"],
            "filename": task["filename"],
            "dt": task["dt"],
            "tensor": t,
        }
    except Exception:
        return None


# -----------------------------


# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Stream+preprocess images from LanceDB for DINOv3 (timm) with worker pool")

    ap.add_argument("--config_db", type=str, required=True, help="Path/URI to LanceDB database for config table (separate from --db)")

    ap.add_argument("--db", type=str, required=True, help="Path/URI to LanceDB database (images)")
    ap.add_argument("--table", type=str, required=True, help="LanceDB image table name")
    ap.add_argument("--img_blob_field", type=str, default="image_blob", help="Blob column with encoded image bytes")

    ap.add_argument("--config_table", type=str, required=True, help="LanceDB table name to store run config (key/value)")

    ap.add_argument("--run_id", type=str, default="", help="Optional run id; if not set, auto-generated")
    ap.add_argument("--author", type=str, default="", help="Optional author (stored in config)")

    ap.add_argument("--out", type=str, default="preprocessed", help="Output root folder (files)")

    ap.add_argument("--model", type=str, default="vit_base_patch16_dinov3", help="Model name from timm")
    ap.add_argument("--image_size", type=int, default=None, help="Force eval size (e.g., 224 or 256)")

    ap.add_argument("--batch", type=int, default=256, help="GPU forward-pass batch size")
    ap.add_argument("--scan_batch", type=int, default=2000, help="Rows per streamed RecordBatch from Lance")
    ap.add_argument("--workers", type=int, default=4, help="Process workers for decode+preprocess")

    ap.add_argument("--shard_size", type=int, default=1000, help="Rows per Parquet shard")

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Inference dtype")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of rows (0 = no limit)")

    ap.add_argument("--save_embeddings", action="store_true", help="Write embeddings to Parquet shards (required for output)")

    args = ap.parse_args()

    if not args.save_embeddings:
        print("ERROR: specify --save_embeddings (this script writes embeddings to Parquet)", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Connect to image DB + config DB
    db_img = lancedb.connect(args.db)
    img_tbl = db_img.open_table(args.table)
    db_cfg = lancedb.connect(args.config_db)

    # Build model + preprocess and capture timm cfg
    model, preprocess, data_cfg = build_model_and_transform(args.model, image_size=args.image_size)

    # No 224 magic: use model cfg input_size as the default if args.image_size not set
    cfg_img = int(data_cfg["input_size"][1])
    target_size = int(args.image_size) if args.image_size is not None else cfg_img

    # Derive patch size + embedding dim
    patch_size_used = "unknown"
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        ps = model.patch_embed.patch_size
        patch_size_used = str(int(ps[0] if isinstance(ps, (tuple, list)) else ps))

    embedding_dim = "unknown"
    if hasattr(model, "num_features"):
        try:
            embedding_dim = str(int(model.num_features))
        except Exception:
            embedding_dim = "unknown"

    global_pool_used = "unknown"
    if hasattr(model, "global_pool"):
        gp = model.global_pool
        global_pool_used = gp if isinstance(gp, str) else gp.__class__.__name__

    # Device & precision
    device = pick_device()
    model.eval()
    model.to(device)

    use_half = (args.dtype == "fp16") and (device in ("cuda", "mps"))
    if use_half:
        try:
            model.half()
        except Exception:
            use_half = False

    # Token geometry: ask model for total tokens; derive patch tokens from (target_size, patch_size)
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        dummy = torch.zeros(1, 3, target_size, target_size, device=device, dtype=model_dtype)
        tokens = model.forward_features(dummy)

    num_tokens_total = int(tokens.shape[1])
    patch_int = int(patch_size_used) if patch_size_used != "unknown" else 0
    if patch_int <= 0:
        print("ERROR: could not infer patch size from model; cannot compute patch tokens.", file=sys.stderr)
        sys.exit(2)

    num_patch_tokens = (target_size // patch_int) * (target_size // patch_int)
    num_extra_tokens = num_tokens_total - num_patch_tokens

    # Determine run_id
    run_id = args.run_id.strip() or str(uuid.uuid4())

    # Record config (KV)
    kv: List[Tuple[str, str]] = []
    kv.append(("created_at", utc_now_iso()))
    if args.author:
        kv.append(("author", args.author))

    kv.append(("run_id", run_id))
    kv.append(("db_uri", args.db))
    kv.append(("img_table", args.table))
    kv.append(("img_blob_field", args.img_blob_field))

    kv.append(("model_name", args.model))
    kv.append(("pretrained", "true"))
    kv.append(("num_classes", "0"))
    kv.append(("global_pool_used", str(global_pool_used)))

    kv.append(("patch_size", str(patch_size_used)))
    kv.append(("embedding_dim", str(embedding_dim)))

    kv.append(("image_size_arg", "" if args.image_size is None else str(args.image_size)))
    kv.append(("image_size_used", str(target_size)))
    kv.append(("input_size_cfg", str(data_cfg.get("input_size"))))

    kv.append(("num_patch_tokens", str(num_patch_tokens)))
    kv.append(("num_extra_tokens", str(num_extra_tokens)))
    kv.append(("num_tokens_total", str(num_tokens_total)))

    kv.append(("mean_used", str(data_cfg.get("mean"))))
    kv.append(("std_used", str(data_cfg.get("std"))))
    kv.append(("interpolation_used", str(data_cfg.get("interpolation"))))
    kv.append(("crop_pct_used", str(data_cfg.get("crop_pct"))))

    kv.append(("device", device))
    kv.append(("dtype_requested", args.dtype))
    kv.append(("dtype_used", "fp16" if use_half else "fp32"))

    kv.append(("batch_size", str(args.batch)))
    kv.append(("scan_batch", str(args.scan_batch)))
    kv.append(("workers", str(args.workers)))
    kv.append(("shard_size", str(args.shard_size)))

    kv.append(("save_embeddings", "true"))
    kv.append(("out_dir", str(out_dir.resolve())))
    kv.append(("tbl_img_emb", str(emb_dir.resolve())))

    kv.append(("torch_version", get_pkg_version(torch)))
    kv.append(("python_version", sys.version.replace("\n", " ")))
    kv.append(("platform", platform.platform()))
    try:
        import timm  # type: ignore

        kv.append(("timm_version", get_pkg_version(timm)))
    except Exception:
        kv.append(("timm_version", "unknown"))

    write_run_config(db_cfg, args.config_table, kv)

    # Streaming scan via lance dataset
    blob_field = args.img_blob_field
    ds = img_tbl.to_lance()
    columns = ["id", "filename", "dt", blob_field]

    # Output buffers
    emb_rows: List[np.ndarray] = []
    meta_rows: List[Tuple[str, int, str, str]] = []
    shard_idx = 0
    rows_in_shard = 0

    # GPU batch buffers
    gpu_batch_tensors: List[torch.Tensor] = []
    gpu_batch_meta: List[Tuple[str, str, str]] = []

    mp_ctx = mp.get_context("spawn")
    pool = mp_ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.model, args.image_size),
    )

    processed = 0
    skipped_missing = 0
    skipped_decode = 0

    try:
        for rb in ds.to_batches(columns=columns, batch_size=args.scan_batch):
            if args.limit and processed >= args.limit:
                break

            col_id = rb.column(rb.schema.get_field_index("id"))
            col_filename = rb.column(rb.schema.get_field_index("filename"))
            col_dt = rb.column(rb.schema.get_field_index("dt"))
            col_blob = rb.column(rb.schema.get_field_index(blob_field))

            tasks: List[Dict[str, Any]] = []
            for i in range(rb.num_rows):
                if args.limit and processed + len(tasks) >= args.limit:
                    break

                blobv = col_blob[i].as_py()
                if blobv is None:
                    skipped_missing += 1
                    continue

                idv = col_id[i].as_py()
                fnamev = col_filename[i].as_py()
                dtv = col_dt[i].as_py()

                tasks.append(
                    {
                        "id": "" if idv is None else str(idv),
                        "filename": "" if fnamev is None else str(fnamev),
                        "dt": "" if dtv is None else str(dtv),
                        "blob": blobv,
                    }
                )

            if not tasks:
                continue

            for out in pool.imap(_worker_decode_and_preprocess, tasks, chunksize=16):
                if args.limit and processed >= args.limit:
                    break

                if out is None:
                    skipped_decode += 1
                    continue

                gpu_batch_tensors.append(out["tensor"])
                gpu_batch_meta.append((out["id"], out["filename"], out["dt"]))
                processed += 1

                if len(gpu_batch_tensors) >= args.batch:
                    imgs = torch.stack(gpu_batch_tensors)  # [B,3,H,W]
                    imgs = imgs.to(device=device, dtype=(torch.float16 if use_half else torch.float32))

                    with torch.inference_mode():
                        feats = model(imgs)
                        feats = torch.nn.functional.normalize(feats, dim=-1)

                    feats_cpu = feats.detach().cpu().numpy().astype(np.float32)

                    for j in range(feats_cpu.shape[0]):
                        emb = feats_cpu[j]
                        emb_rows.append(emb)
                        meta_rows.append(
                            (
                                gpu_batch_meta[j][0],
                                int(emb.shape[0]),
                                gpu_batch_meta[j][1],
                                gpu_batch_meta[j][2],
                            )
                        )

                        rows_in_shard += 1
                        if rows_in_shard >= args.shard_size:
                            shard_idx = flush_shard_parquet(
                                shard_idx=shard_idx,
                                emb_rows=emb_rows,
                                meta_rows=meta_rows,
                                emb_dir=emb_dir,
                            )
                            rows_in_shard = 0

                    gpu_batch_tensors.clear()
                    gpu_batch_meta.clear()

            print(
                f"Processed={processed}  Skipped(missing_blob)={skipped_missing}  Skipped(decode_fail)={skipped_decode}",
                file=sys.stderr,
            )

            if args.limit and processed >= args.limit:
                break

    finally:
        pool.close()
        pool.join()

    # Final partial GPU batch
    if gpu_batch_tensors:
        imgs = torch.stack(gpu_batch_tensors)
        imgs = imgs.to(device=device, dtype=(torch.float16 if use_half else torch.float32))

        with torch.inference_mode():
            feats = model(imgs)
            feats = torch.nn.functional.normalize(feats, dim=-1)

        feats_cpu = feats.detach().cpu().numpy().astype(np.float32)

        for j in range(feats_cpu.shape[0]):
            emb = feats_cpu[j]
            emb_rows.append(emb)
            meta_rows.append(
                (
                    gpu_batch_meta[j][0],
                    int(emb.shape[0]),
                    gpu_batch_meta[j][1],
                    gpu_batch_meta[j][2],
                )
            )
            rows_in_shard += 1

        gpu_batch_tensors.clear()
        gpu_batch_meta.clear()

    # Final shard flush
    if rows_in_shard:
        flush_shard_parquet(
            shard_idx=shard_idx,
            emb_rows=emb_rows,
            meta_rows=meta_rows,
            emb_dir=emb_dir,
        )

    print("\nDone.")
    print(f"- run_id:         {run_id}")
    print(f"- processed:      {processed}")
    print(f"- skipped_blob:   {skipped_missing}")
    print(f"- skipped_decode: {skipped_decode}")
    print(f"- device:         {device}")
    print(f"- dtype_used:     {'fp16' if use_half else 'fp32'}")
    print(f"- image_size:     {target_size}")
    print(f"- patch_size:     {patch_size_used}")
    print(f"- tokens_total:   {num_tokens_total} (patch={num_patch_tokens}, extra={num_extra_tokens})")
    print(f"- embeddings_dir: {emb_dir}")
    print(f"- config_table:   {args.config_table} (replaced per key)")


if __name__ == "__main__":
    main()
