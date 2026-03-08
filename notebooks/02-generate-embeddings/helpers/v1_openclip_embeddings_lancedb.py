#!/usr/bin/env python3
import argparse
import hashlib
import multiprocessing as mp
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_pkg_version(mod) -> str:
    try:
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


def get_git_info() -> Dict[str, str]:
    info = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_branch"] = branch
    except Exception:
        info["git_branch"] = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_dirty"] = "true" if len(status) > 0 else "false"
    except Exception:
        info["git_dirty"] = "unknown"

    return info


def get_script_sha256() -> str:
    try:
        script_path = Path(__file__).resolve()
        h = hashlib.sha256()
        h.update(script_path.read_bytes())
        return h.hexdigest()
    except Exception:
        return "unknown"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_run_config(db, config_table_name: str, kv: List[Tuple[str, str]]) -> None:
    try:
        tbl = db.open_table(config_table_name)
    except Exception:
        rows = []
        for k, v in kv:
            row = {"key": str(k), "value": "" if v is None else str(v)}
            rows.append(row)
        db.create_table(config_table_name, data=rows)
        return

    for k, v in kv:
        key_str = str(k)
        val_str = "" if v is None else str(v)

        key_sql = key_str.replace("'", "''")
        where = "key = '" + key_sql + "'"

        try:
            tbl.delete(where)
        except Exception:
            pass

        tbl.add([{"key": key_str, "value": val_str}])


def build_model_and_transform(model_name: str, pretrained: str, image_size: Optional[int] = None):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )

    if image_size is not None:
        preprocess = open_clip.image_transform(image_size, is_train=False)

    # Enable patch token output alongside the global pooled embedding
    model.visual.output_tokens = True

    return model, preprocess


def build_transform_only(image_size: int):
    """Build preprocessing transform without loading pretrained weights."""
    import open_clip

    return open_clip.image_transform(image_size, is_train=False)


# ---- Worker pool: decode + preprocess ----

_WORKER_PREPROCESS = None


def _worker_init(image_size: int) -> None:
    global _WORKER_PREPROCESS
    _WORKER_PREPROCESS = build_transform_only(image_size)


def _worker_decode_and_preprocess(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        blob = task["blob"]
        im = Image.open(BytesIO(blob)).convert("RGB")
        t = _WORKER_PREPROCESS(im)
        return {"image_id": task["image_id"], "tensor": t}
    except Exception:
        return None


def drop_if_exists(db, name: str) -> None:
    try:
        db.drop_table(name)
    except Exception:
        pass


def create_table_fresh(db, table_name: str, schema: pa.Schema):
    return db.create_table(table_name, schema=schema)


# ---- Attention hook ----


class CLIPAttentionHook:
    """Captures CLS-to-patch attention from the last transformer block.

    OpenCLIP uses nn.MultiheadAttention with a fused in_proj_weight [3C, C],
    so Q and K are sliced from the first two thirds of that weight matrix.
    The transformer operates in LND (seq_len, batch, channels) format.
    """

    def __init__(self, model, num_extra_tokens: int):
        self._attn_module = model.visual.transformer.resblocks[-1].attn
        self._num_extra_tokens = num_extra_tokens
        self._captured_input = None
        self._handle = None

    def register(self):
        def hook_fn(module, inputs, output):
            # inputs[0] is the query in LND format (same as key/value in self-attn)
            self._captured_input = inputs[0].detach().clone()

        self._handle = self._attn_module.register_forward_hook(hook_fn)

    def extract(self):
        """Recompute attention from captured input, return head-averaged CLS-to-patch map."""
        x = self._captured_input
        # open_clip 3.x uses batch_first=True (NLD); older versions used LND
        if not getattr(self._attn_module, "batch_first", True):
            x = x.permute(1, 0, 2)  # LND → NLD
        B, N, C = x.shape
        module = self._attn_module

        num_heads = module.num_heads
        head_dim = C // num_heads

        # Slice Q and K from fused in_proj_weight [3C, C]
        W = module.in_proj_weight
        b = module.in_proj_bias

        q = x @ W[:C].T + b[:C]
        k = x @ W[C : 2 * C].T + b[C : 2 * C]

        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        scale = head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        # CLS token (row 0) attending to patch tokens only
        cls_attn = attn[:, :, 0, self._num_extra_tokens :]

        # Average across heads
        cls_attn_avg = cls_attn.mean(dim=1)

        result = cls_attn_avg.detach().cpu().numpy().astype(np.float32)
        self._captured_input = None
        return result

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---- GPU inference ----


def run_inference(model, imgs, device, use_half, num_extra_tokens, attn_hook=None):
    if use_half:
        imgs = imgs.to(device=device, dtype=torch.float16)
    else:
        imgs = imgs.to(device=device, dtype=torch.float32)

    if attn_hook is not None:
        attn_hook.register()

    with torch.no_grad():
        # model.visual.output_tokens=True → returns (pooled, patch_tokens)
        # pooled: post ln_post + proj → shape [B, img_emb_dim]
        # patch_tokens: post ln_post, pre proj → shape [B, num_patch_tokens, patch_emb_dim]
        pooled, patch_tokens = model.visual(imgs)

        img_vec = F.normalize(pooled.float(), dim=-1)
        patch_tokens = F.normalize(patch_tokens.float(), dim=-1)

    attn_maps = None
    if attn_hook is not None:
        attn_maps = attn_hook.extract()
        attn_hook.remove()

    img_emb = img_vec.detach().cpu().numpy().astype(np.float32)
    patch_emb = patch_tokens.detach().cpu().numpy().astype(np.float32)
    return img_emb, patch_emb, attn_maps


# ---- Batch collector ----


class BatchCollector:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.tensors: List[torch.Tensor] = []
        self.image_ids: List[str] = []

    def add(self, tensor: torch.Tensor, image_id: str):
        self.tensors.append(tensor)
        self.image_ids.append(image_id)
        if len(self.tensors) >= self.batch_size:
            return self.flush()
        return None

    def flush(self):
        if len(self.tensors) == 0:
            return None
        stacked = torch.stack(self.tensors)
        ids = list(self.image_ids)
        self.tensors.clear()
        self.image_ids.clear()
        return stacked, ids


# ---- Async LanceDB writer ----


class AsyncLanceWriter:
    def __init__(self, img_emb_tbl, patch_emb_tbl):
        self._img_tbl = img_emb_tbl
        self._patch_tbl = patch_emb_tbl
        self._queue = queue.Queue(maxsize=2)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, image_ids: List[str], img_emb: np.ndarray, patch_emb: np.ndarray, attn_maps: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put((image_ids, img_emb, patch_emb, attn_maps))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._write_batch(item[0], item[1], item[2], item[3])
            except Exception as e:
                self._error = e
            finally:
                self._queue.task_done()

    def _write_batch(self, image_ids: List[str], img_emb: np.ndarray, patch_emb: np.ndarray, attn_maps: np.ndarray) -> None:
        img_rows = []
        patch_rows = []

        for b in range(len(image_ids)):
            image_id = image_ids[b]
            img_rows.append(
                {
                    "image_id": image_id,
                    "embedding": img_emb[b].tolist(),
                    "attention_map": attn_maps[b].tolist(),
                }
            )

            for p in range(patch_emb.shape[1]):
                patch_id = image_id + ":" + str(p)
                patch_rows.append(
                    {
                        "patch_id": patch_id,
                        "image_id": image_id,
                        "patch_index": int(p),
                        "embedding": patch_emb[b, p].tolist(),
                    }
                )

        if len(img_rows) > 0:
            self._img_tbl.add(img_rows)
        if len(patch_rows) > 0:
            self._patch_tbl.add(patch_rows)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute OpenCLIP image + patch embeddings and write to LanceDB tables (pipelined)")

    ap.add_argument("--db", type=str, required=True, help="LanceDB URI for raw images")
    ap.add_argument("--table", type=str, required=True, help="Raw image table name")
    ap.add_argument("--img_id_field", type=str, default="image_id", help="Image id column in raw table")
    ap.add_argument("--img_blob_field", type=str, default="image_blob", help="Image blob column in raw table")

    ap.add_argument("--config_db", type=str, required=True, help="LanceDB URI for config + output tables")
    ap.add_argument("--config_table", type=str, required=True, help="Config table name (key/value)")

    ap.add_argument("--out_prefix", type=str, required=True, help="Prefix for output tables")

    ap.add_argument("--run_id", type=str, default="", help="Optional run id; if not set, auto-generated")
    ap.add_argument("--author", type=str, default="", help="Optional author")

    ap.add_argument("--model", type=str, default="ViT-L-14", help="OpenCLIP model architecture (e.g. ViT-L-14, ViT-B-16)")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k", help="OpenCLIP pretrained weights tag (e.g. laion2b_s32b_b82k, openai)")
    ap.add_argument("--image_size", type=int, default=None, help="Force eval size (e.g., 224); defaults to model's native size")

    ap.add_argument("--batch", type=int, default=64, help="GPU batch size")
    ap.add_argument("--scan_batch", type=int, default=2000, help="Rows per RecordBatch from Lance")
    ap.add_argument("--workers", type=int, default=4, help="Decode+preprocess workers")

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Inference dtype")
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = no limit)")

    args = ap.parse_args()

    run_id = args.run_id.strip()
    if run_id == "":
        run_id = str(uuid.uuid4())

    import lancedb

    # Connect DBs
    db_img = lancedb.connect(args.db)
    raw_tbl = db_img.open_table(args.table)

    db_out = lancedb.connect(args.config_db)

    img_emb_table_name = args.out_prefix + "_image_embeddings"
    patch_emb_table_name = args.out_prefix + "_patch_embeddings"

    drop_if_exists(db_out, img_emb_table_name)
    drop_if_exists(db_out, patch_emb_table_name)

    # Load model
    model, _ = build_model_and_transform(args.model, args.pretrained, image_size=args.image_size)

    device = pick_device()
    model.eval()
    model.to(device)

    # Determine image size from model's positional embedding if not overridden
    # ViT positional embedding shape: [1 + num_patches, hidden_dim]
    pos_embed = model.visual.positional_embedding
    num_pos_tokens = int(pos_embed.shape[0])  # includes CLS

    # Infer patch size and image size from conv1 (patch embedding layer)
    patch_size_used = int(model.visual.conv1.weight.shape[-1])

    if args.image_size is not None:
        target_size = int(args.image_size)
    else:
        # Back-calculate from positional embedding: sqrt(num_pos_tokens - 1) * patch_size
        num_spatial_tokens = num_pos_tokens - 1  # subtract CLS
        grid = int(round(num_spatial_tokens**0.5))
        target_size = grid * patch_size_used

    if patch_size_used <= 0:
        print("ERROR: could not infer patch size from model.", file=sys.stderr)
        sys.exit(2)

    # Determine token geometry and embedding dims via dummy forward
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        dummy = torch.zeros(1, 3, target_size, target_size, device=device, dtype=model_dtype)
        dummy_pooled, dummy_patch_tokens = model.visual(dummy)

    img_emb_dim = int(dummy_pooled.shape[-1])
    patch_emb_dim = int(dummy_patch_tokens.shape[-1])
    num_patch_tokens = int(dummy_patch_tokens.shape[1])
    num_extra_tokens = num_pos_tokens - num_patch_tokens  # typically 1 (CLS)

    if num_extra_tokens < 0:
        print("ERROR: token math invalid; cannot slice patch tokens.", file=sys.stderr)
        sys.exit(2)

    # Output schemas — image and patch embeddings have different dims
    img_emb_schema = pa.schema(
        [
            pa.field("image_id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), img_emb_dim)),
            pa.field("attention_map", pa.list_(pa.float32(), num_patch_tokens)),
        ]
    )

    patch_emb_schema = pa.schema(
        [
            pa.field("patch_id", pa.string()),
            pa.field("image_id", pa.string()),
            pa.field("patch_index", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), patch_emb_dim)),
        ]
    )

    img_emb_tbl = create_table_fresh(db_out, img_emb_table_name, img_emb_schema)
    patch_emb_tbl = create_table_fresh(db_out, patch_emb_table_name, patch_emb_schema)

    use_half = False
    if args.dtype == "fp16" and (device == "cuda" or device == "mps"):
        try:
            model.half()
            use_half = True
        except Exception:
            use_half = False

    # Write run config
    kv: List[Tuple[str, str]] = []
    kv.append(("created_at", utc_now_iso()))
    if args.author:
        kv.append(("author", args.author))
    kv.append(("run_id", run_id))

    # Git reproducibility
    git_info = get_git_info()
    kv.append(("git_commit", git_info["git_commit"]))
    kv.append(("git_branch", git_info["git_branch"]))
    kv.append(("git_dirty", git_info["git_dirty"]))

    # Script identity
    kv.append(("script_name", os.path.basename(__file__)))
    kv.append(("script_sha256", get_script_sha256()))

    # Source data
    kv.append(("raw_db_uri", args.db))
    kv.append(("raw_table", args.table))
    kv.append(("raw_img_id_field", args.img_id_field))
    kv.append(("raw_img_blob_field", args.img_blob_field))
    raw_row_count = raw_tbl.count_rows()
    kv.append(("raw_table_row_count", str(raw_row_count)))

    # Model config
    kv.append(("model_name", args.model))
    kv.append(("pretrained_tag", args.pretrained))
    kv.append(("image_size_used", str(target_size)))
    kv.append(("patch_size", str(patch_size_used)))
    kv.append(("img_emb_dim", str(img_emb_dim)))
    kv.append(("patch_emb_dim", str(patch_emb_dim)))
    kv.append(("num_patch_tokens", str(num_patch_tokens)))
    kv.append(("num_extra_tokens", str(num_extra_tokens)))
    kv.append(("num_tokens_total", str(num_pos_tokens)))

    # Device and precision
    kv.append(("device", device))
    kv.append(("dtype_requested", args.dtype))
    kv.append(("dtype_used", "fp16" if use_half else "fp32"))

    # Pipeline config
    kv.append(("batch_size", str(args.batch)))
    kv.append(("scan_batch", str(args.scan_batch)))
    kv.append(("workers", str(args.workers)))
    kv.append(("limit", str(args.limit)))
    kv.append(("mp_start_method", "spawn"))

    kv.append(("img_emb_table_current", img_emb_table_name))
    kv.append(("patch_emb_table_current", patch_emb_table_name))

    # Attention map metadata
    attn_layer_index = len(model.visual.transformer.resblocks) - 1
    attn_num_heads = model.visual.transformer.resblocks[attn_layer_index].attn.num_heads
    spatial_h = target_size // patch_size_used
    spatial_w = target_size // patch_size_used
    kv.append(("attention_layer_index", str(attn_layer_index)))
    kv.append(("attention_num_heads_averaged", str(attn_num_heads)))
    kv.append(("attention_spatial_h", str(spatial_h)))
    kv.append(("attention_spatial_w", str(spatial_w)))
    kv.append(("attention_num_patch_tokens", str(num_patch_tokens)))
    kv.append(("attention_dtype", "float32"))
    kv.append(("attention_storage", "pa.list_(pa.float32(), num_patch_tokens)"))

    # Versions
    kv.append(("torch_version", get_pkg_version(torch)))
    kv.append(("python_version", sys.version.replace("\n", " ")))
    kv.append(("platform", platform.platform()))
    try:
        import open_clip

        kv.append(("open_clip_version", get_pkg_version(open_clip)))
    except Exception:
        kv.append(("open_clip_version", "unknown"))

    write_run_config(db_out, args.config_table, kv)

    # Count total rows for progress bar
    total_rows = raw_tbl.count_rows()
    if args.limit and args.limit < total_rows:
        total_rows = args.limit

    # Streaming scan
    ds = raw_tbl.to_lance()
    columns = [args.img_id_field, args.img_blob_field]

    # Pipeline components
    collector = BatchCollector(args.batch)
    writer = AsyncLanceWriter(img_emb_tbl, patch_emb_tbl)
    attn_hook = CLIPAttentionHook(model, num_extra_tokens)

    mp_ctx = mp.get_context("spawn")
    pool = mp_ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(target_size,),
    )

    processed = 0
    skipped_missing = 0
    skipped_decode = 0
    gpu_batches = 0
    t_start = time.time()

    pbar = tqdm(
        total=total_rows,
        desc="Embedding",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    try:
        for rb in ds.to_batches(columns=columns, batch_size=args.scan_batch):
            if args.limit and processed >= args.limit:
                break

            col_id = rb.column(rb.schema.get_field_index(args.img_id_field))
            col_blob = rb.column(rb.schema.get_field_index(args.img_blob_field))

            # Build task list from this record batch
            tasks = []
            for i in range(rb.num_rows):
                if args.limit and (processed + len(tasks)) >= args.limit:
                    break

                blobv = col_blob[i].as_py()
                if blobv is None:
                    skipped_missing += 1
                    continue

                idv = col_id[i].as_py()
                image_id = "" if idv is None else str(idv)
                tasks.append({"image_id": image_id, "blob": blobv})

            if len(tasks) == 0:
                continue

            # Parallel decode + preprocess, no ordering constraint
            for out in pool.imap_unordered(_worker_decode_and_preprocess, tasks, chunksize=16):
                if args.limit and processed >= args.limit:
                    break

                if out is None:
                    skipped_decode += 1
                    continue

                processed += 1
                pbar.update(1)

                ready = collector.add(out["tensor"], out["image_id"])
                if ready is not None:
                    stacked, ids = ready
                    img_emb, patch_emb, attn_maps = run_inference(model, stacked, device, use_half, num_extra_tokens, attn_hook)
                    writer.submit(ids, img_emb, patch_emb, attn_maps)
                    gpu_batches += 1

            if args.limit and processed >= args.limit:
                break

        # Flush remaining tensors
        remaining = collector.flush()
        if remaining is not None:
            stacked, ids = remaining
            img_emb, patch_emb, attn_maps = run_inference(model, stacked, device, use_half, num_extra_tokens, attn_hook)
            writer.submit(ids, img_emb, patch_emb, attn_maps)
            gpu_batches += 1

    finally:
        pool.close()
        pool.join()
        writer.close()
        pbar.close()

    elapsed = time.time() - t_start

    # Write post-run stats to config
    post_kv = []
    post_kv.append(("processed_image_count", str(processed)))
    post_kv.append(("elapsed_seconds", "{:.1f}".format(elapsed)))
    if processed > 0:
        post_kv.append(("throughput_img_per_sec", "{:.1f}".format(processed / elapsed)))
    write_run_config(db_out, args.config_table, post_kv)

    print("\nDone.")
    print("- run_id: " + run_id)
    print("- processed: " + str(processed))
    print("- skipped_blob: " + str(skipped_missing))
    print("- skipped_decode: " + str(skipped_decode))
    print("- gpu_batches: " + str(gpu_batches))
    print("- device: " + device)
    print("- dtype_used: " + ("fp16" if use_half else "fp32"))
    print("- model: " + args.model)
    print("- pretrained: " + args.pretrained)
    print("- image_size: " + str(target_size))
    print("- patch_size: " + str(patch_size_used))
    print("- tokens_total: " + str(num_pos_tokens))
    print("- img_emb_dim: " + str(img_emb_dim))
    print("- patch_emb_dim: " + str(patch_emb_dim))
    print("- img_emb_table: " + img_emb_table_name)
    print("- patch_emb_table: " + patch_emb_table_name)
    print("- config_table: " + args.config_table)
    print("- attention_layer: " + str(attn_layer_index))
    print("- attention_shape: [" + str(spatial_h) + ", " + str(spatial_w) + "]")
    print("- elapsed: " + "{:.1f}".format(elapsed) + "s")
    if processed > 0:
        print("- throughput: " + "{:.1f}".format(processed / elapsed) + " img/s")


if __name__ == "__main__":
    main()
