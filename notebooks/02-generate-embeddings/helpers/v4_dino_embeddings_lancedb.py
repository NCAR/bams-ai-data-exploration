#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import platform
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm


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


def build_model_and_transform(model_name: str, image_size: Optional[int] = None):
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )

    cfg = resolve_data_config({}, model=model)

    if image_size is not None:
        cfg = dict(cfg)
        cfg["input_size"] = (3, int(image_size), int(image_size))

    preprocess = create_transform(**{**cfg, "is_training": False})
    return model, preprocess, cfg


def build_transform_only(model_name: str, image_size: Optional[int] = None):
    """Build preprocessing transform without loading pretrained weights."""
    import timm
    from timm.data import create_transform, resolve_data_config

    model_stub = timm.create_model(model_name, pretrained=False, num_classes=0)
    cfg = resolve_data_config({}, model=model_stub)
    del model_stub

    if image_size is not None:
        cfg = dict(cfg)
        cfg["input_size"] = (3, int(image_size), int(image_size))

    preprocess = create_transform(**{**cfg, "is_training": False})
    return preprocess


# ---- Worker pool: decode + preprocess ----

_WORKER_PREPROCESS = None


def _worker_init(model_name: str, image_size: Optional[int]) -> None:
    global _WORKER_PREPROCESS
    _WORKER_PREPROCESS = build_transform_only(model_name, image_size=image_size)


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


# ---- GPU inference ----


def run_inference(model, imgs: torch.Tensor, device: str, use_half: bool, num_extra_tokens: int):
    if use_half:
        imgs = imgs.to(device=device, dtype=torch.float16)
    else:
        imgs = imgs.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        tokens = model.forward_features(imgs)
        patch_tokens = tokens[:, num_extra_tokens:, :]
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
        img_vec = patch_tokens.mean(dim=1)
        img_vec = torch.nn.functional.normalize(img_vec, dim=-1)

    img_emb = img_vec.detach().cpu().numpy().astype(np.float32)
    patch_emb = patch_tokens.detach().cpu().numpy().astype(np.float32)
    return img_emb, patch_emb


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

    def submit(self, image_ids: List[str], img_emb: np.ndarray, patch_emb: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put((image_ids, img_emb, patch_emb))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._write_batch(item[0], item[1], item[2])
            except Exception as e:
                self._error = e
            finally:
                self._queue.task_done()

    def _write_batch(self, image_ids: List[str], img_emb: np.ndarray, patch_emb: np.ndarray) -> None:
        img_rows = []
        patch_rows = []

        for b in range(len(image_ids)):
            image_id = image_ids[b]
            img_rows.append({"image_id": image_id, "embedding": img_emb[b].tolist()})

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
    ap = argparse.ArgumentParser(description="Compute DINOv3 image + patch embeddings and write to LanceDB tables (pipelined)")

    ap.add_argument("--db", type=str, required=True, help="LanceDB URI for raw images")
    ap.add_argument("--table", type=str, required=True, help="Raw image table name")
    ap.add_argument("--img_id_field", type=str, default="image_id", help="Image id column in raw table")
    ap.add_argument("--img_blob_field", type=str, default="image_blob", help="Image blob column in raw table")

    ap.add_argument("--config_db", type=str, required=True, help="LanceDB URI for config + output tables")
    ap.add_argument("--config_table", type=str, required=True, help="Config table name (key/value)")

    ap.add_argument("--out_prefix", type=str, required=True, help="Prefix for output tables")

    ap.add_argument("--run_id", type=str, default="", help="Optional run id; if not set, auto-generated")
    ap.add_argument("--author", type=str, default="", help="Optional author")

    ap.add_argument("--model", type=str, default="vit_base_patch16_dinov3", help="timm model name")
    ap.add_argument("--image_size", type=int, default=None, help="Force eval size (e.g., 224)")

    ap.add_argument("--batch", type=int, default=256, help="GPU batch size")
    ap.add_argument("--scan_batch", type=int, default=2000, help="Rows per RecordBatch from Lance")
    ap.add_argument("--workers", type=int, default=4, help="Decode+preprocess workers")

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Inference dtype")
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = no limit)")

    args = ap.parse_args()

    run_id = args.run_id.strip()
    if run_id == "":
        run_id = str(uuid.uuid4())

    # Connect DBs
    db_img = lancedb.connect(args.db)
    raw_tbl = db_img.open_table(args.table)

    db_out = lancedb.connect(args.config_db)

    img_emb_table_name = args.out_prefix + "_image_embeddings"
    patch_emb_table_name = args.out_prefix + "_patch_embeddings"

    drop_if_exists(db_out, img_emb_table_name)
    drop_if_exists(db_out, patch_emb_table_name)

    # Load model
    model, _, data_cfg = build_model_and_transform(args.model, image_size=args.image_size)

    patch_size_used = None
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        ps = model.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            patch_size_used = int(ps[0])
        else:
            patch_size_used = int(ps)

    if patch_size_used is None or patch_size_used <= 0:
        print("ERROR: could not infer patch size from model.", file=sys.stderr)
        sys.exit(2)

    cfg_img = int(data_cfg["input_size"][1])
    target_size = cfg_img

    device = pick_device()
    model.eval()
    model.to(device)

    # Determine token geometry
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        dummy = torch.zeros(1, 3, target_size, target_size, device=device, dtype=model_dtype)
        tok = model.forward_features(dummy)

    num_tokens_total = int(tok.shape[1])
    num_patch_tokens = (target_size // patch_size_used) * (target_size // patch_size_used)
    num_extra_tokens = num_tokens_total - num_patch_tokens
    if num_extra_tokens < 0:
        print("ERROR: token math invalid; cannot slice patch tokens.", file=sys.stderr)
        sys.exit(2)

    embedding_dim = int(tok.shape[2])

    # Output schemas
    img_emb_schema = pa.schema(
        [
            pa.field("image_id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        ]
    )

    patch_emb_schema = pa.schema(
        [
            pa.field("patch_id", pa.string()),
            pa.field("image_id", pa.string()),
            pa.field("patch_index", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
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

    kv.append(("raw_db_uri", args.db))
    kv.append(("raw_table", args.table))
    kv.append(("raw_img_id_field", args.img_id_field))
    kv.append(("raw_img_blob_field", args.img_blob_field))

    kv.append(("model_name", args.model))
    kv.append(("image_size_used", str(target_size)))
    kv.append(("patch_size", str(patch_size_used)))
    kv.append(("embedding_dim", str(embedding_dim)))
    kv.append(("num_patch_tokens", str(num_patch_tokens)))
    kv.append(("num_extra_tokens", str(num_extra_tokens)))
    kv.append(("num_tokens_total", str(num_tokens_total)))

    kv.append(("device", device))
    kv.append(("dtype_requested", args.dtype))
    kv.append(("dtype_used", "fp16" if use_half else "fp32"))

    kv.append(("batch_size", str(args.batch)))
    kv.append(("scan_batch", str(args.scan_batch)))
    kv.append(("workers", str(args.workers)))

    kv.append(("img_emb_table_current", img_emb_table_name))
    kv.append(("patch_emb_table_current", patch_emb_table_name))

    kv.append(("torch_version", get_pkg_version(torch)))
    kv.append(("python_version", sys.version.replace("\n", " ")))
    kv.append(("platform", platform.platform()))
    try:
        import timm

        kv.append(("timm_version", get_pkg_version(timm)))
    except Exception:
        kv.append(("timm_version", "unknown"))

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

    mp_ctx = mp.get_context("spawn")
    pool = mp_ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.model, args.image_size),
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
                    img_emb, patch_emb = run_inference(model, stacked, device, use_half, num_extra_tokens)
                    writer.submit(ids, img_emb, patch_emb)
                    gpu_batches += 1

            if args.limit and processed >= args.limit:
                break

        # Flush remaining tensors
        remaining = collector.flush()
        if remaining is not None:
            stacked, ids = remaining
            img_emb, patch_emb = run_inference(model, stacked, device, use_half, num_extra_tokens)
            writer.submit(ids, img_emb, patch_emb)
            gpu_batches += 1

    finally:
        pool.close()
        pool.join()
        writer.close()
        pbar.close()

    elapsed = time.time() - t_start

    print("\nDone.")
    print("- run_id: " + run_id)
    print("- processed: " + str(processed))
    print("- skipped_blob: " + str(skipped_missing))
    print("- skipped_decode: " + str(skipped_decode))
    print("- gpu_batches: " + str(gpu_batches))
    print("- device: " + device)
    print("- dtype_used: " + ("fp16" if use_half else "fp32"))
    print("- image_size: " + str(target_size))
    print("- patch_size: " + str(patch_size_used))
    print("- tokens_total: " + str(num_tokens_total))
    print("- img_emb_table: " + img_emb_table_name)
    print("- patch_emb_table: " + patch_emb_table_name)
    print("- config_table: " + args.config_table)
    print("- elapsed: " + "{:.1f}".format(elapsed) + "s")
    if processed > 0:
        print("- throughput: " + "{:.1f}".format(processed / elapsed) + " img/s")


if __name__ == "__main__":
    main()
