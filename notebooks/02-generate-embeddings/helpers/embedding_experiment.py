import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import lancedb

HELPERS_DIR = Path(__file__).parent
REGISTRY_PATH = HELPERS_DIR / "model_registry.json"


def _load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def list_models() -> None:
    """Print all registered model families."""
    registry = _load_registry()
    print(f"{'Family':<15} {'Script':<45} Description")
    print("-" * 100)
    for name, info in registry.items():
        print(f"{name:<15} {info['script']:<45} {info.get('description', '')}")


def get_model_info(family: str) -> dict:
    """Get registry entry for a model family. Raises KeyError if not found."""
    registry = _load_registry()
    if family not in registry:
        available = ", ".join(registry.keys())
        raise KeyError(f"Unknown model family '{family}'. Available: {available}")
    info = dict(registry[family])
    info["script_path"] = str(HELPERS_DIR / info["script"])
    return info


def setup_experiment(
    project_name: str,
    author: str,
    source_uri,
    source_table: str,
    db_uri,
    project_root=None,
) -> dict:
    """Create config table with base metadata for a new embedding experiment.

    Returns dict with db connection, config table, and derived table names.
    """
    db = lancedb.connect(str(db_uri))

    config_name = project_name + "_config"
    img_emb_name = project_name + "_image_embeddings"
    patch_emb_name = project_name + "_patch_embeddings"

    source_path_str = str(source_uri)
    if project_root is not None:
        try:
            source_path_str = str(Path(source_uri).relative_to(project_root))
        except ValueError:
            pass

    config_data = [
        {"key": "created_at", "value": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"key": "author", "value": str(author)},
        {"key": "source", "value": str(source_table)},
        {"key": "source_path", "value": source_path_str},
        {"key": "tbl_img_emb", "value": img_emb_name},
        {"key": "tbl_patch_emb", "value": patch_emb_name},
    ]

    config_tbl = db.create_table(config_name, data=config_data, mode="overwrite")

    print(f"Config table '{config_name}' created with {len(config_data)} keys.")
    print(f"  Image embeddings table: {img_emb_name}")
    print(f"  Patch embeddings table: {patch_emb_name}")

    return {
        "db": db,
        "config_tbl": config_tbl,
        "config_name": config_name,
        "img_emb_name": img_emb_name,
        "patch_emb_name": patch_emb_name,
    }


def build_cli_command(
    script_path: str,
    source_db,
    source_table: str,
    config_db,
    config_table: str,
    out_prefix: str,
    model: str,
    img_id_field: str = "id",
    batch: int = 256,
    scan_batch: int = 2000,
    workers: int = 4,
    dtype: str = "fp16",
    limit: int = 0,
    image_size=None,
    pretrained: Optional[str] = None,
    extra_args: dict = None,
) -> str:
    """Build the CLI command string for an embedding pipeline script.

    Returns the full shell command as a string.
    """
    parts = [
        sys.executable,
        str(script_path),
        "--db",
        str(source_db),
        "--table",
        str(source_table),
        "--img_id_field",
        str(img_id_field),
        "--out_prefix",
        str(out_prefix),
        "--config_db",
        str(config_db),
        "--config_table",
        str(config_table),
        "--model",
        str(model),
        "--batch",
        str(batch),
        "--scan_batch",
        str(scan_batch),
        "--workers",
        str(workers),
        "--dtype",
        str(dtype),
    ]

    if pretrained is not None:
        parts.extend(["--pretrained", str(pretrained)])

    if limit and limit > 0:
        parts.extend(["--limit", str(limit)])

    if image_size is not None:
        parts.extend(["--image_size", str(image_size)])

    if extra_args:
        for k, v in extra_args.items():
            flag = k if k.startswith("--") else "--" + k
            parts.extend([flag, str(v)])

    return " \\\n  ".join(parts)


def run_experiment(
    script_path: str,
    source_db,
    source_table: str,
    config_db,
    config_table: str,
    out_prefix: str,
    model: str,
    img_id_field: str = "id",
    batch: int = 256,
    scan_batch: int = 2000,
    workers: int = 4,
    dtype: str = "fp16",
    limit: int = 0,
    image_size=None,
    pretrained: Optional[str] = None,
    extra_args: dict = None,
) -> int:
    """Run an embedding pipeline script inline via subprocess.

    Output streams directly to stdout. Returns the process exit code.
    """
    cmd = build_cli_command(
        script_path=script_path,
        source_db=source_db,
        source_table=source_table,
        config_db=config_db,
        config_table=config_table,
        out_prefix=out_prefix,
        model=model,
        img_id_field=img_id_field,
        batch=batch,
        scan_batch=scan_batch,
        workers=workers,
        dtype=dtype,
        limit=limit,
        image_size=image_size,
        pretrained=pretrained,
        extra_args=extra_args,
    )

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode


def load_config(db_uri, config_table_name: str) -> dict:
    """Load a config table into a Python dict."""
    db = lancedb.connect(str(db_uri))
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()
    return dict(zip(df["key"], df["value"]))


def upsert_config(db_uri, config_table_name: str, updates: dict) -> None:
    """Upsert key/value pairs into an existing config table."""
    db = lancedb.connect(str(db_uri))
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()

    for k, v in updates.items():
        k, v = str(k), str(v)
        if k in df["key"].values:
            df.loc[df["key"] == k, "value"] = v
        else:
            df.loc[len(df)] = [k, v]

    import pyarrow as pa

    new_tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl.delete("true")
    tbl.add(new_tbl)


def dir_size_bytes(path) -> int:
    """Return total size in bytes of all files under path (recursive)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += (Path(root) / f).stat().st_size
    return total


def format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (e.g. '1.23 GB')."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def print_table_sizes(db_uri, *table_names) -> None:
    """Print on-disk sizes of LanceDB tables under db_uri.

    Example
    -------
    print_table_sizes(DB_URI,
                      experiment["config_name"],
                      experiment["img_emb_name"],
                      experiment["patch_emb_name"])
    """
    db_path = Path(db_uri)
    print(f"{'Table':<45} {'Size':>10}")
    print("-" * 57)
    total = 0
    for name in table_names:
        p = db_path / f"{name}.lance"
        if p.exists():
            sz = dir_size_bytes(p)
            total += sz
            print(f"{name:<45} {format_bytes(sz):>10}")
        else:
            print(f"{name:<45} {'(not found)':>10}")
    print("-" * 57)
    print(f"{'TOTAL':<45} {format_bytes(total):>10}")
