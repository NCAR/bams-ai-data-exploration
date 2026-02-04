"""
helpers/ingest_images.py

Notebook-friendly ingestion + CLI entrypoint.

- No datetime_utils.
- You can pass an already-open LanceDB table object (recommended for notebooks).
- Datetime parsing is done directly with datetime.strptime using the exact format you provide.

Notebook usage (recommended):
  from helpers.ingest_images import ingest_images_to_table
  ingest_images_to_table(
      table_obj=table,
      image_dir="/path/to/images",
      width=224,
      height=224,
      dt_format="%Y%m%d_%H%M%S_rgb.jpeg",
  )

CLI usage (expects the table already exists):
  python -m helpers.ingest_images \
    --image_dir /path/to/images \
    --db_dir /path/to/lancedb \
    --table images \
    --width 224 \
    --height 224 \
    --thumb_size 64 \
    --dt_format "%Y%m%d_%H%M%S_rgb.jpeg" \
    --batch_size 256

Note:
  This script does NOT drop/recreate tables. Do that explicitly in your notebook/pipeline
  so there are no surprises.
"""

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Union

import lancedb
from tqdm import tqdm

from .image_utils import image_to_jpeg_bytes, image_to_png_bytes, open_rgb_image, resize_image

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_images_flat(image_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in image_dir.iterdir():
        if path.is_file() and is_image_file(path):
            files.append(path)
    files.sort()
    return files


def parse_dt_from_filename(filename: str, fmt: str) -> datetime:
    """
    Parse datetime from filename using an explicit format string.

    Example:
      filename = "20240103_142210_rgb.jpeg"
      fmt      = "%Y%m%d_%H%M%S_rgb.jpeg"
    """
    try:
        return datetime.strptime(filename, fmt)
    except ValueError as e:
        raise ValueError(f"Failed to parse datetime from filename '{filename}' with format '{fmt}'.") from e


def ingest_images_to_table(
    table_obj,
    image_dir: Union[str, Path],
    *,
    width: int,
    height: int,
    dt_format: str,
    thumb_size: int = 64,
    batch_size: int = 256,
) -> int:
    """
    Ingest images from a flat directory into an existing LanceDB table.

    Args:
      table_obj: an open LanceDB table (e.g. db.open_table("images"))
      image_dir: directory containing image files (flat; no recursion)
      width/height: stored main image resolution
      dt_format: exact datetime format applied to the full filename (including extension if you want)
      thumb_size: square thumbnail size
      batch_size: rows per add()

    Returns:
      count of ingested images
    """
    image_dir = Path(image_dir)

    paths = list_images_flat(image_dir)
    if not paths:
        raise ValueError(f"No image files found in directory: {image_dir}")

    batch = []
    count = 0

    for path in tqdm(paths, desc="Ingesting"):
        filename = path.name
        id_name = path.stem
        dt = parse_dt_from_filename(filename, dt_format)

        img = open_rgb_image(path)

        resized = resize_image(img, width, height)
        image_blob = image_to_png_bytes(resized)

        # This ensures that if you insert the exact same image data twice,
        # it generates the same ID (deduplication).
        file_hash = hashlib.md5(image_blob).hexdigest()  # MD5 to generate a deterministic, content-based ID

        thumb = resize_image(img, thumb_size, thumb_size)
        thumb_blob = image_to_jpeg_bytes(thumb, quality=85)

        row = {
            "id": file_hash,
            "filename": filename,
            "dt": dt,
            "image_blob": image_blob,
            "thumb_blob": thumb_blob,
        }

        batch.append(row)
        count += 1

        if len(batch) >= batch_size:
            table_obj.add(batch)
            batch = []

    if batch:
        table_obj.add(batch)

    print("Done.")
    print("Images ingested:", count)

    return count


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--table", default="images")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--thumb_size", type=int, default=64)
    parser.add_argument("--dt_format", required=True)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    db = lancedb.connect(str(args.db_dir))

    # Table must already exist (create it explicitly elsewhere with a schema)
    table_obj = db.open_table(args.table)

    ingest_images_to_table(
        table_obj=table_obj,
        image_dir=args.image_dir,
        width=args.width,
        height=args.height,
        dt_format=args.dt_format,
        thumb_size=args.thumb_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
