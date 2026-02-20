import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

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
    try:
        return datetime.strptime(filename, fmt)
    except ValueError as e:
        raise ValueError(f"Failed to parse datetime from filename '{filename}' with format '{fmt}'.") from e


def _build_row(
    path_str: str,
    *,
    width: int,
    height: int,
    dt_format: Optional[str],
    thumb_size: int,
) -> dict:
    """
    Runs in a worker process. Keep it top-level (picklable).
    """
    path = Path(path_str)
    filename = path.name

    dt: Optional[datetime] = None
    if dt_format is not None:
        dt = parse_dt_from_filename(filename, dt_format)

    img = open_rgb_image(path)

    resized = resize_image(img, width, height)
    image_blob = image_to_png_bytes(resized)

    file_hash = hashlib.md5(image_blob).hexdigest()

    thumb = resize_image(img, thumb_size, thumb_size)
    thumb_blob = image_to_jpeg_bytes(thumb, quality=85)

    row = {
        "id": file_hash,
        "filename": filename,
        "image_blob": image_blob,
        "thumb_blob": thumb_blob,
    }
    if dt is not None:
        row["dt"] = dt

    return row


def ingest_images_to_table(
    table_obj,
    image_dir: Union[str, Path],
    *,
    width: int,
    height: int,
    dt_format: Optional[str] = None,
    thumb_size: int = 64,
    batch_size: int = 256,
    workers: Optional[int] = None,
    max_in_flight: int = 2048,
) -> int:
    """
    Parallelizes image processing; keeps DB writes single-threaded.

    workers: number of processes (defaults to os.cpu_count())
    max_in_flight: caps queued futures to avoid huge memory spikes


    Example:

        from helpers.ingest_images import ingest_images_to_table
        import lancedb

        db = lancedb.connect("/data/my_db")
        table = db.open_table("images")

        ingest_images_to_table(
            table_obj=table,
            image_dir="/data/images",
            width=224,
            height=224,
            dt_format="%Y%m%d_%H%M%S_rgb.jpeg",
            thumb_size=64,
            batch_size=512,
            workers=8,
            max_in_flight=512,
        )

        # Or without dt:
        ingest_images_to_table(
            table_obj=table,
            image_dir="/data/images",
            width=224,
            height=224,
            thumb_size=64,
        )
    """
    image_dir = Path(image_dir)
    paths = list_images_flat(image_dir)
    if not paths:
        raise ValueError(f"No image files found in directory: {image_dir}")

    workers = workers or (os.cpu_count() or 4)

    batch = []
    count = 0

    # Submit tasks in a controlled way to avoid too many futures / memory
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        path_iter = iter(paths)

        def submit_one(p: Path):
            return ex.submit(
                _build_row,
                str(p),
                width=width,
                height=height,
                dt_format=dt_format,
                thumb_size=thumb_size,
            )

        # prime the pump
        for _ in range(min(max_in_flight, len(paths))):
            try:
                p = next(path_iter)
            except StopIteration:
                break
            futures.append(submit_one(p))

        with tqdm(total=len(paths), desc="Ingesting (parallel)") as pbar:
            while futures:
                # as_completed yields finished futures in completion order
                for fut in as_completed(futures, timeout=None):
                    futures.remove(fut)
                    pbar.update(1)

                    try:
                        row = fut.result()
                    except Exception as e:
                        # Decide your behavior: skip or raise.
                        # For ingestion jobs, I usually log + skip.
                        print(f"[WARN] failed: {e}")
                        row = None

                    if row is not None:
                        batch.append(row)
                        count += 1

                        if len(batch) >= batch_size:
                            table_obj.add(batch)
                            batch = []

                    # keep pipeline full
                    try:
                        p = next(path_iter)
                        futures.append(submit_one(p))
                    except StopIteration:
                        pass

                    # break so we re-enter as_completed with updated list
                    break

    if batch:
        table_obj.add(batch)

    print("Done.")
    print("Images ingested:", count)
    return count
