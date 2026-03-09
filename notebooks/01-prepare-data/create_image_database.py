import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Data Preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1a. Create a database for Images
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why a database for images?
    Instead of treating images as loose files in folders, we store them in a database so they become a structured, queryable dataset. This allows fast filtering by time and metadata, consistent preprocessing, reproducible experiments, and direct integration with embedding and search pipelines. The database provides indexing, versioning, and efficient storage of both image content and derived representations, turning the image collection into a reliable data asset rather than an unmanaged directory of files.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note**
    > Before running this notebook, ensure you have the variable and limits determined. See notebook <>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports
    """)
    return


@app.cell
def _():
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import lancedb
    import pyarrow as pa

    # from helpers.ingest_images import ingest_images_to_table
    from helpers.parallel_ingest_images import ingest_images_to_table

    return Path, datetime, ingest_images_to_table, json, lancedb, pa, timezone


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### User Input
    """)
    return


@app.cell
def _(Path, lancedb):
    PROJECT_ROOT = Path.cwd().parent.parent

    IMG_RAW_TBL_NAME = "era5_sample_images"

    db_dir = PROJECT_ROOT / "data" / "lancedb" / "shared_source"

    image_dir = PROJECT_ROOT / "data" / "processed_rgb"

    RESOLUTION = 256

    THUMB_RESOLUTION = 64

    JPEG_QUALITY = 90

    # Connect to DB
    db = lancedb.connect(str(db_dir))
    return (
        IMG_RAW_TBL_NAME,
        JPEG_QUALITY,
        RESOLUTION,
        THUMB_RESOLUTION,
        db,
        db_dir,
        image_dir,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Metadata
    """)
    return


@app.cell
def _(JPEG_QUALITY, RESOLUTION, THUMB_RESOLUTION, datetime, json, timezone):
    # 1. Define the metadata structure

    metadata_dict = {
        # --- 1. CORE IDENTITY ---
        "dataset_name": "ERA5 Hurricane Training Data (RGB Composites)",
        "description": "Daily weather composites (MSL Anomaly, Wind, TCWV) for hurricane detection.",
        "author": "cherukuru",
        "generated_by_script": "e5_channels.ipynb",
        "created_at": datetime.now(timezone.utc).isoformat(),  # Dynamic Timestamp
        # --- 2. SOURCE PROVENANCE (Sorted by Importance) ---
        "source_metadata": {
            "data_source": "ECMWF: https://cds.climate.copernicus.eu, Copernicus Climate Data Store",
            "conventions": "CF-1.6",
            "conversion_logic": "CISL RDA: Conversion from ECMWF GRIB1 data to netCDF4",
            "netcdf_version": "4.6.3",
            "conversion_platform": "Linux r8i6n32 4.12.14-94.41-default #1 SMP",
        },
        # --- 3. OUTPUT SPECIFICATIONS ---
        "image_specs": {
            "resolution": [RESOLUTION, RESOLUTION],
            "thumb_resolution": [THUMB_RESOLUTION, THUMB_RESOLUTION],
            "format": "JPEG",
            "quality": JPEG_QUALITY,
            "resampling": "Bilinear",
            "projection": "Plate Carrée (Equirectangular)",
        },
        # --- 4. SPATIAL & TEMPORAL BOUNDS ---
        "spatial_extent": {
            "lat_min": 15.0,
            "lat_max": 35.0,
            "lon_min": 260.0,
            "lon_max": 330.0,
            "notes": "North Atlantic / Caribbean (approx 100W to 30W)",
        },
        "temporal_extent": {
            "start": "2016-01-01",  # Updated Start
            "end": "2018-12-31",  # Updated End
            "interval": "Daily",
        },
        # --- 5. PHYSICS & CHANNELS ---
        "channels": {
            "red": {
                "variable": "MSL Pressure Anomaly",
                "range": [-20.0, 20.0],
                "unit": "hPa",
                "logic": "Inverted (Low Pressure = Bright Red)",
            },
            "green": {"variable": "10m Wind Speed (Daily Max)", "range": [0.0, 35.0], "unit": "m/s", "logic": "Linear"},
            "blue": {
                "variable": "Total Column Water Vapor (Daily Mean)",
                "range": [20.0, 70.0],
                "unit": "kg/m^2",
                "logic": "Square Root Scaled",
            },
        },
    }

    # 2. Serialize to JSON Bytes
    arrow_metadata = {b"dataset_info": json.dumps(metadata_dict).encode("utf-8"), b"version": b"1.0"}
    return (arrow_metadata,)


@app.cell
def _(IMG_RAW_TBL_NAME, arrow_metadata, db, pa):
    # Image table
    schema = pa.schema(
        [
            pa.field("id", pa.string()),  # MD5 to generate a deterministic, content-based ID
            pa.field("filename", pa.string()),
            pa.field("dt", pa.timestamp("s")),
            pa.field("image_blob", pa.binary()),
            pa.field("thumb_blob", pa.binary()),
        ],
        metadata=arrow_metadata,
    )

    if IMG_RAW_TBL_NAME in db.list_tables():
        db.drop_table(IMG_RAW_TBL_NAME)

    table = db.create_table(IMG_RAW_TBL_NAME, schema=schema)

    print(f"Table {IMG_RAW_TBL_NAME} created.")
    return (table,)


@app.cell
def _():
    # # Config Table (Global Metadata)

    # config_data = [
    #     {"key": "db_name",          "value": DB_NAME},
    #     {"key": "created_at",       "value": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
    #     {"key": "author",           "value": "cherukuru"},
    #     {"key": "project",          "value": "era5 data"},
    #     {"key": "image_width",      "value": "224"},
    #     {"key": "image_height",     "value": "224"},
    #     {"key": "thumb_width",      "value": "64"},
    #     {"key": "height",           "value": "64"},
    #     {"key": "tbl_img_raw",      "value": IMG_RAW_TBL_NAME}
    # ]

    # # Create the config table
    # # We use overwrite mode to ensure we don't have stale configs if re-running
    # if METADATA_TBL_NAME in db.table_names():
    #     db.drop_table(METADATA_TBL_NAME)

    # # Note: We let LanceDB infer the simple schema (key: str, value: str) automatically
    # # by passing the list of dicts directly.
    # config_table = db.create_table(METADATA_TBL_NAME, data=config_data)
    # print(f"Table {METADATA_TBL_NAME} created with global metadata.")
    return


@app.cell
def _():
    # ## Serial workflow
    # ingest_images_to_table(
    #     table,
    #     image_dir=image_dir,
    #     width=RESOLUTION,
    #     height=RESOLUTION,
    #     dt_format="%Y%m%d_rgb.jpeg",
    #     thumb_size=THUMB_RESOLUTION,
    #     batch_size=256
    # )
    return


@app.cell
def _(RESOLUTION, THUMB_RESOLUTION, image_dir, ingest_images_to_table, table):
    # parallel workflow

    ingest_images_to_table(
        table_obj=table,  # Open LanceDB table to write into
        image_dir=image_dir,  # Directory containing input images
        width=RESOLUTION,  # Stored image width
        height=RESOLUTION,  # Stored image height
        dt_format="%Y%m%d_rgb.jpeg",  # Datetime pattern extracted from filename
        thumb_size=THUMB_RESOLUTION,  # Square thumbnail size in pixels
        batch_size=2048,  # Rows written to DB per transaction
        workers=31,  # Number of CPU processes for image processing
        max_in_flight=31 * 16,  # Max images allowed in RAM at once (memory safety)
    )
    return


@app.cell
def _(table):
    table.schema
    return


@app.cell
def _(json, table):
    json.loads(table.schema.metadata[b"dataset_info"])
    return


@app.cell
def _(table):
    import io
    import os

    from PIL import Image

    table.search().select(["id"]).limit(10).to_pandas()

    row = table.search().limit(1000).to_pandas().iloc[999]
    img = Image.open(io.BytesIO(row["image_blob"]))

    print(img.size)
    img
    return os, row


@app.cell
def _(row):
    row["id"]
    return


@app.cell
def _(Path, db_dir, os):
    def dir_size_bytes(path: Path) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += (Path(root) / f).stat().st_size
        return total


    table_path = db_dir / "era5_sample_images.lance"


    # size_bytes = dir_size_bytes("/glade/work/ncheruku/research/bams-ai-data-exploration/data/lancedb/shared_source/era5/dinov3_image_embeddings.lance")

    size_bytes = dir_size_bytes(table_path)


    print(f"{size_bytes / 1024**2:.2f} MB")
    return


if __name__ == "__main__":
    app.run()
