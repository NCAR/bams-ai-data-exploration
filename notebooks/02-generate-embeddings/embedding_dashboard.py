import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import lancedb
    import io
    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    return lancedb, mo, pd, plt


@app.function
def list_experiments(db_path: str) -> list:
    """Scan a LanceDB directory for experiment names by finding *_config.lance dirs."""
    from pathlib import Path
    p = Path(db_path)
    if not p.exists() or not p.is_dir():
        return []
    return [
        d.name[: -len("_config.lance")]
        for d in sorted(p.iterdir())
        if d.is_dir() and d.name.endswith("_config.lance")
    ]


@app.function
def load_config_dict(db, config_table_name: str) -> dict:
    """Load a config table into a Python dict keyed by the 'key' column."""
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()
    return dict(zip(df["key"], df["value"]))


@app.function
def resolve_source_path(experiments_db_path: str, source_path_from_config: str) -> str:
    """
    Resolve source_path from config to an absolute path.
    - If already absolute and exists → return as-is
    - If relative → walk up from experiments_db_path until the join exists
    - Returns None if unresolvable
    """
    from pathlib import Path
    p = Path(source_path_from_config)
    if p.is_absolute():
        return str(p) if p.exists() else None
    candidate = Path(experiments_db_path)
    for _ in range(10):
        candidate = candidate.parent
        resolved = candidate / source_path_from_config
        if resolved.exists():
            return str(resolved)
    return None


@app.function
def fetch_image_by_filename(src_img_tbl, filename: str):
    """Load a PIL image from src_img_tbl by filename."""
    import io
    from PIL import Image
    row = src_img_tbl.search().where(f"filename = '{filename}'").limit(1).to_pandas().iloc[0]
    return Image.open(io.BytesIO(row["image_blob"])).convert("RGBA")


@app.function
def overlay_grid(img, patch_size: int = 16, color=(255, 255, 255, 50), width: int = 1):
    """Draw a patch grid overlay on a PIL RGBA image."""
    from PIL import Image, ImageDraw
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for x in range(0, w + 1, patch_size):
        d.line([(x, 0), (x, h)], fill=color, width=width)
    for y in range(0, h + 1, patch_size):
        d.line([(0, y), (w, y)], fill=color, width=width)
    return Image.alpha_composite(img, overlay)


@app.function
def patch_box_from_index(patch_index: int, img_w: int = 256, img_h: int = 256, patch_size: int = 16):
    """Return (x0, y0, x1, y1) bounding box for a patch index."""
    grid_w = img_w // patch_size
    grid_h = img_h // patch_size
    n_patches = grid_w * grid_h
    if patch_index < 0 or patch_index >= n_patches:
        raise ValueError(f"patch_index {patch_index} out of range [0, {n_patches - 1}]")
    row = patch_index // grid_w
    col = patch_index % grid_w
    x0 = col * patch_size
    y0 = row * patch_size
    return (x0, y0, x0 + patch_size, y0 + patch_size)


@app.function
def highlight_patch(img, box, fill=(255, 0, 0, 0), outline=(255, 0, 0, 220), outline_width: int = 2):
    """Draw a highlighted rectangle over a patch on a PIL RGBA image."""
    from PIL import Image, ImageDraw
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle(box, fill=fill, outline=outline, width=outline_width)
    return Image.alpha_composite(img, overlay)


@app.function
def make_extent_map(lat_min, lat_max, lon_min, lon_max, num_patch_tokens, patch_size=16, theme="light", experiment=""):
    """Cartopy map cropped to spatial extent with patch grid line overlay."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import math

    is_dark = theme == "dark"
    colors = (
        {"land": "#3a3a3a", "ocean": "#1e3a5f", "coast": "#aaaaaa",
         "grid": "#666666", "text": "#e0e0e0", "bg": "#1a1a1a"}
        if is_dark else
        {"land": "#d4d4d4", "ocean": "#a8c8e8", "coast": "#555555",
         "grid": "#888888", "text": "#222222", "bg": "#ffffff"}
    )

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj})
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    # Crop exactly to extent — no padding
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Geographic features
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=colors["ocean"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),  facecolor=colors["land"],  zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   edgecolor=colors["coast"], linewidth=0.8, zorder=2)

    # Lat/lon axis labels — no grid lines
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": colors["text"], "fontsize": 8}
    gl.ylabel_style = {"color": colors["text"], "fontsize": 8}

    # Patch grid lines only — no fill
    n_side = int(math.sqrt(num_patch_tokens))
    img_size = n_side * patch_size
    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side
    for i in range(1, n_side):
        ax.plot([lon_min, lon_max], [lat_min + i * lat_step] * 2,
                transform=proj, color=colors["grid"], linewidth=0.4, zorder=3)
    for j in range(1, n_side):
        ax.plot([lon_min + j * lon_step] * 2, [lat_min, lat_max],
                transform=proj, color=colors["grid"], linewidth=0.4, zorder=3)

    _title = f"{img_size}×{img_size}px  |  {n_side}×{n_side} patch grid ({num_patch_tokens} patches)"
    if experiment:
        _title = f"{experiment}  —  {_title}"
    ax.set_title(_title, color=colors["text"], fontsize=10)
    fig.tight_layout()
    return fig


@app.cell
def _(mo):
    map_theme = mo.ui.radio(
        options=["system", "light", "dark"],
        value=mo.app_meta().theme,
        label="Map theme",
        inline=True,
    )
    map_theme
    return (map_theme,)


@app.cell
def _(mo):
    embedding_db_path = mo.ui.text(
        value="",
        placeholder="e.g. /data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
    embedding_db_path
    return (embedding_db_path,)


@app.cell
def _(embedding_db_path, mo):
    _experiments = list_experiments(embedding_db_path.value)

    if not _experiments:
        experiment_selector = mo.ui.dropdown(options=[], label="Experiment")
        mo.stop(True, mo.callout(
            mo.md(f"No experiments found at `{embedding_db_path.value}`"),
            kind="warn",
        ))
    else:
        experiment_selector = mo.ui.dropdown(
            options=_experiments,
            value=_experiments[0],
            label="Experiment",
        )

    experiment_selector
    return (experiment_selector,)


@app.cell
def _(embedding_db_path, experiment_selector, lancedb, mo):
    mo.stop(experiment_selector.value is None, mo.callout(
        mo.md("Select an experiment above."), kind="info"
    ))

    _exp = experiment_selector.value
    db = lancedb.connect(embedding_db_path.value)

    config = load_config_dict(db, f"{_exp}_config")

    _raw_source_path = config.get("source_path", "")
    _source_path = resolve_source_path(embedding_db_path.value, _raw_source_path)
    mo.stop(_source_path is None, mo.callout(
        mo.md(f"Could not resolve source DB path from config `source_path`: `{_raw_source_path}`"),
        kind="danger",
    ))

    source_db = lancedb.connect(_source_path)
    img_emb_tbl = db.open_table(config["tbl_img_emb"])
    patch_emb_tbl = db.open_table(config["tbl_patch_emb"])
    src_img_tbl = source_db.open_table(config["source"])
    return config, img_emb_tbl, patch_emb_tbl, src_img_tbl


@app.cell
def _(
    config,
    experiment_selector,
    img_emb_tbl,
    map_theme,
    mo,
    patch_emb_tbl,
    pd,
    plt,
    src_img_tbl,
):
    import json

    def scroll(tbl, height="240px"):
        return mo.Html(
            f'<div style="max-height:{height};overflow-y:auto">'
            + mo.as_html(tbl).text
            + "</div>"
        )

    def schema_df(tbl):
        return pd.DataFrame([{"Column": f.name, "Type": str(f.type)} for f in tbl.schema])

    # Source tab: nested tabs — Metadata first, Columns second
    _raw_meta = src_img_tbl.schema.metadata or {}
    _dataset_info = json.loads(_raw_meta.get(b"dataset_info", "{}"))

    def flatten(d, prefix=""):
        rows = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(flatten(v, key))
            else:
                rows.append({"Key": key, "Value": str(v)})
        return rows

    _src_tab = mo.ui.tabs({
        "Metadata": scroll(mo.ui.table(pd.DataFrame(flatten(_dataset_info)), selection=None))
        if _dataset_info
        else mo.md("*(no schema metadata)*"),
        "Variables": scroll(mo.ui.table(schema_df(src_img_tbl), selection=None)),
    })

    # Index info helper
    def index_info(tbl):
        indices = list(tbl.list_indices())
        if not indices:
            return mo.md("*(no indexes)*")
        rows = []
        for idx in indices:
            stats = tbl.index_stats(idx.name)
            row = {
                "Name": idx.name,
                "Type": getattr(idx, "index_type", getattr(idx, "type", "?")),
                "Variables": ", ".join(idx.columns),
            }
            if stats is not None:
                row["Indexed rows"] = getattr(stats, "num_indexed_rows", "—")
                row["Unindexed rows"] = getattr(stats, "num_unindexed_rows", "—")
            rows.append(row)
        return mo.ui.table(pd.DataFrame(rows), selection=None)

    # Experiment tab: nested tabs for Config / Image Embeddings / Patch Embeddings
    _img_n = img_emb_tbl.count_rows()
    _patch_n = patch_emb_tbl.count_rows()
    _cfg_df = pd.DataFrame(list(config.items()), columns=["Key", "Value"])
    _exp_tab = mo.ui.tabs({
        "Config": scroll(mo.ui.table(_cfg_df, selection=None)),
        f"Image Embeddings ({_img_n:,})": mo.vstack([
            mo.md("### Schema"),
            scroll(mo.ui.table(schema_df(img_emb_tbl), selection=None)),
            mo.md("### Indexes"),
            index_info(img_emb_tbl),
        ]),
        f"Patch Embeddings ({_patch_n:,})": mo.vstack([
            mo.md("### Schema"),
            scroll(mo.ui.table(schema_df(patch_emb_tbl), selection=None)),
            mo.md("### Indexes"),
            index_info(patch_emb_tbl),
        ]),
    })

    # Map tab — spatial extent + patch grid
    _ext = _dataset_info.get("spatial_extent", {})
    if _ext and "num_patch_tokens" in config:
        _map_fig = make_extent_map(
            lat_min=float(_ext["lat_min"]), lat_max=float(_ext["lat_max"]),
            lon_min=float(_ext["lon_min"]), lon_max=float(_ext["lon_max"]),
            num_patch_tokens=int(config["num_patch_tokens"]),
            patch_size=int(config.get("patch_size", 16)),
            theme=map_theme.value,
            experiment=experiment_selector.value,
        )
        _map_tab = mo.as_html(_map_fig)
        plt.close(_map_fig)
    else:
        _map_tab = mo.callout(
            mo.md("Spatial extent or `num_patch_tokens` not available."), kind="warn"
        )

    mo.ui.tabs({"Source": _src_tab, "Experiment": _exp_tab, "Map": _map_tab})
    return


if __name__ == "__main__":
    app.run()
