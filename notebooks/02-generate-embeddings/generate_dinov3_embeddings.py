import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import lancedb
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    from helpers.embedding_experiment import (
        build_cli_command,
        get_model_info,
        list_models,
        load_config,
        print_table_sizes,
        run_experiment,
        setup_experiment,
    )

    return (
        Path,
        build_cli_command,
        get_model_info,
        lancedb,
        list_models,
        load_config,
        np,
        plt,
        print_table_sizes,
        run_experiment,
        setup_experiment,
    )


@app.cell
def _(list_models):
    list_models()
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path.cwd().parent.parent

    SOURCE_URI = PROJECT_ROOT / "data" / "lancedb" / "shared_source"
    IMG_RAW_TBL_NAME = "era5_sample_images"
    DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"
    AUTHOR = "Cherukuru. N. W"
    return AUTHOR, DB_URI, IMG_RAW_TBL_NAME, PROJECT_ROOT, SOURCE_URI


@app.cell
def _(get_model_info):
    # Model-specific config — change this for a different model family
    PROJECT_NAME = "openclip"
    model_info = get_model_info(PROJECT_NAME)

    MODEL = model_info["default_model"]
    SCRIPT = model_info["script_path"]
    BATCH = model_info["default_batch"]
    WORKERS = model_info["default_workers"]
    SCAN_BATCH = model_info.get("default_scan_batch", 1000)

    print(f"Family: {PROJECT_NAME}")
    print(f"Script: {model_info['script']}")
    print(f"Model:  {MODEL}")
    return BATCH, MODEL, PROJECT_NAME, SCAN_BATCH, SCRIPT, WORKERS


@app.cell
def _(
    AUTHOR,
    DB_URI,
    IMG_RAW_TBL_NAME,
    PROJECT_NAME,
    PROJECT_ROOT,
    SOURCE_URI,
    setup_experiment,
):
    experiment = setup_experiment(
        PROJECT_NAME, AUTHOR, SOURCE_URI, IMG_RAW_TBL_NAME,
        DB_URI, project_root=PROJECT_ROOT,
    )
    return (experiment,)


@app.cell
def _(
    BATCH,
    DB_URI,
    IMG_RAW_TBL_NAME,
    MODEL,
    PROJECT_NAME,
    SCAN_BATCH,
    SCRIPT,
    SOURCE_URI,
    WORKERS,
    experiment,
    run_experiment,
):
    # Case 1: Run inline (interactive notebook workflow)
    run_experiment(
        SCRIPT, SOURCE_URI, IMG_RAW_TBL_NAME,
        DB_URI, experiment["config_name"], PROJECT_NAME,
        MODEL, batch=BATCH, scan_batch=SCAN_BATCH, workers=WORKERS,
    )
    return


@app.cell
def _(
    BATCH,
    DB_URI,
    IMG_RAW_TBL_NAME,
    MODEL,
    PROJECT_NAME,
    SCAN_BATCH,
    SCRIPT,
    SOURCE_URI,
    WORKERS,
    build_cli_command,
    experiment,
):
    # Case 2: Build CLI command for PBS / external job submission
    cmd = build_cli_command(
        SCRIPT, SOURCE_URI, IMG_RAW_TBL_NAME,
        DB_URI, experiment["config_name"], PROJECT_NAME,
        MODEL, batch=BATCH, scan_batch=SCAN_BATCH, workers=WORKERS,
    )
    print(cmd)
    return


@app.cell
def _(DB_URI, experiment, load_config):
    # Inspect config after run completes (works for both Case 1 and Case 2)
    config = load_config(DB_URI, experiment["config_name"])
    config
    return


@app.cell
def _(DB_URI, experiment, print_table_sizes):
    print_table_sizes(
        DB_URI,
        experiment["config_name"],
        experiment["img_emb_name"],
        experiment["patch_emb_name"],
    )
    return


@app.cell
def _(DB_URI, experiment, lancedb):
    db = lancedb.connect(str(DB_URI))
    patch_tbl = db.open_table(experiment["patch_emb_name"])
    return db, patch_tbl


@app.cell
def _(db, experiment):
    img_tbl = db.open_table(experiment["img_emb_name"])
    return (img_tbl,)


@app.cell
def _(np, plt):
    def preview_image_embedding(tbl, idx: int = 0) -> None:
        """Preview a single row from the image embeddings table by index.

        Prints a column summary and renders two plots:
          - Left:  histogram of the embedding vector values
          - Right: attention map reshaped to its spatial grid (e.g. 16×16)

        Parameters
        ----------
        tbl : LanceDB Table
            The image embeddings table (must have columns: image_id, embedding, attention_map).
        idx : int
            Row index to inspect.
        """
        row = tbl.to_lance().take([idx]).to_pydict()

        image_id = row["image_id"][0]
        emb = np.array(row["embedding"][0], dtype=np.float32)
        attn = np.array(row["attention_map"][0], dtype=np.float32)

        # Auto-detect spatial grid from attention map length
        spatial = int(round(len(attn) ** 0.5))
        attn_2d = attn.reshape(spatial, spatial)

        # ── Column summary ──────────────────────────────────────────────────
        print(f"idx         : {idx}")
        print(f"image_id    : {image_id}")
        print(f"embedding   : dim={len(emb)}, norm={np.linalg.norm(emb):.6f}, "
              f"min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")
        print(f"attention   : patches={len(attn)} ({spatial}×{spatial}), "
              f"min={attn.min():.4f}, max={attn.max():.4f}, sum={attn.sum():.4f}")

        # ── Plots ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"idx={idx}  |  {image_id}", fontsize=10)

        # Embedding histogram
        axes[0].hist(emb, bins=64, color="steelblue", edgecolor="none")
        axes[0].axvline(emb.mean(), color="tomato", linestyle="--",
                        label=f"mean = {emb.mean():.3f}")
        axes[0].set_title(f"Embedding  (dim={len(emb)}, ‖v‖={np.linalg.norm(emb):.4f})")
        axes[0].set_xlabel("value")
        axes[0].set_ylabel("count")
        axes[0].legend(fontsize=8)

        # Attention map heatmap
        im = axes[1].imshow(attn_2d, cmap="inferno", interpolation="nearest")
        axes[1].set_title(f"Attention map  ({spatial}×{spatial})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return (preview_image_embedding,)


@app.cell
def _(img_tbl, preview_image_embedding):
    preview_image_embedding(img_tbl, idx=789)
    return


@app.cell
def _(patch_tbl):
    patch_tbl.schema
    return


@app.cell
def _(patch_tbl):
    row = patch_tbl.search().limit(1).to_pandas().iloc[0]
    row["image_id"]
    return


@app.cell
def _():
    import torch

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    return


@app.cell
def _(patch_tbl):
    patch_tbl.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=128, num_sub_vectors=96, accelerator="cuda", vector_column_name="embedding")
    return


if __name__ == "__main__":
    app.run()
