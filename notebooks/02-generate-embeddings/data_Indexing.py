import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import lancedb

    return (lancedb,)


@app.function
def get_metadata_value(table, key_name, value_column="value"):
    """
    Retrieves a single value from a LanceDB table given a key.

    Args:
        table: The opened LanceDB table object.
        key_name: The string key to look for (e.g., 'tbl_img_emb').
        value_column: The name of the column containing the data.
    """
    # Filter for the key and select only the necessary column
    result = table.search().where(f"key='{key_name}'").select([value_column]).to_pandas()

    if not result.empty:
        return result[value_column].iloc[0]
    return None


@app.cell
def _(lancedb):
    embedding_db_path = "/glade/work/ncheruku/research/bams-ai-data-exploration/data/lancedb/experiments/era5"
    source_img_path = "/glade/work/ncheruku/research/bams-ai-data-exploration/data/lancedb/shared_source"

    project_name = "dinov3"
    src_img_tbl_name = "era5_sample_images"

    # Connect and open table
    db = lancedb.connect(embedding_db_path)
    source_db = lancedb.connect(source_img_path)
    config_tbl = db.open_table(project_name + "_config")
    _img_emb_tbl = db.open_table(get_metadata_value(config_tbl, "img_emb_table_current"))
    patch_emb_tbl = db.open_table(get_metadata_value(config_tbl, "patch_emb_table_current"))
    src_img_tbl = source_db.open_table(src_img_tbl_name)

    df = patch_emb_tbl.to_pandas()
    df
    # patch_emb_tbl.count_rows()
    return df, patch_emb_tbl, src_img_tbl


@app.cell
def _(src_img_tbl):
    src_img_tbl.schema
    # patch_emb_tbl= db.open_table(get_metadata_value(config_tbl, "patch_emb_table_current"))
    return


@app.cell
def _(patch_emb_tbl):
    patch_emb_tbl.schema
    return


@app.cell
def _():
    # CPU based

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.decomposition import PCA

    # # Materialize embeddings to a contiguous float32 matrix
    # X = np.asarray(df["embedding"].to_list(), dtype=np.float32)  # shape (N, 768)

    # pca = PCA(n_components=768, svd_solver="randomized", random_state=0)
    # pca.fit(X)

    # cum = np.cumsum(pca.explained_variance_ratio_)

    # plt.figure()
    # plt.plot(np.arange(1, 769), cum)
    # plt.xlabel("Number of principal components")
    # plt.ylabel("Cumulative explained variance ratio")
    # plt.ylim(0, 1.01)
    # plt.grid(True)

    # for t in [0.80, 0.90, 0.95, 0.99]:
    #     k = int(np.searchsorted(cum, t) + 1)
    #     plt.axhline(t, linestyle="--",color="red" )
    #     plt.text(768 * 0.65, t + 0.01, f"{int(t*100)}% @ {k} PCs")

    # plt.show()
    return


@app.cell
def _(df):
    # GPU based — requires RAPIDS/CUDA (cuml)
    import matplotlib.pyplot as plt
    import numpy as np
    from cuml.decomposition import PCA
    X = np.asarray(df['embedding'].to_list(), dtype=np.float32)
    _pca = PCA(n_components=768, output_type='numpy')
    # Convert embeddings → contiguous float32 matrix
    _pca.fit(X)
    cum = np.cumsum(_pca.explained_variance_ratio_)
    # GPU PCA
    plt.figure()
    plt.plot(np.arange(1, 769), cum)
    plt.xlabel('Number of principal components')
    # Cumulative explained variance
    plt.ylabel('Cumulative explained variance ratio')
    plt.ylim(0, 1.01)
    # Plot
    plt.grid(True)
    for t in [0.8, 0.9, 0.95, 0.99]:
        k = int(np.searchsorted(cum, t) + 1)
        plt.axhline(t, linestyle='--', color='red')
        plt.text(768 * 0.65, t + 0.01, f'{int(t * 100)}% @ {k} PCs')
    plt.show()
    return (np,)


@app.cell
def _(df, np):
    X_1 = np.asarray(df['embedding'].to_list(), dtype=np.float32)  # (N, 768)
    _patch_ids = df['patch_id'].to_numpy()
    return (X_1,)


@app.cell
def _(X_1, np):
    # requires RAPIDS/CUDA
    from cuml.decomposition import PCA as cuPCA
    _pca = cuPCA(n_components=320, output_type='numpy')
    X363 = _pca.fit_transform(X_1).astype(np.float32)
    return (X363,)


@app.cell
def _(X363, np):
    # requires RAPIDS/CUDA
    import pandas as pd
    from cuml.manifold import UMAP as cuUMAP

    umap = cuUMAP(
        n_components=2,
        n_neighbors=30,  # tweak
        min_dist=0.05,  # tweak
        metric="cosine",
        output_type="numpy",
        random_state=42,
    )
    XY = umap.fit_transform(X363).astype(np.float32)  # (N, 2)
    return XY, pd


@app.cell
def _(pd, src_img_tbl):
    # Convert to Arrow, select columns, then to Pandas
    img_df = src_img_tbl.to_arrow().select(["id", "dt"]).to_pandas()

    # Ensure dtype and build lookup
    img_df["dt"] = pd.to_datetime(img_df["dt"])

    image_to_dt = dict(zip(img_df["id"], img_df["dt"], strict=False), strict=False)
    return (image_to_dt,)


@app.cell
def _(XY, patch_emb_tbl):
    patch_df = patch_emb_tbl.to_arrow().select(["image_id"]).to_pandas()

    assert len(patch_df) == len(XY)
    return (patch_df,)


@app.cell
def _(image_to_dt, patch_df, pd):
    image_ids = patch_df["image_id"].values

    # Vectorized mapping
    dt_for_point = pd.Series(image_ids).map(image_to_dt).values
    return (dt_for_point,)


@app.cell
def _(dt_for_point, pd):
    dt_series = pd.to_datetime(dt_for_point)

    month_for_point = dt_series.month.astype("int8").values
    doy_for_point = dt_series.dayofyear.astype("int16").values
    return doy_for_point, month_for_point


@app.cell
def _(XY, doy_for_point, month_for_point, pd):
    df_xy = pd.DataFrame(
        {
            "x": XY[:, 0],
            "y": XY[:, 1],
            "month": pd.Categorical(month_for_point),
            "doy": doy_for_point,
        }
    )
    return (df_xy,)


@app.cell
def _(df_xy):
    import datashader as ds
    import holoviews as hv
    import panel as pn
    from holoviews.operation.datashader import datashade, dynspread
    hv.extension('bokeh')
    pts = hv.Points(df_xy, kdims=['x', 'y'], vdims=['month', 'doy'])

    def make_plot(mode):
        # Points object
        if mode == 'Month':
            color_key_month = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f', 9: '#bcbd22', 10: '#17becf', 11: '#aec7e8', 12: '#ffbb78'}
            shaded = datashade(pts, aggregator=ds.count_cat('month'), color_key=color_key_month, width=3000, height=3000)
            title = 'UMAP Projection — Colored by Month'
        else:
            shaded = datashade(pts, aggregator=ds.mean('doy'), cmap='viridis', width=2000, height=2000)
            title = 'UMAP Projection — Colored by Day of Year'
        return dynspread(shaded).opts(width=1000, height=1000, bgcolor='#0e0e0e', show_grid=True, gridstyle={'grid_line_color': '#666666', 'grid_line_alpha': 0.6}, xlabel='UMAP-1', ylabel='UMAP-2', title=title)
    month_plot = make_plot('Month')
    doy_plot = make_plot('Day of Year')
    layout = pn.Tabs(('Month', month_plot), ('DOY', doy_plot))
    return (layout,)


if __name__ == "__main__":
    app.run()
