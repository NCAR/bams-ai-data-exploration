"""Tests for setup_experiment, load_config, upsert_config — uses tmp_path."""
import pytest

from helpers.embedding_experiment import load_config, setup_experiment, upsert_config

CONFIG_KEYS = {"created_at", "author", "source", "source_path", "tbl_img_emb", "tbl_patch_emb"}


@pytest.fixture()
def exp(tmp_path):
    """Run setup_experiment in a temp LanceDB dir and return the result dict."""
    return setup_experiment(
        project_name="test_proj",
        author="tester",
        source_uri=tmp_path / "source",
        source_table="raw_images",
        db_uri=tmp_path / "db",
    )


# ── setup_experiment ──────────────────────────────────────────────────────────

def test_setup_returns_expected_keys(exp):
    assert {"db", "config_tbl", "config_name", "img_emb_name", "patch_emb_name"} <= exp.keys()


def test_setup_config_table_name(exp):
    assert exp["config_name"] == "test_proj_config"


def test_setup_derived_table_names(exp):
    assert exp["img_emb_name"] == "test_proj_image_embeddings"
    assert exp["patch_emb_name"] == "test_proj_patch_embeddings"


def test_setup_config_has_all_keys(tmp_path, exp):
    config = load_config(tmp_path / "db", exp["config_name"])
    assert CONFIG_KEYS <= config.keys()


def test_setup_config_author(tmp_path, exp):
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["author"] == "tester"


def test_setup_config_source_table(tmp_path, exp):
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["source"] == "raw_images"


def test_setup_source_path_absolute_when_no_project_root(tmp_path):
    source = tmp_path / "somewhere" / "source"
    result = setup_experiment(
        project_name="abs_test",
        author="a",
        source_uri=source,
        source_table="tbl",
        db_uri=tmp_path / "db2",
        project_root=None,
    )
    config = load_config(tmp_path / "db2", result["config_name"])
    assert config["source_path"] == str(source)


def test_setup_source_path_relative_when_inside_project_root(tmp_path):
    project_root = tmp_path / "project"
    source = project_root / "data" / "source"
    result = setup_experiment(
        project_name="rel_test",
        author="a",
        source_uri=source,
        source_table="tbl",
        db_uri=tmp_path / "db3",
        project_root=project_root,
    )
    config = load_config(tmp_path / "db3", result["config_name"])
    assert config["source_path"] == "data/source"


def test_setup_source_path_absolute_when_outside_project_root(tmp_path):
    """Path outside project_root falls back to absolute (no ValueError crash)."""
    project_root = tmp_path / "project"
    source = tmp_path / "elsewhere" / "source"
    result = setup_experiment(
        project_name="outside_test",
        author="a",
        source_uri=source,
        source_table="tbl",
        db_uri=tmp_path / "db4",
        project_root=project_root,
    )
    config = load_config(tmp_path / "db4", result["config_name"])
    assert config["source_path"] == str(source)


def test_setup_overwrite_is_idempotent(tmp_path):
    """Calling setup_experiment twice on same name doesn't raise (mode=overwrite)."""
    kwargs = dict(
        project_name="dup",
        author="a",
        source_uri=tmp_path / "src",
        source_table="tbl",
        db_uri=tmp_path / "db5",
    )
    setup_experiment(**kwargs)
    setup_experiment(**kwargs)  # should not raise


# ── load_config ───────────────────────────────────────────────────────────────

def test_load_config_returns_dict(tmp_path, exp):
    config = load_config(tmp_path / "db", exp["config_name"])
    assert isinstance(config, dict)


def test_load_config_tbl_names_match_setup(tmp_path, exp):
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["tbl_img_emb"] == exp["img_emb_name"]
    assert config["tbl_patch_emb"] == exp["patch_emb_name"]


# ── upsert_config ─────────────────────────────────────────────────────────────

def test_upsert_updates_existing_key(tmp_path, exp):
    upsert_config(tmp_path / "db", exp["config_name"], {"author": "new_author"})
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["author"] == "new_author"


def test_upsert_adds_new_key(tmp_path, exp):
    upsert_config(tmp_path / "db", exp["config_name"], {"new_key": "new_value"})
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["new_key"] == "new_value"


def test_upsert_preserves_other_keys(tmp_path, exp):
    upsert_config(tmp_path / "db", exp["config_name"], {"author": "changed"})
    config = load_config(tmp_path / "db", exp["config_name"])
    assert "source" in config  # untouched key still present


def test_upsert_coerces_int_to_string(tmp_path, exp):
    upsert_config(tmp_path / "db", exp["config_name"], {"batch_size": 64})
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["batch_size"] == "64"


def test_upsert_multiple_keys(tmp_path, exp):
    upsert_config(tmp_path / "db", exp["config_name"], {"k1": "v1", "k2": "v2"})
    config = load_config(tmp_path / "db", exp["config_name"])
    assert config["k1"] == "v1"
    assert config["k2"] == "v2"
