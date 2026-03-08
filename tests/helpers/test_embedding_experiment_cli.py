"""Tests for build_cli_command() — pure string output, no I/O."""
import pytest

from helpers.embedding_experiment import build_cli_command

BASE_KWARGS = dict(
    script_path="/scripts/embed.py",
    source_db="/data/source.lance",
    source_table="images",
    config_db="/data/exp.lance",
    config_table="my_config",
    out_prefix="my_exp",
    model="ViT-L-14",
)


def build(**overrides) -> str:
    return build_cli_command(**{**BASE_KWARGS, **overrides})


# ── Required flags ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("flag,value", [
    ("--db", "/data/source.lance"),
    ("--table", "images"),
    ("--model", "ViT-L-14"),
    ("--config_db", "/data/exp.lance"),
    ("--config_table", "my_config"),
    ("--out_prefix", "my_exp"),
    ("--batch", "256"),
    ("--scan_batch", "2000"),
    ("--workers", "4"),
    ("--dtype", "fp16"),
    ("--img_id_field", "id"),
])
def test_required_flags_present(flag, value):
    cmd = build()
    assert flag in cmd
    assert value in cmd


# ── pretrained ─────────────────────────────────────────────────────────────────

def test_pretrained_included_when_set():
    cmd = build(pretrained="laion2b_s32b_b82k")
    assert "--pretrained" in cmd
    assert "laion2b_s32b_b82k" in cmd


def test_pretrained_omitted_when_none():
    cmd = build(pretrained=None)
    assert "--pretrained" not in cmd


# ── limit ─────────────────────────────────────────────────────────────────────

def test_limit_omitted_when_zero():
    cmd = build(limit=0)
    assert "--limit" not in cmd


def test_limit_included_when_positive():
    cmd = build(limit=100)
    assert "--limit" in cmd
    assert "100" in cmd


def test_limit_omitted_when_negative():
    cmd = build(limit=-1)
    assert "--limit" not in cmd


# ── image_size ─────────────────────────────────────────────────────────────────

def test_image_size_omitted_when_none():
    cmd = build(image_size=None)
    assert "--image_size" not in cmd


def test_image_size_included_when_set():
    cmd = build(image_size=224)
    assert "--image_size" in cmd
    assert "224" in cmd


# ── extra_args ─────────────────────────────────────────────────────────────────

def test_extra_args_with_dashes():
    cmd = build(extra_args={"--custom_flag": "42"})
    assert "--custom_flag" in cmd
    assert "42" in cmd


def test_extra_args_without_dashes():
    """Keys without leading -- get -- prepended."""
    cmd = build(extra_args={"custom_flag": "42"})
    assert "--custom_flag" in cmd


def test_extra_args_none_does_not_crash():
    cmd = build(extra_args=None)
    assert isinstance(cmd, str)


def test_extra_args_multiple():
    cmd = build(extra_args={"foo": "1", "bar": "2"})
    assert "--foo" in cmd
    assert "--bar" in cmd


# ── defaults ──────────────────────────────────────────────────────────────────

def test_default_batch_in_output():
    cmd = build()
    assert "--batch" in cmd
    assert "256" in cmd


def test_custom_batch_in_output():
    cmd = build(batch=64)
    assert "64" in cmd


def test_default_dtype_fp16():
    cmd = build()
    assert "fp16" in cmd
