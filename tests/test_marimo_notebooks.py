"""Tests that all marimo notebooks (.py files under notebooks/) are structurally valid."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"


def _marimo_py_files():
    """Discover marimo notebook .py files (identified by the marimo scaffold)."""
    return sorted(
        p for p in NOTEBOOKS_DIR.rglob("*.py")
        if "app = marimo.App(" in p.read_text()
    )


MARIMO_PY_FILES = _marimo_py_files()


# ── 1. Valid Python syntax ────────────────────────────────────────────────────

@pytest.mark.parametrize("nb_path", MARIMO_PY_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_is_valid_python(nb_path):
    source = nb_path.read_text()
    try:
        ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in {nb_path.relative_to(ROOT)}: {e}")


# ── 2. Contains marimo import ─────────────────────────────────────────────────

@pytest.mark.parametrize("nb_path", MARIMO_PY_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_imports_marimo(nb_path):
    source = nb_path.read_text()
    assert "import marimo" in source, (
        f"{nb_path.relative_to(ROOT)} does not contain 'import marimo'"
    )


# ── 3. Contains marimo app scaffold ──────────────────────────────────────────

@pytest.mark.parametrize("nb_path", MARIMO_PY_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_has_app_scaffold(nb_path):
    source = nb_path.read_text()
    assert "app = marimo.App(" in source, (
        f"{nb_path.relative_to(ROOT)} is missing 'app = marimo.App(...)' scaffold"
    )
