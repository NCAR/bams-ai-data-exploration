"""Tests that README.md stays consistent with the actual project state."""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def readme_text():
    return (ROOT / "README.md").read_text()


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


# ── 1. Python version consistency ─────────────────────────────────────────────

def test_python_version_consistent(readme_text, pyproject):
    badge_match = re.search(r"python-(\d+\.\d+)-blue\.svg", readme_text)
    assert badge_match, "Python version badge not found in README"
    badge_ver = badge_match.group(1)

    python_version_file = (ROOT / ".python-version").read_text().strip()

    requires = pyproject["project"]["requires-python"]
    toml_ver = re.search(r"(\d+\.\d+)", requires).group(1)

    assert badge_ver == python_version_file, (
        f"README badge ({badge_ver}) != .python-version ({python_version_file})"
    )
    assert badge_ver == toml_ver, (
        f"README badge ({badge_ver}) != pyproject.toml requires-python ({toml_ver})"
    )


# ── 2. Badge target files exist ───────────────────────────────────────────────

def _extract_inrepo_badge_paths(readme_text):
    """Return repo-relative file paths embedded in badge URLs (blob/main/...)."""
    return re.findall(r"blob/main/([^\)\"'\s]+)", readme_text)


def test_badge_target_files_exist(readme_text):
    paths = _extract_inrepo_badge_paths(readme_text)
    assert paths, "No in-repo badge paths found — check the regex if badges were added"
    for rel in paths:
        assert (ROOT / rel).exists(), f"Badge links to missing file: {rel}"


# ── 3. Referenced directories/files exist ────────────────────────────────────

@pytest.mark.parametrize("rel_path", [
    "notebooks/01-prepare-data",
    "notebooks/02-generate-embeddings",
    "tests",
    "main.py",
])
def test_referenced_path_exists(rel_path):
    assert (ROOT / rel_path).exists(), f"README references missing path: {rel_path}"


# ── 4. Required README sections present ──────────────────────────────────────

@pytest.mark.parametrize("heading", [
    "## Getting Started",
    "## How to Review and Contribute",
    "## Development Tips",
])
def test_required_section_present(heading, readme_text):
    assert heading in readme_text, f"Required section missing from README: {heading!r}"


# ── 5. Test command matches CI ────────────────────────────────────────────────

def test_readme_contains_pytest_command(readme_text):
    assert "uv run pytest tests/" in readme_text, (
        "README should document the test command 'uv run pytest tests/'"
    )
