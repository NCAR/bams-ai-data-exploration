"""Tests for format_bytes() and dir_size_bytes() — pure/filesystem helpers."""
import pytest

from helpers.embedding_experiment import dir_size_bytes, format_bytes


# ── format_bytes ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [
    (0, "0.00 B"),
    (1, "1.00 B"),
    (1023, "1023.00 B"),
    (1024, "1.00 KB"),
    (1024 * 1024, "1.00 MB"),
    (1024 ** 3, "1.00 GB"),
    (1024 ** 4, "1.00 TB"),
    (1024 ** 5, "1.00 PB"),
])
def test_format_bytes_boundary_values(n, expected):
    assert format_bytes(n) == expected


def test_format_bytes_fractional_gb():
    n = int(1.5 * 1024 ** 3)
    result = format_bytes(n)
    assert result.endswith("GB")
    assert result.startswith("1.5")


def test_format_bytes_returns_string():
    assert isinstance(format_bytes(512), str)


# ── dir_size_bytes ────────────────────────────────────────────────────────────

def test_dir_size_bytes_empty_dir(tmp_path):
    assert dir_size_bytes(tmp_path) == 0


def test_dir_size_bytes_single_file(tmp_path):
    f = tmp_path / "file.bin"
    f.write_bytes(b"x" * 100)
    assert dir_size_bytes(tmp_path) == 100


def test_dir_size_bytes_multiple_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 200)
    (tmp_path / "b.bin").write_bytes(b"x" * 300)
    assert dir_size_bytes(tmp_path) == 500


def test_dir_size_bytes_nested(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.bin").write_bytes(b"x" * 100)
    (sub / "nested.bin").write_bytes(b"x" * 150)
    assert dir_size_bytes(tmp_path) == 250


def test_dir_size_bytes_accepts_string_path(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"x" * 50)
    assert dir_size_bytes(str(tmp_path)) == 50
