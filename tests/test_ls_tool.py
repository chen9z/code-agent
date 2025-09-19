from pathlib import Path

import pytest

from tools.ls import LSTool


def create_sample_dir(root: Path) -> None:
    (root / "alpha").mkdir()
    (root / "beta.txt").write_text("beta\n")
    (root / "gamma.log").write_text("gamma\n")
    (root / "alpha" / "nested.txt").write_text("nested\n")


def test_ls_lists_entries(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path=str(tmp_path.resolve()))

    assert result["path"] == str(tmp_path.resolve())
    assert result["entries"] == ["alpha/", "beta.txt", "gamma.log"]
    assert result["directories"] == ["alpha/"]
    assert sorted(result["files"]) == ["beta.txt", "gamma.log"]
    assert result["count"] == 3


def test_ls_requires_absolute_path(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path="relative/path")

    assert result["error"] == "path must be an absolute path"


def test_ls_handles_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    result = LSTool().execute(path=str(missing.resolve()))

    assert result["error"].startswith("Directory does not exist")


def test_ls_errors_on_file_path(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content\n")

    result = LSTool().execute(path=str(file_path.resolve()))

    assert result["error"].startswith("Path is not a directory")


def test_ls_applies_ignore_patterns(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path=str(tmp_path.resolve()), ignore=["*.log", "alpha"])

    assert result["entries"] == ["beta.txt"]
    assert result["ignore"] == ["*.log", "alpha"]


def test_ls_rejects_non_string_ignore(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path=str(tmp_path.resolve()), ignore=[123])

    assert result["error"] == "ignore patterns must be strings"


def test_ls_permission_denied(monkeypatch, tmp_path):
    create_sample_dir(tmp_path)
    directory = tmp_path.resolve()

    def raise_permission_error(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr(Path, "iterdir", lambda self: raise_permission_error())

    result = LSTool().execute(path=str(directory))

    assert result["error"].startswith("Permission denied listing directory")
