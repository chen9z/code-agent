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
    print("LS content:", result["content"])

    assert result["status"] == "success"
    assert "alpha/" in result["content"]
    assert "beta.txt" in result["content"]
    assert "gamma.log" in result["content"]
    assert result["data"]["path"] == str(tmp_path.resolve())
    assert result["data"]["entries"] == ["alpha/", "beta.txt", "gamma.log"]
    assert result["data"]["directories"] == ["alpha/"]
    assert sorted(result["data"]["files"]) == ["beta.txt", "gamma.log"]
    assert len(result["data"]["entries"]) == 3


def test_ls_requires_absolute_path(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path="relative/path")

    assert result["status"] == "error"
    assert result["content"] == "path must be an absolute path"
    assert result["data"]["error"] == "path must be an absolute path"


def test_ls_handles_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    result = LSTool().execute(path=str(missing.resolve()))

    assert result["status"] == "error"
    assert result["content"].startswith("Directory does not exist")


def test_ls_errors_on_file_path(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content\n")

    result = LSTool().execute(path=str(file_path.resolve()))

    assert result["status"] == "error"
    assert result["content"].startswith("Path is not a directory")


def test_ls_applies_ignore_patterns(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path=str(tmp_path.resolve()), ignore=["*.log", "alpha"])

    assert result["status"] == "success"
    assert result["data"]["entries"] == ["beta.txt"]
    assert result["data"]["ignore"] == ["*.log", "alpha"]


def test_ls_rejects_non_string_ignore(tmp_path):
    create_sample_dir(tmp_path)

    result = LSTool().execute(path=str(tmp_path.resolve()), ignore=[123])

    assert result["status"] == "error"
    assert result["content"] == "ignore patterns must be strings"


def test_ls_permission_denied(monkeypatch, tmp_path):
    create_sample_dir(tmp_path)
    directory = tmp_path.resolve()

    def raise_permission_error(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr(Path, "iterdir", lambda self: raise_permission_error())

    result = LSTool().execute(path=str(directory))

    assert result["status"] == "error"
    assert result["content"].startswith("Permission denied listing directory")
