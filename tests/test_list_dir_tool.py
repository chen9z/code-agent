from pathlib import Path

import pytest

from tools.list_dir import ListDirTool


def create_structure(root: Path) -> None:
    (root / "alpha").mkdir()
    (root / "sub").mkdir()
    (root / "file1.txt").write_text("file1\n")
    (root / "sub" / "nested.txt").write_text("nested\n")
    (root / "sub" / "deep").mkdir()
    (root / "sub" / "deep" / "final.log").write_text("final\n")


def test_list_dir_basic(tmp_path):
    create_structure(tmp_path)

    result = ListDirTool().execute(dir_path=str(tmp_path))

    assert result["dir_path"] == str(tmp_path.resolve())
    assert result["max_depth"] == 3
    assert result["entries"] == [
        "alpha/",
        "sub/",
        "file1.txt",
        "sub/deep/",
        "sub/nested.txt",
        "sub/deep/final.log",
    ]
    assert result["count"] == 6
    assert "error" not in result


def test_list_dir_respects_max_depth(tmp_path):
    create_structure(tmp_path)

    result = ListDirTool().execute(dir_path=str(tmp_path), max_depth=1)

    assert result["entries"] == [
        "alpha/",
        "sub/",
        "file1.txt",
        "sub/deep/",
        "sub/nested.txt",
    ]


def test_list_dir_with_relative_path(tmp_path, monkeypatch):
    create_structure(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = ListDirTool().execute(dir_path=".")

    assert result["entries"][0] == "alpha/"


def test_list_dir_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    result = ListDirTool().execute(dir_path=str(missing))

    assert result["error"].startswith("Directory does not exist")


def test_list_dir_negative_depth(tmp_path):
    tmp_path.mkdir(exist_ok=True)

    result = ListDirTool().execute(dir_path=str(tmp_path), max_depth=-1)

    assert result["error"] == "max_depth must be greater than or equal to 0"
