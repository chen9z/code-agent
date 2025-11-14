import os

import pytest

from tools.glob import GlobTool


def test_glob_matches_mimic_find_output(tmp_path):
    tool = GlobTool()

    first = tmp_path / "first.txt"
    first.write_text("first")
    second_dir = tmp_path / "nested"
    second_dir.mkdir()
    second = second_dir / "second.txt"
    second.write_text("second")
    third = tmp_path / "third.txt"
    third.write_text("third")

    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    os.utime(third, (3, 3))

    result = tool.execute(pattern="**/*.txt", path=str(tmp_path))

    assert result["status"] == "success"
    data = result["data"]
    assert data["matches"] == [
        "third.txt",
        "nested/second.txt",
        "first.txt",
    ]
    assert data["search_path"] == str(tmp_path.resolve())
    assert "count" not in data
    assert result["content"].splitlines() == [
        str(tmp_path / "third.txt"),
        str(tmp_path / "nested" / "second.txt"),
        str(tmp_path / "first.txt"),
    ]


def test_glob_defaults_to_cwd(tmp_path, monkeypatch):
    target = tmp_path / "example.py"
    target.write_text("print('hello')\n")
    monkeypatch.chdir(tmp_path)

    result = GlobTool().execute(pattern="*.py")

    data = result["data"]
    assert data["matches"] == ["example.py"]
    assert data["search_path"] == str(tmp_path.resolve())
    assert result["content"] == str(target)


def test_glob_returns_error_payload_for_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    result = GlobTool().execute(pattern="*.py", path=str(missing))

    assert result["status"] == "error"
    assert result["content"].startswith("Search directory does not exist")
    assert result["data"]["search_path"] == str(missing)
