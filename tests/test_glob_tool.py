from pathlib import Path

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

    result = tool.execute(pattern="**/*.txt", path=str(tmp_path))

    assert result["matches"] == [
        "first.txt",
        "nested/second.txt",
        "third.txt",
    ]
    assert result["search_path"] == str(tmp_path.resolve())
    assert result["count"] == 3
    assert result["content"].splitlines() == [
        str(tmp_path / "first.txt"),
        str(tmp_path / "nested" / "second.txt"),
        str(tmp_path / "third.txt"),
    ]


def test_glob_defaults_to_cwd(tmp_path, monkeypatch):
    target = tmp_path / "example.py"
    target.write_text("print('hello')\n")
    monkeypatch.chdir(tmp_path)

    result = GlobTool().execute(pattern="*.py")

    assert result["matches"] == ["example.py"]
    assert result["search_path"] == str(tmp_path.resolve())
    assert result["content"] == str(target)


def test_glob_returns_error_payload_for_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    result = GlobTool().execute(pattern="*.py", path=str(missing))

    assert result["error"].startswith("Search directory does not exist")
    assert result["search_path"] == str(missing)
