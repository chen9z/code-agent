from pathlib import Path

import pytest

from tools.read import ReadTool


def create_file(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_read_tool_basic(tmp_path):
    file_path = tmp_path / "sample.txt"
    create_file(file_path, ["alpha", "beta", "gamma"])

    result = ReadTool().execute(file_path=str(file_path))

    assert result["status"] == "success"
    data = result["data"]
    assert data["file_path"] == str(file_path.resolve())
    assert data["offset"] == 1
    assert data["limit"] == 2000
    assert data["count"] == 3
    assert result["content"] == "\n".join(
        [
            "     1→alpha",
            "     2→beta",
            "     3→gamma",
        ]
    )
    assert data["has_more"] is False
    assert data["truncated"] is False
    assert data["display"] == [("result", "Read 3 lines")]


def test_read_tool_offset_and_limit(tmp_path):
    file_path = tmp_path / "sample.txt"
    create_file(file_path, ["line1", "line2", "line3", "line4"])

    result = ReadTool().execute(file_path=str(file_path), offset=2, limit=2)

    assert result["status"] == "success"
    data = result["data"]
    assert data["count"] == 2
    assert result["content"].splitlines() == [
        "     2→line2",
        "     3→line3",
    ]
    assert data["has_more"] is True


def test_read_tool_default_limit_enforced(tmp_path):
    file_path = tmp_path / "many.txt"
    lines = [f"line {i}" for i in range(1, 2105)]
    create_file(file_path, lines)

    result = ReadTool().execute(file_path=str(file_path))

    data = result["data"]
    assert data["count"] == 2000
    assert data["has_more"] is True
    assert result["content"].splitlines()[0] == "     1→line 1"
    assert result["content"].splitlines()[-1] == "  2000→line 2000"


def test_read_tool_truncates_long_lines(tmp_path):
    file_path = tmp_path / "long.txt"
    long_line = "a" * 2100
    file_path.write_text(long_line + "\n")

    result = ReadTool().execute(file_path=str(file_path))

    data = result["data"]
    assert data["count"] == 1
    assert data["truncated"] is True
    assert result["content"].endswith("a" * 2000 + "… (truncated)")


def test_read_tool_errors_on_relative_path(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content\n")

    result = ReadTool().execute(file_path="relative/path.txt")

    assert result["status"] == "error"
    assert result["content"] == "file_path must be an absolute path"
    assert result["data"]["error"] == "file_path must be an absolute path"


def test_read_tool_missing_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    result = ReadTool().execute(file_path=str(file_path))

    assert result["status"] == "error"
    assert result["content"].startswith("File does not exist")
