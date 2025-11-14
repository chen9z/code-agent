from pathlib import Path

import pytest

from tools.read import DEFAULT_LIMIT, MAX_LINE_LENGTH, ReadTool, format_line_with_arrow


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
    assert data["limit"] == DEFAULT_LIMIT
    assert data["count"] == 3
    assert result["content"] == "\n".join(
        [
            format_line_with_arrow(1, "alpha"),
            format_line_with_arrow(2, "beta"),
            format_line_with_arrow(3, "gamma"),
        ]
    )
    assert data["has_more"] is False
    assert data["display"] == "Read 3 lines"


def test_read_tool_offset_and_limit(tmp_path):
    file_path = tmp_path / "sample.txt"
    create_file(file_path, ["line1", "line2", "line3", "line4"])

    result = ReadTool().execute(file_path=str(file_path), offset=2, limit=2)

    assert result["status"] == "success"
    data = result["data"]
    assert data["count"] == 2
    assert result["content"].splitlines() == [
        format_line_with_arrow(2, "line2"),
        format_line_with_arrow(3, "line3"),
    ]
    assert data["has_more"] is True


def test_read_tool_reads_entire_file_by_default(tmp_path):
    file_path = tmp_path / "many.txt"
    lines = [f"line {i}" for i in range(1, 2105)]
    create_file(file_path, lines)

    result = ReadTool().execute(file_path=str(file_path))

    data = result["data"]
    assert data["has_more"] is True
    assert result["content"].splitlines()[0] == format_line_with_arrow(1, "line 1")
    assert result["content"].splitlines()[-1] == format_line_with_arrow(DEFAULT_LIMIT, f"line {DEFAULT_LIMIT}")


def test_read_tool_truncates_long_lines(tmp_path):
    file_path = tmp_path / "long.txt"
    long_line = "a" * 2100
    file_path.write_text(long_line + "\n")

    result = ReadTool().execute(file_path=str(file_path))

    data = result["data"]
    assert data["count"] == 1
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
