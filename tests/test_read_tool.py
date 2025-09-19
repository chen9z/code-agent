from pathlib import Path

import pytest

from tools.read import ReadTool


def create_file(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_read_tool_basic(tmp_path):
    file_path = tmp_path / "sample.txt"
    create_file(file_path, ["alpha", "beta", "gamma"])

    result = ReadTool().execute(file_path=str(file_path))

    assert result["file_path"] == str(file_path.resolve())
    assert result["offset"] == 1
    assert result["limit"] == 2000
    assert result["count"] == 3
    assert result["result"] == "\n".join(
        [
            "     1\talpha",
            "     2\tbeta",
            "     3\tgamma",
        ]
    )
    assert result["has_more"] is False
    assert result["truncated"] is False


def test_read_tool_offset_and_limit(tmp_path):
    file_path = tmp_path / "sample.txt"
    create_file(file_path, ["line1", "line2", "line3", "line4"])

    result = ReadTool().execute(file_path=str(file_path), offset=2, limit=2)

    assert result["count"] == 2
    assert result["result"].splitlines() == [
        "     2\tline2",
        "     3\tline3",
    ]
    assert result["has_more"] is True


def test_read_tool_default_limit_enforced(tmp_path):
    file_path = tmp_path / "many.txt"
    lines = [f"line {i}" for i in range(1, 2105)]
    create_file(file_path, lines)

    result = ReadTool().execute(file_path=str(file_path))

    assert result["count"] == 2000
    assert result["has_more"] is True
    assert result["result"].splitlines()[0] == "     1\tline 1"
    assert result["result"].splitlines()[-1] == "  2000\tline 2000"


def test_read_tool_truncates_long_lines(tmp_path):
    file_path = tmp_path / "long.txt"
    long_line = "a" * 2100
    file_path.write_text(long_line + "\n")

    result = ReadTool().execute(file_path=str(file_path))

    assert result["count"] == 1
    assert result["truncated"] is True
    assert result["result"].endswith("a" * 2000 + "… (truncated)")


def test_read_tool_errors_on_relative_path(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content\n")

    result = ReadTool().execute(file_path="relative/path.txt")

    assert "error" in result
    assert result["error"] == "file_path must be an absolute path"


def test_read_tool_missing_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    result = ReadTool().execute(file_path=str(file_path))

    assert result["error"].startswith("File does not exist")
