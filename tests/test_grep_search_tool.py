from pathlib import Path

import pytest

from tools.grep import GrepSearchTool, MAX_MATCHES


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_grep_search_basic(tmp_path, monkeypatch):
    write_file(tmp_path / "a.py", "def foo():\n    return 1\n")

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="foo")

    assert result["count"] == 1
    assert result["matches"][0]["path"].endswith("a.py")
    assert result["matches"][0]["line"] == 1
    assert result["matches"][0]["match"] == "foo"
    assert result["truncated"] is False
    assert result["result"].splitlines() == [
        str(tmp_path / "a.py"),
        "       1→def foo():",
        "             ^^^",
    ]


def test_grep_search_case_insensitive(tmp_path, monkeypatch):
    write_file(tmp_path / "module.txt", "Alpha\n")

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="alpha", case_sensitive=False)

    assert result["count"] == 1
    assert "Alpha" in result["result"]


def test_grep_search_include_exclude(tmp_path, monkeypatch):
    write_file(tmp_path / "keep" / "file.txt", "value\n")
    write_file(tmp_path / "ignore" / "file.txt", "value\n")

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(
        query="value",
        include_pattern="keep/**",
        exclude_pattern="ignore/**",
    )

    assert result["count"] == 1
    assert result["matches"][0]["path"].endswith("keep/file.txt")
    assert result["result"].splitlines()[0] == str(tmp_path / "keep" / "file.txt")


def test_grep_search_truncates_results(tmp_path, monkeypatch):
    lines = "\n".join([f"match {i}" for i in range(MAX_MATCHES + 5)])
    write_file(tmp_path / "many.txt", lines)

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="match")

    assert result["count"] == MAX_MATCHES
    assert result["truncated"] is True
    assert "match" in result["result"]


def test_grep_search_invalid_regex(tmp_path, monkeypatch):
    write_file(tmp_path / "file.txt", "content\n")
    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="(")

    assert "error" in result
    assert "regex" in result["error"].lower()


def test_grep_search_no_matches(tmp_path, monkeypatch):
    write_file(tmp_path / "file.txt", "content\n")
    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="missing")

    assert result["count"] == 0
    assert result["matches"] == []
    assert result["truncated"] is False
    assert result["result"] == "[no matches]"
