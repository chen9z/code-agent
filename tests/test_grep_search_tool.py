from pathlib import Path

import pytest

from tools.grep import GrepSearchTool, MAX_DISPLAY_MATCHES, MAX_MATCHES


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_grep_search_basic(tmp_path, monkeypatch):
    write_file(tmp_path / "a.py", "def foo():\n    return 1\n")

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="foo")

    assert result["status"] == "success"
    matches = result["data"]["matches"]
    assert len(matches) == 1
    assert matches[0]["path"].endswith("a.py")
    assert matches[0]["line"] == 1
    assert matches[0]["match"] == "foo"
    assert result["content"].splitlines() == [
        str(tmp_path / "a.py"),
        "       1→def foo():",
        "             ^^^",
    ]
    display_text = result["data"].get("display", "")
    assert isinstance(display_text, str)
    assert "a.py" in display_text


def test_grep_search_case_insensitive(tmp_path, monkeypatch):
    write_file(tmp_path / "module.txt", "Alpha\n")

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="alpha", case_sensitive=False)

    assert len(result["data"]["matches"]) == 1
    assert "Alpha" in result["content"]


def test_grep_search_include_exclude(tmp_path, monkeypatch):
    write_file(tmp_path / "keep" / "file.txt", "value\n")
    write_file(tmp_path / "ignore" / "file.txt", "value\n")

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(
        query="value",
        include_pattern="keep/**",
        exclude_pattern="ignore/**",
    )

    assert len(result["data"]["matches"]) == 1
    assert result["data"]["matches"][0]["path"].endswith("keep/file.txt")
    assert result["content"].splitlines()[0] == str(tmp_path / "keep" / "file.txt")


def test_grep_search_truncates_results(tmp_path, monkeypatch):
    lines = "\n".join([f"match {i}" for i in range(MAX_MATCHES + 5)])
    write_file(tmp_path / "many.txt", lines)

    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="match")

    assert len(result["data"]["matches"]) == MAX_MATCHES
    assert "match" in result["content"]
    display_text = result["data"].get("display", "")
    assert isinstance(display_text, str)
    assert "+{} more matches".format(MAX_MATCHES - MAX_DISPLAY_MATCHES) in display_text
    assert "results truncated" not in display_text


def test_grep_search_invalid_regex(tmp_path, monkeypatch):
    write_file(tmp_path / "file.txt", "content\n")
    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="(")

    assert result["status"] == "error"
    assert "regex" in result["content"].lower()


def test_grep_search_no_matches(tmp_path, monkeypatch):
    write_file(tmp_path / "file.txt", "content\n")
    monkeypatch.chdir(tmp_path)

    result = GrepSearchTool().execute(query="missing")

    assert len(result["data"]["matches"]) == 0
    assert result["data"]["matches"] == []
    assert result["content"] == "[no matches]"
    assert result["data"].get("display") == "No matches"
