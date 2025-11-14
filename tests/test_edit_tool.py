from tools.edit import EditTool


def test_edit_updates_existing_file(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("hello world\n")

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="world",
        new_string="there",
    )

    assert result["status"] == "success"
    assert target.read_text() == "hello there\n"


def test_edit_requires_unique_match(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("foo bar\nfoo baz\n")

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="foo",
        new_string="qux",
    )

    assert result["status"] == "error"
    assert result["content"].startswith("old_string matched multiple times")


def test_edit_replace_all(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("count = count + 1\ncount += count\n")

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="count",
        new_string="total",
        replace_all=True,
    )

    assert result["status"] == "success"
    assert result["data"]["replacements"] == 4
    assert "total = total" in target.read_text()


def test_edit_create_new_file(tmp_path):
    target = tmp_path / "new.py"

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="",
        new_string="print('hi')\n",
    )

    assert result["status"] == "success"
    assert target.read_text() == "print('hi')\n"


def test_edit_rejects_empty_old_string_for_existing_file(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha\n")

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="",
        new_string="beta",
    )

    assert result["status"] == "error"
    assert "old_string must not be empty" in result["content"]


def test_edit_allows_consecutive_edits(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha\n")

    editor = EditTool()
    first = editor.execute(
        file_path=str(target.resolve()),
        old_string="alpha",
        new_string="beta",
    )
    second = editor.execute(
        file_path=str(target.resolve()),
        old_string="beta",
        new_string="gamma",
    )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert target.read_text() == "gamma\n"
