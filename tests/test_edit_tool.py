from tools.read import ReadTool
from tools.edit import EditTool


def test_edit_requires_read(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("hello world\n")

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="world",
        new_string="there",
    )

    assert result["status"] == "error"
    assert result["content"].startswith("File must be read")


def test_edit_succeeds_after_read(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("value = 1\n")
    ReadTool().execute(file_path=str(target.resolve()))

    result = EditTool().execute(
        file_path=str(target.resolve()),
        old_string="value = 1",
        new_string="value = 2",
    )

    assert result["status"] == "success"
    assert result["content"] == "ok"
    assert target.read_text() == "value = 2\n"


def test_edit_requires_unique_match(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("foo bar\nfoo baz\n")
    ReadTool().execute(file_path=str(target.resolve()))

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
    ReadTool().execute(file_path=str(target.resolve()))

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
    assert result["content"] == "ok"
    assert target.read_text() == "print('hi')\n"


def test_edit_clears_read_state(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha\n")
    reader = ReadTool()
    reader.execute(file_path=str(target.resolve()))

    editor = EditTool()
    editor.execute(
        file_path=str(target.resolve()),
        old_string="alpha",
        new_string="beta",
    )

    second = editor.execute(
        file_path=str(target.resolve()),
        old_string="beta",
        new_string="gamma",
    )

    assert second["status"] == "error"
    assert second["content"].startswith("File must be read")
