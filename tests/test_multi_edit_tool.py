from tools.read import ReadTool
from tools.multi_edit import MultiEditTool


def test_multi_edit_sequence(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("value = 1\nprint(value)\n")
    ReadTool().execute(file_path=str(target.resolve()))

    result = MultiEditTool().execute(
        file_path=str(target.resolve()),
        edits=[
            {"old_string": "value", "new_string": "total", "replace_all": True},
            {"old_string": "total = 1", "new_string": "total = 2"},
        ],
    )

    assert result["result"] == "ok"
    assert target.read_text() == "total = 2\nprint(total)\n"


def test_multi_edit_atomic_on_failure(tmp_path):
    target = tmp_path / "config.ini"
    initial = "host=localhost\nport=8000\n"
    target.write_text(initial)
    ReadTool().execute(file_path=str(target.resolve()))

    result = MultiEditTool().execute(
        file_path=str(target.resolve()),
        edits=[
            {"old_string": "host=localhost", "new_string": "host=127.0.0.1"},
            {"old_string": "missing", "new_string": "found"},
        ],
    )

    assert result["error"].startswith("old_string was not found")
    assert target.read_text() == initial


def test_multi_edit_requires_read(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("alpha\n")

    result = MultiEditTool().execute(
        file_path=str(target.resolve()),
        edits=[{"old_string": "alpha", "new_string": "beta"}],
    )

    assert result["error"].startswith("File must be read")


def test_multi_edit_new_file(tmp_path):
    target = tmp_path / "created.txt"

    result = MultiEditTool().execute(
        file_path=str(target.resolve()),
        edits=[
            {"old_string": "", "new_string": "line1\n"},
            {"old_string": "line1", "new_string": "line2"},
        ],
    )

    assert result["result"] == "ok"
    assert target.read_text() == "line2\n"


def test_multi_edit_clears_read_state(tmp_path):
    target = tmp_path / "script.py"
    target.write_text("print('hi')\n")
    ReadTool().execute(file_path=str(target.resolve()))

    tool = MultiEditTool()
    tool.execute(
        file_path=str(target.resolve()),
        edits=[{"old_string": "print('hi')\n", "new_string": "print('hello')\n"}],
    )

    second = tool.execute(
        file_path=str(target.resolve()),
        edits=[{"old_string": "print('hello')\n", "new_string": "print('bye')\n"}],
    )

    assert second["error"].startswith("File must be read")


def test_multi_edit_replace_all(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("foo foo foo\n")
    ReadTool().execute(file_path=str(target.resolve()))

    result = MultiEditTool().execute(
        file_path=str(target.resolve()),
        edits=[{"old_string": "foo", "new_string": "bar", "replace_all": True}],
    )

    assert result["replacements"] == [3]
    assert target.read_text() == "bar bar bar\n"
