from tools.read import ReadTool
from tools.write import WriteTool


def test_write_creates_new_file(tmp_path):
    target = tmp_path / "new.txt"

    result = WriteTool().execute(file_path=str(target.resolve()), content="hello")

    assert result["content"].startswith("File created successfully at:")
    assert result["bytes_written"] == len("hello".encode("utf-8"))
    assert target.read_text() == "hello"


def test_write_requires_absolute_path(tmp_path):
    result = WriteTool().execute(file_path="relative/path.txt", content="data")

    assert result["error"] == "file_path must be an absolute path"


def test_write_requires_read_before_overwrite(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("first\n")

    result = WriteTool().execute(file_path=str(target.resolve()), content="second")

    assert result["error"].startswith("File must be read")


def test_write_succeeds_after_read(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("initial\n")
    reader = ReadTool()
    reader.execute(file_path=str(target.resolve()))

    result = WriteTool().execute(file_path=str(target.resolve()), content="updated")

    assert result["content"].startswith("File created successfully at:")
    assert target.read_text() == "updated"


def test_write_detects_changes_since_read(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("initial\n")
    reader = ReadTool()
    reader.execute(file_path=str(target.resolve()))

    # External modification after read
    target.write_text("changed\n")

    result = WriteTool().execute(file_path=str(target.resolve()), content="updated")

    assert result["error"].startswith("File must be read")


def test_write_requires_existing_parent(tmp_path):
    target = tmp_path / "missing" / "nested.txt"

    result = WriteTool().execute(file_path=str(target.resolve()), content="data")

    assert result["error"].startswith("Parent directory does not exist")


def test_write_rejects_directory(tmp_path):
    directory = tmp_path / "folder"
    directory.mkdir()

    result = WriteTool().execute(file_path=str(directory.resolve()), content="data")

    assert result["error"].startswith("Cannot write to a directory")


def test_write_requires_reread_after_write(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("initial\n")
    reader = ReadTool()
    reader.execute(file_path=str(target.resolve()))

    writer = WriteTool()
    writer.execute(file_path=str(target.resolve()), content="updated")

    result = writer.execute(file_path=str(target.resolve()), content="again")

    assert result["error"].startswith("File must be read")
