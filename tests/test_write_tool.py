from tools.write import WriteTool


def test_write_creates_new_file(tmp_path):
    target = tmp_path / "new.txt"

    result = WriteTool().execute(file_path=str(target.resolve()), content="hello")

    assert result["status"] == "success"
    assert result["content"].startswith("File created successfully at:")
    assert result["data"]["bytes_written"] == len("hello".encode("utf-8"))
    assert target.read_text() == "hello"


def test_write_requires_absolute_path(tmp_path):
    result = WriteTool().execute(file_path="relative/path.txt", content="data")

    assert result["status"] == "error"
    assert result["content"] == "file_path must be an absolute path"


def test_write_overwrites_existing_file_without_prior_read(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("first\n")

    result = WriteTool().execute(file_path=str(target.resolve()), content="second")

    assert result["status"] == "success"
    assert target.read_text() == "second"


def test_write_requires_existing_parent(tmp_path):
    target = tmp_path / "missing" / "nested.txt"

    result = WriteTool().execute(file_path=str(target.resolve()), content="data")

    assert result["status"] == "error"
    assert result["content"].startswith("Parent directory does not exist")


def test_write_rejects_directory(tmp_path):
    directory = tmp_path / "folder"
    directory.mkdir()

    result = WriteTool().execute(file_path=str(directory.resolve()), content="data")

    assert result["status"] == "error"
    assert result["content"].startswith("Cannot write to a directory")


def test_write_allows_sequential_overwrites(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("initial\n")

    writer = WriteTool()
    first = writer.execute(file_path=str(target.resolve()), content="updated")
    second = writer.execute(file_path=str(target.resolve()), content="again")

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert target.read_text() == "again"
