"""
Test module for codebase retrieval functionality.
"""

import os
from pathlib import Path
import tempfile

import pytest


def test_project_index_adapter():
    """Test project index adapter functionality."""
    from retrieval.index import Index

    original_store = os.environ.get("CODEBASE_QDRANT_PATH")
    with tempfile.TemporaryDirectory() as store_dir:
        os.environ["CODEBASE_QDRANT_PATH"] = store_dir
        adapter = Index()

    if original_store is not None:
        os.environ["CODEBASE_QDRANT_PATH"] = original_store
    else:
        os.environ.pop("CODEBASE_QDRANT_PATH", None)

    assert adapter is not None
    assert hasattr(adapter, "index_project")
    assert hasattr(adapter, "search")
    assert hasattr(adapter, "format_search_results")


def test_index_and_search_flow():
    """Index a project and run semantic search using helper APIs."""
    from retrieval.index import create_index

    with tempfile.TemporaryDirectory() as test_project, tempfile.TemporaryDirectory() as store_dir:
        original_store = os.environ.get("CODEBASE_QDRANT_PATH")

        def set_store(suffix: str) -> None:
            os.environ["CODEBASE_QDRANT_PATH"] = str(Path(store_dir) / suffix)

        test_file = Path(test_project) / "test.py"
        test_file.write_text("def hello_world():\n    print('Hello, World!')\n", encoding="utf-8")

        set_store("index")
        adapter = create_index()
        index_info = adapter.index_project(str(Path(test_project)), show_progress=False)
        assert "collection_name" in index_info
        assert "chunk_size" in index_info and index_info["chunk_size"] > 0

        set_store("search")
        results = create_index().search(Path(test_project).name, "hello world")
        assert isinstance(results, list)

        if original_store is not None:
            os.environ["CODEBASE_QDRANT_PATH"] = original_store
        else:
            os.environ.pop("CODEBASE_QDRANT_PATH", None)


def test_missing_parameters():
    """Ensure helper functions validate required parameters."""
    from retrieval.index import create_index

    index = create_index()
    with pytest.raises(KeyError):
        index.search("project", "hello")
