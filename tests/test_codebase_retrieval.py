"""
Test module for codebase retrieval functionality.
"""

import os
from pathlib import Path
import tempfile

import pytest
def test_repository_adapter():
    """Test repository adapter functionality."""
    from integrations.repository import RepositoryAdapter

    original_store = os.environ.get("CODEBASE_QDRANT_PATH")
    with tempfile.TemporaryDirectory() as store_dir:
        os.environ["CODEBASE_QDRANT_PATH"] = store_dir
        adapter = RepositoryAdapter()

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
    from codebase_retrieval import index_project, search_project

    with tempfile.TemporaryDirectory() as test_project, tempfile.TemporaryDirectory() as store_dir:
        original_store = os.environ.get("CODEBASE_QDRANT_PATH")

        def set_store(suffix: str) -> None:
            os.environ["CODEBASE_QDRANT_PATH"] = str(Path(store_dir) / suffix)

        test_file = Path(test_project) / "test.py"
        test_file.write_text("def hello_world():\n    print('Hello, World!')\n", encoding="utf-8")

        set_store("index")
        result = index_project(test_project)
        assert result["status"] == "success"
        assert result.get("chunk_size")

        set_store("search")
        result = search_project(Path(test_project).name, "hello world")
        assert result["status"] == "success"

        if original_store is not None:
            os.environ["CODEBASE_QDRANT_PATH"] = original_store
        else:
            os.environ.pop("CODEBASE_QDRANT_PATH", None)


def test_missing_parameters():
    """Ensure helper functions validate required parameters."""
    from codebase_retrieval import index_project, search_project

    with pytest.raises(ValueError):
        index_project("")
    with pytest.raises(ValueError):
        search_project("", "test")
    with pytest.raises(ValueError):
        search_project("project", "")
