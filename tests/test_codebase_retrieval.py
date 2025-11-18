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

    with tempfile.TemporaryDirectory() as test_project:
        original_store = os.environ.get("CODEBASE_QDRANT_PATH")
        os.environ["CODEBASE_QDRANT_PATH"] = test_project

        test_file = Path(test_project) / "test.py"
        test_file.write_text("def hello_world():\n    print('Hello, World!')\n", encoding="utf-8")

        adapter = create_index()
        index_info = adapter.index_project(str(Path(test_project)), show_progress=False)
        assert "collection_name" in index_info
        assert "chunk_size" in index_info and index_info["chunk_size"] > 0
        project_key = index_info["project_key"]

        results = adapter.search(project_key, "hello world")
        assert isinstance(results, list)

        adapter._indexer.close()
        fresh = create_index()
        hits = fresh.search(project_key, "hello world", project_path=test_project)
        assert hits and any("hello_world" in doc.content for doc in hits)

        if original_store is not None:
            os.environ["CODEBASE_QDRANT_PATH"] = original_store
        else:
            os.environ.pop("CODEBASE_QDRANT_PATH", None)


def test_missing_parameters():
    """Ensure helper functions validate required parameters."""
    from retrieval.index import create_index

    index = create_index()
    with pytest.raises(KeyError):
        index.search("non-existent-key", "hello")


def test_search_with_project_key_disambiguates_duplicate_names(tmp_path, monkeypatch):
    """Ensure project_key can disambiguate same-named directories."""
    from retrieval.index import create_index

    repo_a = tmp_path / "app" / "repo"
    repo_b = tmp_path / "other" / "repo"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    (repo_a / "alpha.py").write_text("def alpha_only():\n    return 1\n", encoding="utf-8")
    (repo_b / "alpha.py").write_text("def beta_only():\n    return 2\n", encoding="utf-8")

    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))
    adapter = create_index()

    info_a = adapter.index_project(str(repo_a), show_progress=False, refresh=True)
    info_b = adapter.index_project(str(repo_b), show_progress=False, refresh=True)

    assert info_a["project_name"] == info_b["project_name"]
    assert info_a["project_key"] != info_b["project_key"]

    hits_a = adapter.search(
        info_a["project_key"],
        "alpha_only",
    )
    assert hits_a and any("alpha_only" in doc.content for doc in hits_a)

    hits_b = adapter.search(
        info_b["project_key"],
        "beta_only",
    )
    assert hits_b and any("beta_only" in doc.content for doc in hits_b)


def test_search_allows_project_path_without_index(monkeypatch, tmp_path):
    """Search should accept a project_path and build index on the fly."""
    from retrieval.index import create_index
    from retrieval.codebase_indexer import compute_project_key

    repo = tmp_path / "solo" / "repo"
    repo.mkdir(parents=True)
    (repo / "alpha.py").write_text("def lone_func():\n    return 42\n", encoding="utf-8")

    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))

    adapter = create_index()

    key = compute_project_key(repo)
    hits = adapter.search(
        project_key=key,
        query="lone_func",
        project_path=str(repo),
    )
    assert hits and any("lone_func" in doc.content for doc in hits)

    hits_again = adapter.search(
        project_key=key,
        query="lone_func",
    )
    assert hits_again and any("lone_func" in doc.content for doc in hits_again)


def test_compute_project_key_helper(tmp_path):
    from retrieval.codebase_indexer import compute_project_key

    repo = tmp_path / "pkg"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    key1 = compute_project_key(repo)
    key2 = compute_project_key(str(repo))
    assert key1 == key2
