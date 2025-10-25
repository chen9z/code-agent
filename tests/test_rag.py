"""
Test module for RAG integration functionality.
"""

import os
from pathlib import Path
import pytest
import tempfile


def test_nodes_creation():
    """Test that RAG nodes can be instantiated."""
    from code_rag import RAGIndexNode, RAGSearchNode, RAGQueryNode

    assert RAGIndexNode() is not None
    assert RAGSearchNode() is not None
    assert RAGQueryNode() is not None


def test_repository_adapter():
    """Test repository adapter functionality."""
    from integrations.repository import RepositoryAdapter
    
    # Test fallback mode (since chat-codebase may not be available)
    adapter = RepositoryAdapter()
    
    assert adapter is not None
    assert hasattr(adapter, 'index_project')
    assert hasattr(adapter, 'search')
    assert hasattr(adapter, 'format_search_results')


def test_config():
    """Test configuration loading (minimal)."""
    from configs.manager import get_config

    cfg = get_config()

    assert cfg is not None
    assert hasattr(cfg, 'rag')
    assert hasattr(cfg.rag, 'embedding_model')
    assert hasattr(cfg.rag, 'rerank_model')


def test_rag_operations():
    """Test basic RAG operations using the flow API."""
    from code_rag import run_rag_workflow

    # Create a temporary test project
    with tempfile.TemporaryDirectory() as test_project, tempfile.TemporaryDirectory() as store_dir:
        original_store = os.environ.get("CODEBASE_QDRANT_PATH")

        def set_store(suffix: str) -> None:
            os.environ["CODEBASE_QDRANT_PATH"] = str(Path(store_dir) / suffix)

        # Create a test file
        test_file = os.path.join(test_project, "test.py")
        with open(test_file, "w") as f:
            f.write("def hello_world():\n    print('Hello, World!')\n")

        # Test indexing
        set_store("index")
        result = run_rag_workflow(action="index", project_path=test_project)
        assert result["status"] == "success"

        # Test search
        set_store("search")
        result = run_rag_workflow(
            action="search",
            project_name=Path(test_project).name,
            query="hello world",
        )
        assert result["status"] == "success"

        # Test query
        set_store("query")
        result = run_rag_workflow(
            action="query",
            project_name=Path(test_project).name,
            question="What does this code do?",
        )
        assert result["status"] == "success"

        if original_store is not None:
            os.environ["CODEBASE_QDRANT_PATH"] = original_store
        else:
            os.environ.pop("CODEBASE_QDRANT_PATH", None)


def test_rag_flow():
    """Test RAG flow functionality."""
    from code_rag import run_rag_workflow

    with tempfile.TemporaryDirectory() as test_project, tempfile.TemporaryDirectory() as store_dir:
        original_store = os.environ.get("CODEBASE_QDRANT_PATH")

        def set_store(suffix: str) -> None:
            os.environ["CODEBASE_QDRANT_PATH"] = str(Path(store_dir) / suffix)

        # Create a test file
        test_file = os.path.join(test_project, "test.py")
        with open(test_file, "w") as f:
            f.write("def hello_world():\n    return 'Hello, World!'\n")

        # Test indexing flow
        try:
            set_store("index")
            result = run_rag_workflow(action="index", project_path=test_project)
            assert isinstance(result, dict) and result.get("status") == "success"
        except Exception as e:
            pytest.fail(f"Index flow failed: {e}")

        # Test search flow
        try:
            set_store("search")
            result = run_rag_workflow(
                action="search",
                project_name=Path(test_project).name,
                query="function definition",
            )
            assert isinstance(result, dict) and result.get("status") == "success"
        except Exception as e:
            pytest.fail(f"Search flow failed: {e}")

        # Test query flow
        try:
            set_store("query")
            result = run_rag_workflow(
                action="query",
                project_name=Path(test_project).name,
                question="What does this code do?",
            )
            assert isinstance(result, dict) and result.get("status") == "success"
        except Exception as e:
            pytest.fail(f"Query flow failed: {e}")

        if original_store is not None:
            os.environ["CODEBASE_QDRANT_PATH"] = original_store
        else:
            os.environ.pop("CODEBASE_QDRANT_PATH", None)
