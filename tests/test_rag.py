"""
Test module for RAG integration functionality.
"""

import os
import pytest
import tempfile


def test_rag_tool_creation():
    """Test that RAG tool can be created successfully."""
    from tools.rag_tool import create_rag_tool
    
    rag_tool = create_rag_tool()
    
    assert rag_tool is not None
    assert hasattr(rag_tool, 'name')
    assert hasattr(rag_tool, 'description')
    assert hasattr(rag_tool, 'parameters')
    
    # Test parameters
    params = rag_tool.parameters
    assert "action" in params["properties"]


def test_repository_adapter():
    """Test repository adapter functionality."""
    from integration.repository import RepositoryAdapter
    
    # Test fallback mode (since chat-codebase may not be available)
    adapter = RepositoryAdapter()
    
    assert adapter is not None
    assert hasattr(adapter, 'index_project')
    assert hasattr(adapter, 'search')
    assert hasattr(adapter, 'format_search_results')


def test_config():
    """Test configuration loading (minimal)."""
    from config.manager import get_config

    cfg = get_config()

    assert cfg is not None
    assert hasattr(cfg, 'rag')
    assert hasattr(cfg.rag, 'embedding_model')
    assert hasattr(cfg.rag, 'rerank_model')


def test_rag_operations():
    """Test basic RAG operations."""
    from tools.rag_tool import create_rag_tool
    
    rag_tool = create_rag_tool()
    
    # Create a temporary test project
    with tempfile.TemporaryDirectory() as test_project:
        # Create a test file
        test_file = os.path.join(test_project, "test.py")
        with open(test_file, "w") as f:
            f.write("def hello_world():\n    print('Hello, World!')\n")
        
        # Test indexing (will use fallback since chat-codebase not available)
        result = rag_tool.execute(action="index", project_path=test_project)
        assert result["status"] == "success"
        
        # Test search (fallback mode)
        result = rag_tool.execute(
            action="search", 
            project_name="test_project", 
            query="hello world"
        )
        assert result["status"] == "success"
        
        # Test query (fallback mode) 
        result = rag_tool.execute(
            action="query",
            project_name="test_project",
            query="What does this code do?"
        )
        assert result["status"] == "success"


def test_rag_flow():
    """Test RAG flow functionality."""
    from rag_flow import run_rag_workflow
    
    with tempfile.TemporaryDirectory() as test_project:
        # Create a test file
        test_file = os.path.join(test_project, "test.py")
        with open(test_file, "w") as f:
            f.write("def hello_world():\n    return 'Hello, World!'\n")
        
        # Test indexing flow - should complete without error
        try:
            result = run_rag_workflow(action="index", project_path=test_project)
            # Flow returns None when completed successfully
            assert result is None
        except Exception as e:
            pytest.fail(f"Index flow failed: {e}")
        
        # Test search flow - should complete without error
        try:
            result = run_rag_workflow(
                action="search", 
                project_name="test_project", 
                query="function definition"
            )
            # Flow returns None when completed successfully
            assert result is None
        except Exception as e:
            pytest.fail(f"Search flow failed: {e}")
        
        # Test query flow - should complete without error
        try:
            result = run_rag_workflow(
                action="query", 
                project_name="test_project", 
                question="What does this code do?"
            )
            # Flow returns None when completed successfully
            assert result is None
        except Exception as e:
            pytest.fail(f"Query flow failed: {e}")
