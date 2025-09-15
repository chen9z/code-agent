"""
Basic tests for package structure and imports.
"""

def test_package_import():
    """Test that the package can be imported successfully."""
    # Package import test removed - flat structure doesn't have code_agent package
    pass

def test_main_import():
    """Test that main modules can be imported."""
    import rag_flow
    from config import rag_config
    from tools import rag_tool
    from integration import repository
    
    assert rag_flow is not None
    assert rag_config is not None
    assert rag_tool is not None
    assert repository is not None

def test_rag_flow_import():
    """Test RAG flow specific imports."""
    from rag_flow import RAGFlow, RAGIndexNode, RAGSearchNode, RAGQueryNode
    
    assert RAGFlow is not None
    assert RAGIndexNode is not None
    assert RAGSearchNode is not None
    assert RAGQueryNode is not None