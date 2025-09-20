"""
Basic tests for package structure and imports.
"""

def test_package_import():
    """Test that the package can be imported successfully."""
    # Package import test removed - flat structure doesn't have code_agent package
    pass

def test_main_import():
    """Test that main modules can be imported."""
    import code_rag
    from configs import manager as config_manager
    import code_rag as rf
    from integrations import repository
    
    assert code_rag is not None
    assert config_manager is not None
    assert hasattr(rf, 'RAGIndexNode')
    assert hasattr(rf, 'RAGSearchNode')
    assert hasattr(rf, 'RAGQueryNode')
    assert repository is not None

def test_code_rag_import():
    """Test RAG flow specific imports via code_rag."""
    from code_rag import RAGFlow, RAGIndexNode, RAGSearchNode, RAGQueryNode
    
    assert RAGFlow is not None
    assert RAGIndexNode is not None
    assert RAGSearchNode is not None
    assert RAGQueryNode is not None
