"""
Basic tests for package structure and imports.
"""


def test_package_import():
    """Test that the package can be imported successfully."""
    # Package import test removed - flat structure doesn't have code_agent package
    pass


def test_main_import():
    """Test that main modules can be imported."""
    import codebase_retrieval
    from config import config as config_manager
    from retrieval import index as retrieval_index

    assert codebase_retrieval is not None
    assert config_manager is not None
    assert retrieval_index is not None


def test_codebase_retrieval_import():
    """Test helper functions exposed via codebase_retrieval."""
    import codebase_retrieval as cr

    assert hasattr(cr, "main")
