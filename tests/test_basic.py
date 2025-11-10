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
    from configs import config as config_manager
    from integrations import repository

    assert codebase_retrieval is not None
    assert config_manager is not None
    assert hasattr(codebase_retrieval, "index_project")
    assert hasattr(codebase_retrieval, "search_project")
    assert callable(codebase_retrieval.index_project)
    assert repository is not None


def test_codebase_retrieval_import():
    """Test helper functions exposed via codebase_retrieval."""
    from codebase_retrieval import index_project, search_project

    assert callable(index_project)
    assert callable(search_project)
