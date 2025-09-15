#!/usr/bin/env python3
"""
Tests for the centralized configuration management system.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from config.manager import get_config, reload_config, validate_config, ConfigManager


def test_config_manager_singleton():
    """Test that ConfigManager is a proper singleton."""
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    
    assert manager1 is manager2
    assert id(manager1) == id(manager2)


def test_get_config_singleton():
    """Test that get_config returns the same instance."""
    config1 = get_config()
    config2 = get_config()
    
    assert config1 is config2
    assert id(config1) == id(config2)


def test_config_structure():
    """Test that config has the expected structure."""
    config = get_config()
    
    # Check that all expected sections exist
    assert hasattr(config, 'rag')
    assert hasattr(config, 'llm')
    assert hasattr(config, 'vectordb')
    assert hasattr(config, 'app')
    
    # Check RAG config
    assert hasattr(config.rag, 'vector_store_path')
    assert hasattr(config.rag, 'embedding_model')
    assert hasattr(config.rag, 'rerank_model')
    assert hasattr(config.rag, 'llm_model')
    assert hasattr(config.rag, 'openai_api_base')
    assert hasattr(config.rag, 'openai_api_key')
    assert hasattr(config.rag, 'default_search_limit')
    
    # Check LLM config
    assert hasattr(config.llm, 'model')
    assert hasattr(config.llm, 'api_base')
    assert hasattr(config.llm, 'api_key')
    assert hasattr(config.llm, 'temperature')
    assert hasattr(config.llm, 'max_tokens')
    
    # Check VectorDB config
    assert hasattr(config.vectordb, 'host')
    assert hasattr(config.vectordb, 'port')
    assert hasattr(config.vectordb, 'collection_name')
    
    # Check App settings
    assert hasattr(config.app, 'debug')
    assert hasattr(config.app, 'log_level')
    assert hasattr(config.app, 'max_workers')


def test_config_default_values():
    """Test that config has sensible default values."""
    config = get_config()
    
    # RAG defaults
    assert config.rag.vector_store_path == "./storage"
    assert config.rag.embedding_model == "openai-like"
    assert config.rag.rerank_model == "api"
    assert config.rag.llm_model == "deepseek-chat"
    assert config.rag.default_search_limit == 5
    
    # LLM defaults
    assert config.llm.model == "deepseek-chat"
    assert config.llm.temperature == 0.1
    assert config.llm.max_tokens == 2000
    
    # VectorDB defaults
    assert config.vectordb.host == "localhost"
    assert config.vectordb.port == 6333
    assert config.vectordb.collection_name == "code_embeddings"
    
    # App defaults
    assert config.app.debug == False
    assert config.app.log_level == "INFO"
    assert config.app.max_workers == 4


def test_environment_variable_override():
    """Test that environment variables override defaults."""
    
    with patch.dict(os.environ, {
        'RAG_VECTOR_STORE_PATH': '/custom/storage',
        'RAG_EMBEDDING_MODEL': 'custom-embedding',
        'LLM_MODEL': 'custom-model',
        'LLM_TEMPERATURE': '0.5',
        'VECTORDB_HOST': 'custom-host',
        'VECTORDB_PORT': '9999',
        'APP_DEBUG': 'true',
        'APP_LOG_LEVEL': 'DEBUG'
    }):
        # Reload config to pick up environment variables
        config = reload_config()
        
        # Check that environment variables override defaults
        assert config.rag.vector_store_path == "/custom/storage"
        assert config.rag.embedding_model == "custom-embedding"
        assert config.llm.model == "custom-model"
        assert config.llm.temperature == 0.5
        assert config.vectordb.host == "custom-host"
        assert config.vectordb.port == 9999
        assert config.app.debug == True
        assert config.app.log_level == "DEBUG"


def test_config_validation():
    """Test that configuration validation works."""
    # Default config should be valid
    assert validate_config() == True
    
    # Test with invalid values - Pydantic will raise ValidationError
    # and our fallback will use safe defaults
    with patch.dict(os.environ, {
        'LLM_TEMPERATURE': '2.5',  # Out of range (0.0-2.0)
        'APP_MAX_WORKERS': '50'    # Out of range (1-32)
    }):
        # Our fallback should use safe defaults when validation fails
        config = reload_config()
        
        # Should use safe defaults (not the invalid values)
        assert config.llm.temperature == 0.1  # Safe default
        assert config.app.max_workers == 4    # Safe default
        
        # Config should still be considered valid (fallback worked)
        assert validate_config() == True


def test_config_reload():
    """Test that config reloading works correctly."""
    config1 = get_config()
    original_path = config1.rag.vector_store_path
    
    # Change environment and reload
    with patch.dict(os.environ, {
        'RAG_VECTOR_STORE_PATH': '/reloaded/path'
    }):
        config2 = reload_config()
        
        # Should be the same object (singleton)
        assert config1 is config2
        
        # But values should be updated
        assert config2.rag.vector_store_path == "/reloaded/path"
    
    # Clean up environment and restore original config
    with patch.dict(os.environ, {
        'RAG_VECTOR_STORE_PATH': original_path
    }):
        reload_config()
        
        # Should be back to original
        assert get_config().rag.vector_store_path == original_path


def test_config_backward_compatibility():
    """Test backward compatibility with old get_rag_config function."""
    from config.manager import get_rag_config
    
    rag_config = get_rag_config()
    main_config = get_config()
    
    # Should return the same RAG config instance
    assert rag_config is main_config.rag
    
    # Should have the same values
    assert rag_config.vector_store_path == main_config.rag.vector_store_path
    assert rag_config.embedding_model == main_config.rag.embedding_model


def test_config_serialization():
    """Test that config can be serialized to dict."""
    config = get_config()
    config_dict = config.dict()
    
    # Should contain all sections
    assert 'rag' in config_dict
    assert 'llm' in config_dict
    assert 'vectordb' in config_dict
    assert 'app' in config_dict
    
    # Should be able to recreate from dict
    from config.manager import AppConfig
    recreated_config = AppConfig.parse_obj(config_dict)
    
    assert recreated_config.rag.vector_store_path == config.rag.vector_store_path
    assert recreated_config.llm.model == config.llm.model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])