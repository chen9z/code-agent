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
    """Test that config has the expected minimal structure."""
    config = get_config()

    # Minimal sections
    assert hasattr(config, 'rag')
    assert hasattr(config, 'llm')

    # RAG minimal settings
    assert hasattr(config.rag, 'embedding_model')
    assert hasattr(config.rag, 'rerank_model')

    # LLM settings
    assert hasattr(config.llm, 'model')
    assert hasattr(config.llm, 'api_base')
    assert hasattr(config.llm, 'api_key')
    assert hasattr(config.llm, 'temperature')
    assert hasattr(config.llm, 'max_tokens')


def test_config_default_values():
    """Test that config has sensible default values."""
    config = get_config()

    # RAG defaults
    assert config.rag.embedding_model == "openai-like"
    assert config.rag.rerank_model == "api"

    # LLM defaults
    assert config.llm.model == "deepseek-chat"
    assert config.llm.temperature == 0.1
    assert config.llm.max_tokens == 2000


def test_environment_variable_override():
    """Test that environment variables override defaults."""

    with patch.dict(os.environ, {
        'RAG_EMBEDDING_MODEL': 'custom-embedding',
        'RAG_RERANK_MODEL': 'custom-reranker',
        'LLM_MODEL': 'custom-model',
        'LLM_TEMPERATURE': '0.5',
    }):
        # Reload config to pick up environment variables
        config = reload_config()

        # Check that environment variables override defaults
        assert config.rag.embedding_model == "custom-embedding"
        assert config.rag.rerank_model == "custom-reranker"
        assert config.llm.model == "custom-model"
        assert config.llm.temperature == 0.5


def test_config_validation():
    """Test that configuration validation works."""
    # Default config should be valid
    assert validate_config() == True
    
    # Test with invalid values - Pydantic will raise ValidationError
    # and our fallback will use safe defaults
    with patch.dict(os.environ, {
        'LLM_TEMPERATURE': '2.5',  # Out of range (0.0-2.0)
    }):
        # Our fallback should use safe defaults when validation fails
        config = reload_config()
        
        # Should use safe defaults (not the invalid values)
        assert config.llm.temperature == 0.1  # Safe default
        
        # Config should still be considered valid (fallback worked)
        assert validate_config() == True


def test_config_reload():
    """Test that config reloading works correctly."""
    config1 = get_config()
    original_embedding = config1.rag.embedding_model

    # Change environment and reload
    with patch.dict(os.environ, {
        'RAG_EMBEDDING_MODEL': 'reloaded-embedding'
    }):
        config2 = reload_config()

        # Should be the same object (singleton)
        assert config1 is config2

        # But values should be updated
        assert config2.rag.embedding_model == "reloaded-embedding"

    # Clean up environment and restore original config
    with patch.dict(os.environ, {
        'RAG_EMBEDDING_MODEL': original_embedding
    }):
        reload_config()

        # Should be back to original
        assert get_config().rag.embedding_model == original_embedding


def test_config_backward_compatibility():
    """Test backward compatibility with old get_rag_config function."""
    from config.manager import get_rag_config

    rag_config = get_rag_config()
    main_config = get_config()

    # Should return the same RAG config instance
    assert rag_config is main_config.rag

    # Should have the same values
    assert rag_config.embedding_model == main_config.rag.embedding_model
    assert rag_config.rerank_model == main_config.rag.rerank_model


def test_config_serialization():
    """Test that config can be serialized to dict."""
    config = get_config()
    config_dict = config.dict()

    # Should contain minimal sections
    assert 'rag' in config_dict
    assert 'llm' in config_dict

    # Should be able to recreate from dict
    from config.manager import AppConfig
    recreated_config = AppConfig.parse_obj(config_dict)

    assert recreated_config.rag.embedding_model == config.rag.embedding_model
    assert recreated_config.llm.model == config.llm.model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
