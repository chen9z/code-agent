#!/usr/bin/env python3
"""
Tests for the centralized configuration management system.
"""

import os
import pytest
from unittest.mock import patch

from configs.manager import AppConfig, get_config, reload_config, validate_config


def test_get_config_singleton():
    """Test that get_config returns the same cached instance."""
    config1 = get_config()
    config2 = get_config()
    
    assert config1 is config2
    assert id(config1) == id(config2)


def test_config_structure():
    """Test that config has the expected minimal structure."""
    config = get_config()

    assert isinstance(config, AppConfig)
    assert hasattr(config, 'rag')
    assert hasattr(config, 'llm')
    assert hasattr(config.rag, 'embedding_model')
    assert hasattr(config.rag, 'rerank_model')
    assert hasattr(config.rag, 'chunk_size')
    assert hasattr(config.llm, 'model')
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
    # RAG defaults
    assert config.rag.chunk_size == 200


def test_environment_variable_override():
    """Test that environment variables override defaults."""

    with patch.dict(os.environ, {
        'RAG_EMBEDDING_MODEL': 'custom-embedding',
        'RAG_RERANK_MODEL': 'custom-reranker',
        'LLM_MODEL': 'custom-model',
        'LLM_TEMPERATURE': '0.5',
        'RAG_CHUNK_SIZE': '123',
    }):
        # Reload config to pick up environment variables
        config = reload_config()
        
        # Check that environment variables override defaults
        assert config.rag.embedding_model == "custom-embedding"
        assert config.rag.rerank_model == "custom-reranker"
        assert config.llm.model == "custom-model"
        assert config.llm.temperature == 0.5
        assert config.rag.chunk_size == 123


def test_config_validation():
    """Test that configuration validation works."""
    # Default config should be valid
    assert validate_config() == True
    
    with patch.dict(os.environ, {
        'LLM_TEMPERATURE': 'not-a-number',
    }):
        config = reload_config()
        assert config.llm.temperature == 0.1
        assert validate_config() == True


def test_config_reload():
    """Test that config reloading works correctly."""
    config1 = get_config()
    original_embedding = config1.rag.embedding_model
    original_chunk_size = config1.rag.chunk_size

    # Change environment and reload
    with patch.dict(os.environ, {
        'RAG_EMBEDDING_MODEL': 'reloaded-embedding',
        'RAG_CHUNK_SIZE': '321'
    }):
        config2 = reload_config()

        # Should be a fresh instance after reload
        assert config1 is not config2
        assert config2 is get_config()

        # But values should be updated
        assert config2.rag.embedding_model == "reloaded-embedding"
        assert config2.rag.chunk_size == 321

    # Clean up environment and restore original config
    reload_config()
    cfg = get_config()
    assert cfg.rag.embedding_model == original_embedding
    assert cfg.rag.chunk_size == original_chunk_size


def test_config_backward_compatibility():
    """Test backward compatibility with old get_rag_config function."""
    from configs.manager import get_rag_config

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
    config_dict = config.to_dict()
    assert config_dict['rag']['embedding_model'] == config.rag.embedding_model
    assert config_dict['llm']['model'] == config.llm.model

    recreated = AppConfig.from_dict(config_dict)
    assert recreated.rag.embedding_model == config.rag.embedding_model
    assert recreated.llm.model == config.llm.model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
