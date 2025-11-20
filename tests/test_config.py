#!/usr/bin/env python3
"""
Tests for the centralized configuration management system.
"""

import os
import pytest
from unittest.mock import patch

from config.config import AppConfig, get_config, reload_config


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
    assert hasattr(config, "rag_embedding_model")
    assert hasattr(config, "rag_rerank_model")
    assert hasattr(config, "rag_chunk_size")
    assert hasattr(config, "llm_model")
    assert hasattr(config, "llm_temperature")
    assert hasattr(config, "llm_max_tokens")
    assert hasattr(config, "cli_tool_timeout_seconds")


def test_config_default_values():
    """Test that config has sensible default values."""
    with patch.dict(os.environ, {}, clear=True):
        config = reload_config()

        assert config.rag_embedding_model == "openai-like"
        assert config.rag_rerank_model == "api"
        assert config.rag_chunk_size == 2048
        assert config.llm_model == "deepseek-chat"
        assert config.llm_temperature == 0.1
        assert config.llm_max_tokens == 2000
        assert config.cli_tool_timeout_seconds == 60

    reload_config()


def test_environment_variable_override():
    """Test that environment variables override defaults."""

    with patch.dict(os.environ, {
        'EMBEDDING_MODEL': 'custom-embedding',
        'RERANK_MODEL': 'custom-reranker',
        'MODEL': 'custom-model',
        'TEMPERATURE': '0.5',
        'CHUNK_SIZE': '123',
        'TOOL_TIMEOUT_SECONDS': '75',
    }):
        # Reload config to pick up environment variables
        config = reload_config()

        # Check that environment variables override defaults
        assert config.rag_embedding_model == "custom-embedding"
        assert config.rag_rerank_model == "custom-reranker"
        assert config.llm_model == "custom-model"
        assert config.llm_temperature == 0.5
        assert config.rag_chunk_size == 123
        assert config.cli_tool_timeout_seconds == 75

    # 恢复默认配置，避免影响其他测试
    reload_config()


def test_invalid_environment_values_reset_to_default():
    """Invalid env inputs should fall back to safe defaults."""
    with patch.dict(os.environ, {
        'TEMPERATURE': 'not-a-number',
        'MAX_TOKENS': 'oops',
        'CHUNK_SIZE': '-10',
        'TOOL_TIMEOUT_SECONDS': '0',
        'OPIK_ENABLED': 'maybe',
    }):
        config = reload_config()

        assert config.llm_temperature == 0.1
        assert config.llm_max_tokens == 2000
        assert config.rag_chunk_size == 2048
        assert config.cli_tool_timeout_seconds == 60
        assert config.llm_opik_enabled is True

    reload_config()


def test_config_reload():
    """Test that config reloading works correctly."""
    config1 = get_config()
    original_embedding = config1.rag_embedding_model
    original_chunk_size = config1.rag_chunk_size

    # Change environment and reload
    with patch.dict(os.environ, {
        'EMBEDDING_MODEL': 'reloaded-embedding',
        'CHUNK_SIZE': '321',
        'TOOL_TIMEOUT_SECONDS': '120',
    }):
        config2 = reload_config()

        # Should be a fresh instance after reload
        assert config1 is not config2
        assert config2 is get_config()

        # But values should be updated
        assert config2.rag_embedding_model == "reloaded-embedding"
        assert config2.rag_chunk_size == 321
        assert config2.cli_tool_timeout_seconds == 120

    # Clean up environment and restore original config
    reload_config()
    cfg = get_config()
    assert cfg.rag_embedding_model == original_embedding
    assert cfg.rag_chunk_size == original_chunk_size


def test_config_serialization():
    """Test that config can be serialized to dict."""
    config = get_config()
    config_dict = config.to_dict()
    assert config_dict['rag_embedding_model'] == config.rag_embedding_model
    assert config_dict['llm_model'] == config.llm_model
    assert config_dict['cli_tool_timeout_seconds'] == config.cli_tool_timeout_seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
