import os
from typing import Optional, Dict, Any
from functools import lru_cache

try:
    from pydantic.v1 import BaseModel, BaseSettings, Field, ValidationError
    from pydantic.v1.env_settings import SettingsSourceCallable
    from pydantic.v1 import validator
    from pydantic.v1.error_wrappers import ErrorWrapper
    from pydantic.v1.main import ModelMetaclass
    from pydantic.v1.fields import FieldInfo
    from pydantic.v1.utils import update_not_none
    from pydantic.v1.types import SecretStr
    from pydantic.v1 import parse_obj_as
except ImportError:
    from pydantic import BaseModel, BaseSettings, Field, ValidationError
    from pydantic.env_settings import SettingsSourceCallable
    from pydantic import validator
    from pydantic.error_wrappers import ErrorWrapper
    from pydantic.main import ModelMetaclass
    from pydantic.fields import FieldInfo
    from pydantic.utils import update_not_none
    from pydantic.types import SecretStr
    from pydantic import parse_obj_as

"""Centralized application configuration.

Defines all config sections inline to avoid scattering configuration across
multiple modules. RAG-related settings are kept under the unified manager.
"""



class LLMConfig(BaseSettings):
    """Configuration for LLM components."""
    
    model: str = Field(
        default="deepseek-chat",
        description="LLM model to use for generation"
    )
    
    api_base: Optional[str] = Field(
        default=None,
        description="API base URL for LLM service"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM service"
    )
    
    temperature: float = Field(
        default=0.1,
        description="Temperature for LLM generation",
        ge=0.0,
        le=2.0
    )
    
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens to generate",
        ge=1,
        le=8000
    )
    
    class Config:
        env_prefix = "LLM_"
        case_sensitive = False


class RAGConfig(BaseSettings):
    """Configuration for RAG components."""

    # Embedding model settings
    embedding_model: str = Field(
        default="openai-like",
        description="Embedding model to use (openai-like only for API mode)"
    )

    # Reranking settings
    rerank_model: str = Field(
        default="api",
        description="Reranking model to use (api only for API mode)"
    )

    # Chunking settings
    chunk_size: int = Field(
        default=200,
        description="Maximum number of lines per chunk",
        ge=1,
        le=2000,
    )

    class Config:
        env_prefix = "RAG_"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AppConfig(BaseModel):
    """Main application configuration."""
    
    rag: RAGConfig
    llm: LLMConfig
    
    class Config:
        arbitrary_types_allowed = True


class ConfigManager:
    """Singleton configuration manager."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from environment variables and defaults."""
        try:
            # Load individual configs (single source of truth)
            rag_config = RAGConfig()
            llm_config = LLMConfig()
            
            # Create main config
            self._config = AppConfig(
                rag=rag_config,
                llm=llm_config,
            )
        except Exception as e:
            # If validation fails, use safe defaults
            print(f"Configuration validation failed, using safe defaults: {e}")
            
            # Create config with safe default values
            self._config = AppConfig(
                rag=RAGConfig(embedding_model="openai-like", rerank_model="api"),
                llm=LLMConfig(
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=2000
                ),
            )
    
    def get_config(self) -> AppConfig:
        """Get the application configuration."""
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from environment variables."""
        # Store the current config object reference
        current_config = self._config
        
        # Load new configuration
        self._load_config()
        
        # Update the existing config object instead of replacing it
        # This maintains the singleton behavior
        if hasattr(current_config, '__dict__'):
            current_config.__dict__.update(self._config.__dict__)
            self._config = current_config
        
        return self._config
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            # This will raise ValidationError if config is invalid
            AppConfig.parse_obj(self._config.dict())
            return True
        except ValidationError:
            return False


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get the application configuration singleton."""
    manager = ConfigManager()
    return manager.get_config()


def reload_config() -> AppConfig:
    """Reload configuration from sources."""
    manager = ConfigManager()
    return manager.reload_config()


def validate_config() -> bool:
    """Validate current configuration."""
    manager = ConfigManager()
    return manager.validate_config()


# For backward compatibility
def get_rag_config() -> RAGConfig:
    """Get RAG configuration (backward compatibility)."""
    return get_config().rag
