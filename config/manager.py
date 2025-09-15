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

from .rag_config import RAGConfig


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


class VectorDBConfig(BaseSettings):
    """Configuration for Vector Database."""
    
    host: str = Field(
        default="localhost",
        description="Vector database host"
    )
    
    port: int = Field(
        default=6333,
        description="Vector database port",
        ge=1,
        le=65535
    )
    
    collection_name: str = Field(
        default="code_embeddings",
        description="Default collection name"
    )
    
    class Config:
        env_prefix = "VECTORDB_"
        case_sensitive = False


class AppSettings(BaseSettings):
    """Application settings."""
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    
    max_workers: int = Field(
        default=4,
        description="Maximum worker threads",
        ge=1,
        le=32
    )
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


class AppConfig(BaseModel):
    """Main application configuration."""
    
    rag: RAGConfig
    llm: LLMConfig
    vectordb: VectorDBConfig
    app: AppSettings
    
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
            # Load individual configs
            rag_config = RAGConfig()
            llm_config = LLMConfig()
            vectordb_config = VectorDBConfig()
            app_settings = AppSettings()
            
            # Create main config
            self._config = AppConfig(
                rag=rag_config,
                llm=llm_config,
                vectordb=vectordb_config,
                app=app_settings
            )
        except Exception as e:
            # If validation fails, use safe defaults
            print(f"Configuration validation failed, using safe defaults: {e}")
            
            # Create config with safe default values
            self._config = AppConfig(
                rag=RAGConfig(
                    vector_store_path="./storage",
                    embedding_model="openai-like",
                    rerank_model="api",
                    llm_model="deepseek-chat",
                    default_search_limit=5
                ),
                llm=LLMConfig(
                    model="deepseek-chat",
                    temperature=0.1,
                    max_tokens=2000
                ),
                vectordb=VectorDBConfig(
                    host="localhost",
                    port=6333,
                    collection_name="code_embeddings"
                ),
                app=AppSettings(
                    debug=False,
                    log_level="INFO",
                    max_workers=4
                )
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