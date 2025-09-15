from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, BaseSettings, Field, ValidationError

"""Minimal centralized configuration used by the project.

Only includes settings referenced in code: LLM options and RAG parameters
(embedding/rerank models and chunk_size).
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


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get cached application configuration."""
    try:
        return AppConfig(rag=RAGConfig(), llm=LLMConfig())
    except Exception:
        # Fallback to safe defaults
        return AppConfig(
            rag=RAGConfig(embedding_model="openai-like", rerank_model="api"),
            llm=LLMConfig(model="deepseek-chat", temperature=0.1, max_tokens=2000),
        )


def reload_config() -> AppConfig:
    """Reload configuration by clearing cache and re-instantiating."""
    get_config.cache_clear()
    return get_config()


def validate_config() -> bool:
    """Validate that current environment yields a valid configuration."""
    try:
        AppConfig(rag=RAGConfig(), llm=LLMConfig())
        return True
    except ValidationError:
        return False


def get_rag_config() -> RAGConfig:
    """Backward-compatible accessor for RAG config."""
    return get_config().rag
