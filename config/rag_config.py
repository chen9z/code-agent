import os
from typing import Optional
try:
    from pydantic.v1 import BaseSettings, Field
except ImportError:
    from pydantic import BaseSettings, Field


class RAGConfig(BaseSettings):
    """Configuration for RAG components."""
    
    # Vector database settings
    vector_store_path: str = Field(
        default="./storage",
        description="Path to the Qdrant vector database storage"
    )
    
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
    
    # LLM settings
    llm_model: str = Field(
        default="deepseek-chat",
        description="LLM model to use for generation"
    )
    
    # API settings
    openai_api_base: Optional[str] = Field(
        default=None,
        description="OpenAI API base URL"
    )
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    # Search settings
    default_search_limit: int = Field(
        default=5,
        description="Default number of search results to return"
    )
    
    class Config:
        env_prefix = "RAG_"
        case_sensitive = False

    def __init__(self, **kwargs):
        # Set default values from environment variables
        kwargs.setdefault("openai_api_base", os.getenv("OPENAI_API_BASE"))
        kwargs.setdefault("openai_api_key", os.getenv("OPENAI_API_KEY"))
        super().__init__(**kwargs)


def get_rag_config() -> RAGConfig:
    """Get the RAG configuration."""
    return RAGConfig()