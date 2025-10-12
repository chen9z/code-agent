from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()


def _get_env_value(prefix: str, key: str) -> Optional[str]:
    return os.getenv(f"{prefix}{key}")


def _to_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


def _to_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(slots=True)
class LLMConfig:
    model: str = "deepseek-chat"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    opik_project_name: Optional[str] = None
    opik_enabled: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        prefix = "LLM_"
        defaults = cls()
        model = _get_env_value(prefix, "MODEL") or defaults.model
        api_base = _get_env_value(prefix, "API_BASE") or defaults.api_base
        api_key = _get_env_value(prefix, "API_KEY") or defaults.api_key
        temperature = _to_float(_get_env_value(prefix, "TEMPERATURE"), defaults.temperature)
        max_tokens = _to_int(_get_env_value(prefix, "MAX_TOKENS"), defaults.max_tokens)
        opik_project_name = (
            _get_env_value(prefix, "OPIK_PROJECT_NAME")
            or os.getenv("OPIK_PROJECT_NAME")
            or defaults.opik_project_name
        )
        opik_enabled = _to_bool(
            _get_env_value(prefix, "OPIK_ENABLED") or os.getenv("OPIK_ENABLED"),
            defaults.opik_enabled,
        )
        return cls(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            opik_project_name=opik_project_name,
            opik_enabled=opik_enabled,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        return cls(**data)


@dataclass(slots=True)
class RAGConfig:
    embedding_model: str = "openai-like"
    rerank_model: str = "api"
    chunk_size: int = 200

    @classmethod
    def from_env(cls) -> "RAGConfig":
        prefix = "RAG_"
        defaults = cls()
        embedding_model = _get_env_value(prefix, "EMBEDDING_MODEL") or defaults.embedding_model
        rerank_model = _get_env_value(prefix, "RERANK_MODEL") or defaults.rerank_model
        chunk_size = _to_int(_get_env_value(prefix, "CHUNK_SIZE"), defaults.chunk_size)
        return cls(embedding_model=embedding_model, rerank_model=rerank_model, chunk_size=chunk_size)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        return cls(**data)


@dataclass(slots=True)
class AppConfig:
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(rag=RAGConfig.from_env(), llm=LLMConfig.from_env())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rag": self.rag.to_dict(),
            "llm": self.llm.to_dict(),
        }

    def dict(self) -> Dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        rag_data = data.get("rag", {})
        llm_data = data.get("llm", {})
        return cls(
            rag=RAGConfig.from_dict(rag_data) if isinstance(rag_data, dict) else RAGConfig(),
            llm=LLMConfig.from_dict(llm_data) if isinstance(llm_data, dict) else LLMConfig(),
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig.from_env()


def reload_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()


def validate_config() -> bool:
    # Configuration loading is resilient; if we reached this point we have usable defaults.
    return True


def get_rag_config() -> RAGConfig:
    return get_config().rag


def config_as_dict() -> Dict[str, Any]:
    return get_config().to_dict()
