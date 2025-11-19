from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()


def _to_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
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
class AppConfig:
    rag_embedding_model: str = "openai-like"
    rag_rerank_model: str = "api"
    rag_chunk_size: int = 200
    llm_model: str = "deepseek-chat"
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    llm_opik_project_name: Optional[str] = None
    llm_opik_enabled: bool = True
    cli_tool_timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "AppConfig":
        defaults = cls()
        rag_embedding_model = os.getenv("EMBEDDING_MODEL") or defaults.rag_embedding_model
        rag_rerank_model = os.getenv("RERANK_MODEL") or defaults.rag_rerank_model
        rag_chunk_size = _to_int(os.getenv("CHUNK_SIZE"), defaults.rag_chunk_size)

        llm_model = os.getenv("MODEL") or defaults.llm_model
        llm_api_base = os.getenv("API_BASE") or defaults.llm_api_base
        llm_api_key = os.getenv("API_KEY") or defaults.llm_api_key
        llm_temperature = _to_float(os.getenv("TEMPERATURE"), defaults.llm_temperature)
        llm_max_tokens = _to_int(os.getenv("MAX_TOKENS"), defaults.llm_max_tokens)
        llm_opik_project_name = os.getenv("OPIK_PROJECT_NAME") or defaults.llm_opik_project_name
        llm_opik_enabled = _to_bool(os.getenv("OPIK_ENABLED"), defaults.llm_opik_enabled)

        cli_timeout_candidate = os.getenv("TOOL_TIMEOUT_SECONDS")
        cli_tool_timeout_seconds = _to_int(cli_timeout_candidate, defaults.cli_tool_timeout_seconds)
        if cli_tool_timeout_seconds <= 0:
            cli_tool_timeout_seconds = defaults.cli_tool_timeout_seconds

        return cls(
            rag_embedding_model=rag_embedding_model,
            rag_rerank_model=rag_rerank_model,
            rag_chunk_size=rag_chunk_size,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_opik_project_name=llm_opik_project_name,
            llm_opik_enabled=llm_opik_enabled,
            cli_tool_timeout_seconds=cli_tool_timeout_seconds,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig.from_env()


def reload_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()
