from __future__ import annotations

import os
from typing import Generator, List, Dict, Any
from config.manager import get_config


class BaseLLMClient:
    def get_response(self, model: str, messages: List[Dict[str, str]], stream: bool = False) -> Generator[str, None, None]:
        raise NotImplementedError


class StubLLMClient(BaseLLMClient):
    """Offline-friendly stub for environments without API credentials."""

    def get_response(self, model: str, messages: List[Dict[str, str]], stream: bool = False) -> Generator[str, None, None]:
        content = "\n".join([m.get("content", "") for m in messages])
        note = (
            "[stub-llm] No OPENAI_API_KEY/BASE configured. "
            "Returning a placeholder answer based on provided context length.\n"
        )
        answer = note + (content[:500] + ("..." if len(content) > 500 else ""))
        if stream:
            for i in range(0, len(answer), 64):
                yield answer[i : i + 64]
        else:
            yield answer


class OpenAICompatLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_response(self, model: str, messages: List[Dict[str, str]], stream: bool = False) -> Generator[str, None, None]:
        if stream:
            r = self.client.chat.completions.create(model=model, messages=messages, stream=True)
            for chunk in r:
                delta = getattr(getattr(chunk, "choices", [{}])[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    yield delta.content
        else:
            r = self.client.chat.completions.create(model=model, messages=messages)
            yield r.choices[0].message.content


def get_default_llm_client() -> BaseLLMClient:
    """Construct default LLM client using centralized config.

    Precedence:
    1) LLM_* from config.manager (recommended)
    2) OPENAI_API_KEY / OPENAI_API_BASE environment variables (fallback)
    """
    cfg = get_config()
    llm_cfg = cfg.llm

    api_key = llm_cfg.api_key or os.getenv("OPENAI_API_KEY")
    base_url = llm_cfg.api_base or os.getenv("OPENAI_API_BASE")

    if not api_key:
        return StubLLMClient()
    return OpenAICompatLLMClient(api_key=api_key, base_url=base_url)
