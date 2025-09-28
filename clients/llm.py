from __future__ import annotations

import os
from typing import Generator, List, Dict, Any
from configs.manager import get_config


class BaseLLMClient:
    def get_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        raise NotImplementedError

    def create_with_tools(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | Dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAICompatLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str | None = None, *, temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_temperature = temperature

    def get_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        stream: bool = False,
    ) -> Generator[str, None, None]:
        resolved_temp = self.default_temperature if temperature is None else temperature
        if stream:
            r = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=resolved_temp,
            )
            for chunk in r:
                delta = getattr(getattr(chunk, "choices", [{}])[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    yield delta.content
        else:
            r = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=resolved_temp,
            )
            yield r.choices[0].message.content

    def create_with_tools(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | Dict[str, Any] | None = None,
        parallel_tool_calls: bool = True,
        temperature: float | None = None,
    ) -> Any:
        resolved_temp = self.default_temperature if temperature is None else temperature
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice or "auto",
            parallel_tool_calls=parallel_tool_calls,
            temperature=resolved_temp,
        )


def get_default_llm_client() -> BaseLLMClient:
    """Construct default LLM client using centralized config.

    Precedence:
    1) LLM_* from configs.manager (recommended)
    2) OPENAI_API_KEY / OPENAI_API_BASE environment variables (fallback)
    """
    cfg = get_config()
    llm_cfg = cfg.llm

    api_key = llm_cfg.api_key or os.getenv("OPENAI_API_KEY")
    base_url = llm_cfg.api_base or os.getenv("OPENAI_API_BASE")

    if not api_key:
        raise RuntimeError(
            "LLM configuration missing: set configs.manager llm.api_key or OPENAI_API_KEY environment variable."
        )
    return OpenAICompatLLMClient(api_key=api_key, base_url=base_url, temperature=llm_cfg.temperature or 0.0)


if __name__ == "__main__":
    client = get_default_llm_client()
    chunks = client.get_response(
        "deepseek-chat",
        [{"role": "user", "content": "hi"}],
        stream=False,  # 默认为 False，这里写明更直观
    )
    text = "".join(chunks)
    print(text)