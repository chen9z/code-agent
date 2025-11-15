from __future__ import annotations

import os
from typing import Generator, List, Dict, Any
from configs.config import get_config

try:
    from opik.integrations.openai import track_openai
except ImportError:  # pragma: no cover - optional dependency guard
    track_openai = None  # type: ignore[assignment]

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
            model: str,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]],
            tool_choice: str | Dict[str, Any] | None = None,
            temperature: float | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAICompatLLMClient(BaseLLMClient):
    def __init__(
            self,
            api_key: str,
            base_url: str | None = None,
            *,
            temperature: float = 0.7,
            opik_project_name: str | None = None,
            opik_enabled: bool = True,
    ):
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        if track_openai and opik_enabled:
            opik_url_override = os.getenv("OPIK_URL_OVERRIDE")
            opik_base_url = os.getenv("OPIK_BASE_URL")
            if not opik_url_override and opik_base_url:
                os.environ["OPIK_URL_OVERRIDE"] = opik_base_url
            project_name = opik_project_name or os.getenv("OPIK_PROJECT_NAME")
            if project_name:
                os.environ["OPIK_PROJECT_NAME"] = project_name
            client = track_openai(client, project_name=project_name)

        self.client = client
        self.default_temperature = temperature
        self.opik_enabled = bool(track_openai and opik_enabled)

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
            tool_choice="auto",
            parallel_tool_calls: bool = True,
            temperature=0.7,
    ) -> Any:
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )


def get_default_llm_client() -> BaseLLMClient:
    """Construct default LLM client using centralized config.

    Precedence:
    1) LLM_* from configs.manager (recommended)
    2) OPENAI_API_KEY / OPENAI_API_BASE environment variables (fallback)
    """
    cfg = get_config()

    api_key = cfg.llm_api_key or os.getenv("OPENAI_API_KEY")
    base_url = cfg.llm_api_base or os.getenv("OPENAI_API_BASE")

    if not api_key:
        raise RuntimeError(
            "LLM configuration missing: set configs.manager llm.api_key or OPENAI_API_KEY environment variable."
        )
    return OpenAICompatLLMClient(
        api_key=api_key,
        base_url=base_url,
        temperature=cfg.llm_temperature or 0.7,
        opik_project_name=cfg.llm_opik_project_name,
        opik_enabled=cfg.llm_opik_enabled,
    )


if __name__ == "__main__":
    client = get_default_llm_client()
    chunks = client.get_response(
        "deepseek-chat",
        [{"role": "user", "content": "hi"}],
        stream=False,  # 默认为 False，这里写明更直观
    )
    text = "".join(chunks)
    print(text)
