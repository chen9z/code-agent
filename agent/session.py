"""Code Agent orchestration without the legacy Flow/Node runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ui.emission import OutputCallback, OutputMessage, create_emit_event
from adapters.llm.llm import get_default_llm_client
from config.config import get_config
from config.prompt import (
    SECURITY_SYSTEM_PROMPT,
    _BASE_SYSTEM_PROMPT,
    compose_system_prompt,
)
from runtime.tool_runner import ToolExecutionRunner, ToolResult
from tools.registry import ToolRegistry, create_default_registry
from opik import track as opik_track


def build_code_agent_system_prompt(
        *,
        base_prompt: str = _BASE_SYSTEM_PROMPT,
        environment: Optional[Mapping[str, Any]] = None,
        include_security_prompt: bool = True,
) -> str:
    """Compose the system prompt using the shared helper."""

    sections: List[str] = []
    if include_security_prompt:
        sections.append(SECURITY_SYSTEM_PROMPT)
    return compose_system_prompt(base_prompt, extra_sections=sections, environment=environment)


def _emit(output_callback: Optional[OutputCallback], message: OutputMessage) -> None:
    if callable(output_callback):
        output_callback(message)


def _prepare_messages(
        existing_history: Optional[Iterable[Mapping[str, Any]]],
        system_prompt: str,
        user_input: str,
) -> List[Dict[str, Any]]:
    cloned_history = [dict(message) for message in (existing_history or [])]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    messages.extend(message for message in cloned_history if message.get("role") != "system")

    if user_input:
        messages.append({"role": "user", "content": user_input})

    return messages


class CodeAgentSession:
    """In-memory conversation session for the Code Agent CLI."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        registry_factory: Optional[Callable[[Optional[Iterable[str]]], ToolRegistry]] = None,
        tool_allowlist: Optional[Iterable[str]] = None,
        llm_client: Any = None,
        max_iterations: int = 25,
        system_prompt: Optional[str] = None,
        environment: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        workspace: Optional[str | Path] = None,
        temperature: Optional[float] = None,
        tool_timeout_seconds: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None

        # registry construction: prefer explicit instance, otherwise factory, otherwise default
        if registry is not None:
            self.registry = registry
        else:
            allow: Optional[Sequence[str]] = list(tool_allowlist) if tool_allowlist is not None else None
            if registry_factory:
                self.registry = registry_factory(allow)
            else:
                self.registry = create_default_registry(
                    include=allow,
                    project_root=self.workspace,
                )

        self.llm_client = llm_client or get_default_llm_client()
        self.max_iterations = max_iterations if max_iterations >= 1 else 1

        base_env: Dict[str, Any] = {}
        if environment is not None:
            base_env.update(environment)
        elif self.workspace is not None:
            base_env["cwd"] = str(self.workspace)
        if context:
            base_env.update(context)
        self.environment = base_env if base_env else None

        cfg = get_config()
        self.model = cfg.llm_model
        self.temperature = float(temperature) if temperature is not None else float(cfg.llm_temperature)
        self.tool_timeout_seconds = (
            float(tool_timeout_seconds)
            if tool_timeout_seconds is not None and tool_timeout_seconds > 0
            else float(cfg.cli_tool_timeout_seconds)
        )
        self._opik_project_name = cfg.llm_opik_project_name
        self._opik_track = opik_track if cfg.llm_opik_enabled else None

        if system_prompt is None:
            resolved_prompt = build_code_agent_system_prompt(
                base_prompt=_BASE_SYSTEM_PROMPT,
                environment=self.environment,
            )
        else:
            resolved_prompt = system_prompt
        self.system_prompt = resolved_prompt
        self.verbose = verbose
        self.executor = ToolExecutionRunner(
            self.registry,
            default_timeout_seconds=self.tool_timeout_seconds,
            opik_track_enabled=bool(self._opik_track),
            opik_project_name=self._opik_project_name,
        )
        self.messages: List[Dict[str, Any]] = []

    def run_turn(
            self,
            user_input: str,
            *,
            output_callback: Optional[OutputCallback] = None,
    ) -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty")

        runner = self._run_turn_impl
        if self._opik_track:
            metadata = self._build_turn_metadata(user_input)
            runner = self._opik_track(
                name="code_agent_turn",
                type="general",
                tags=["code-agent", "session"],
                metadata=metadata,
                project_name=self._opik_project_name,
            )(runner)
        return runner(user_input=user_input, output_callback=output_callback)

    def _run_turn_impl(
            self,
            user_input: str,
            *,
            output_callback: Optional[OutputCallback] = None,
    ) -> Dict[str, Any]:
        messages = _prepare_messages(self.messages, self.system_prompt, user_input)
        _emit(output_callback, create_emit_event("user", user_input))
        self._log_message("user", user_input)

        tool_results: List[ToolResult] = []
        iterations = 0

        final_content: Optional[str] = None
        while True:
            response = self._call_llm(messages)
            assistant_content = response.get("content")
            if assistant_content:
                _emit(output_callback, create_emit_event("assistant", assistant_content))
                self._log_message("assistant", assistant_content)
            tool_calls = response.get("tool_calls")
            if not tool_calls:
                final_content = assistant_content
                break
            outputs = self.executor.run(
                tool_calls,
                messages=messages,
                output_callback=output_callback,
                timeout_override=self.tool_timeout_seconds,
            )
            tool_results.extend(outputs)
            if self.verbose and outputs:
                for result in outputs:
                    info = result.as_dict()
                    name = info.get("inputs", {}).get("name")
                    status = info.get("result", {}).get("status")
                    args = info.get("inputs", {}).get("arguments")
                    self._debug(f"tool={name} status={status} args={args}")
            iterations += 1
            if iterations >= self.max_iterations:
                break

        result = {
            "content": final_content,
            "tool_results": list(tool_results),
            "messages": list(messages),
        }
        self.messages = [msg for msg in messages if self._is_valid_message(msg)]
        if self.verbose and final_content:
            self._debug("final response captured for turn")
        return result

    def _build_turn_metadata(self, user_input: str) -> Dict[str, Any]:
        content = user_input.strip()
        env = None
        if self.environment:
            env = {
                str(key): self._stringify_env_value(value)
                for key, value in self.environment.items()
            }
        metadata: Dict[str, Any] = {
            "model": self.model,
            "workspace": str(self.workspace) if self.workspace else None,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "history_length": len(self.messages),
            "input": content,
            "environment": env,
        }
        return {
            key: value
            for key, value in metadata.items()
            if value not in (None, "", {})
        }

    def _call_llm(
            self,
            messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tools = list(self.registry.to_openai_tools())
        response = self.llm_client.create_with_tools(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
        )
        if self.verbose:
            self._debug(f"LLM call with {len(messages)} messages")
            last = messages[-1]
            self._debug(f"LLM latest message role={last.get('role')} content={self._preview(last.get('content'))}")
        plan = self._parse_tool_response(response)

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": plan.get("content")
        }

        tool_calls = plan.get("tool_calls") or []
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call.get("name"),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for call in tool_calls
                if call.get("name")
            ]
        messages.append(assistant_message)
        return plan

    def set_tool_timeout_seconds(self, seconds: Optional[float]) -> None:
        if seconds is None or seconds <= 0:
            return
        self.tool_timeout_seconds = float(seconds)
        self.executor.set_default_timeout(self.tool_timeout_seconds)

    @staticmethod
    def _stringify_env_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        return value

    def _debug(self, message: str) -> None:
        print(f"[CodeAgentSession] {message}")

    def _log_message(self, role: str, content: Optional[str]) -> None:
        if not self.verbose or content is None:
            return
        preview = content if len(content) <= 500 else content[:497] + "..."
        self._debug(f"{role}: {preview}")

    def _preview(self, content: Optional[str]) -> Optional[str]:
        if content is None:
            return None
        return content if len(content) <= 500 else content[:497] + "..."

    @staticmethod
    def _is_valid_message(raw: Any) -> bool:
        return isinstance(raw, dict) and isinstance(raw.get("role"), str) and "content" in raw

    @staticmethod
    def _parse_tool_response(response: Any) -> Dict[str, Any]:
        message = CodeAgentSession._extract_message(response)
        if not message:
            return {"tool_calls": [], "content": None}

        tool_calls = CodeAgentSession._extract_tool_calls(message)
        content_text = CodeAgentSession._message_content_to_str(
            CodeAgentSession._get_attr(message, "content")
        )

        if tool_calls:
            normalized_calls: List[Dict[str, Any]] = []
            for idx, call in enumerate(tool_calls):
                function = CodeAgentSession._get_attr(call, "function", {})
                name = CodeAgentSession._get_attr(function, "name")
                arguments_str = CodeAgentSession._get_attr(function, "arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    arguments = {}
                normalized_calls.append(
                    {
                        "id": CodeAgentSession._get_attr(call, "id", f"call-{idx}"),
                        "name": str(name) if name else "",
                        "arguments": arguments,
                    }
                )
            return {
                "tool_calls": [c for c in normalized_calls if c["name"]],
                "content": content_text,
            }

        return {
            "tool_calls": [],
            "content": content_text,
        }

    @staticmethod
    def _extract_message(response: Any) -> Any:
        if response is None:
            return None
        if hasattr(response, "choices"):
            choices = getattr(response, "choices", [])
            if choices:
                return getattr(choices[0], "message", None)
        if isinstance(response, dict):
            choices = response.get("choices")
            if choices:
                message = choices[0].get("message")
                if message is not None:
                    return message
        return None

    @staticmethod
    def _extract_tool_calls(message: Any) -> List[Any]:
        tool_calls = CodeAgentSession._get_attr(message, "tool_calls")
        if tool_calls is None:
            return []
        if isinstance(tool_calls, list):
            return tool_calls
        return []

    @staticmethod
    def _message_content_to_str(content: Any) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
