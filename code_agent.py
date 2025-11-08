"""Code Agent orchestration without the legacy Flow/Node runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from cli.code_agent_cli import (
    run_cli_main as _run_cli_main,
    run_code_agent_cli as _run_code_agent_cli,
    run_code_agent_once as _run_code_agent_once,
)
from cli.rich_output import (
    create_rich_output,
    preview_payload as _preview_payload,
    stringify_payload as _stringify_payload,
)
from clients.llm import get_default_llm_client
from configs.manager import get_config
from core.prompt import (
    SECURITY_SYSTEM_PROMPT,
    _BASE_SYSTEM_PROMPT,
    _SUMMARY_INSTRUCTIONS,
    compose_system_prompt,
)
from core.tool_output_store import ToolOutputStore
from nodes.tool_execution import ToolExecutionRunner, ToolOutput
from tools.registry import ToolRegistry, create_default_registry

if TYPE_CHECKING:
    from rich.console import Console


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


def _emit(output_callback: Optional[Callable[[str], None]], message: str) -> None:
    if callable(output_callback):
        output_callback(message)


def _prepare_history(
    existing_history: Optional[Iterable[Mapping[str, Any]]],
    system_prompt: str,
    user_input: str,
) -> List[Dict[str, Any]]:
    history = [dict(message) for message in (existing_history or [])]
    if not history:
        history = [{"role": "system", "content": system_prompt}]
    elif history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system_prompt})
    else:
        history[0] = {"role": "system", "content": system_prompt}

    if user_input:
        history.append({"role": "user", "content": user_input})

    if history:
        filtered = [history[0]] + [m for m in history[1:] if m.get("role") != "system"]
    else:
        filtered = history
    return filtered


class ToolPlanner:
    """Uses the LLM to produce a tool execution plan."""

    def __init__(self, registry: ToolRegistry, model: str, llm_client=None) -> None:
        self.registry = registry
        self.model = model
        self.llm = llm_client or get_default_llm_client()

    def plan(
        self,
        history: List[Dict[str, Any]],
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        tools = list(self.registry.to_openai_tools())
        response = self.llm.create_with_tools(
            model=self.model,
            messages=history,
            tools=tools,
            parallel_tool_calls=True,
        )
        plan = self._parse_tool_response(response)
        plan["raw_text"] = self._extract_raw_response(response)

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": plan.get("thoughts")
            or f"Tool plan: {json.dumps(plan.get(tool_calls, []), ensure_ascii=False)}",
        }

        tool_calls = plan.get("tool_calls") or []
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call.get("key", ""),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for call in tool_calls
                if call.get("key")
            ]
        history.append(assistant_message)
        thoughts = assistant_message.get("content")
        if thoughts:
            _emit(output_callback, f"[assistant:planner] {thoughts}")
        for call in tool_calls:
            name = call.get("key")
            if not name:
                continue
            args_preview = _preview_payload(call.get("arguments") or {}, 180)
            message_body = name
            if args_preview and args_preview not in {"{}", "null"}:
                message_body += f" | args: {args_preview}"
            _emit(output_callback, f"[plan] {message_body}")
        return plan

    @staticmethod
    def _parse_tool_response(response: Any) -> Dict[str, Any]:
        message = ToolPlanner._extract_message(response)
        if not message:
            return {"tool_calls": [], "final_response": None, "thoughts": None}

        tool_calls = ToolPlanner._extract_tool_calls(message)
        content_text = ToolPlanner._message_content_to_str(
            ToolPlanner._get_attr(message, "content")
        )

        if tool_calls:
            normalized_calls: List[Dict[str, Any]] = []
            for idx, call in enumerate(tool_calls):
                function = ToolPlanner._get_attr(call, "function", {})
                name = ToolPlanner._get_attr(function, "name")
                arguments_str = ToolPlanner._get_attr(function, "arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    arguments = {}
                normalized_calls.append(
                    {
                        "id": ToolPlanner._get_attr(call, "id", f"call-{idx}"),
                        "key": str(name) if name else "",
                        "arguments": arguments,
                    }
                )
            return {
                "tool_calls": [c for c in normalized_calls if c["key"]],
                "final_response": None,
                "thoughts": content_text,
            }

        return {
            "tool_calls": [],
            "final_response": content_text,
            "thoughts": content_text,
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
        tool_calls = ToolPlanner._get_attr(message, "tool_calls")
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
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content)

    @staticmethod
    def _extract_raw_response(response: Any) -> str:
        try:
            if hasattr(response, "model_dump_json"):
                return response.model_dump_json()
            if hasattr(response, "model_dump"):
                return json.dumps(response.model_dump(), ensure_ascii=False)
            if isinstance(response, dict):
                return json.dumps(response, ensure_ascii=False)
            return str(response)
        except Exception:  # pragma: no cover - diagnostics only
            return str(response)

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)


class SummaryBuilder:
    """Creates the final natural language response using the LLM."""

    def __init__(self, model: str, llm_client=None, system_prompt: str = _BASE_SYSTEM_PROMPT) -> None:
        self.model = model
        self.llm = llm_client or get_default_llm_client()
        self.system_prompt = system_prompt

    def summarize(
        self,
        history: List[Dict[str, Any]],
        tool_results: List[Any],
        tool_plan: Mapping[str, Any],
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        trimmed_history = self._trim_history(history)
        results_summary_lines: List[str] = []
        for result in tool_results:
            if isinstance(result, ToolOutput):
                key = result.key
                if result.error:
                    results_summary_lines.append(
                        f"Tool {key} failed with error: {result.error}"
                    )
                else:
                    payload = result.result.data if result.result else None
                    results_summary_lines.append(
                        f"Tool {key} succeeded with result: {_preview_payload(payload, 400)}"
                    )
                continue

            key = result.get("key") if isinstance(result, dict) else None
            error_payload = result.get("error") if isinstance(result, dict) else None
            if error_payload:
                results_summary_lines.append(
                    f"Tool {key} failed with error: {error_payload}"
                )
            else:
                payload = result.get("result") if isinstance(result, dict) else result
                results_summary_lines.append(
                    f"Tool {key} succeeded with result: {_preview_payload(payload, 400)}"
                )
        if not results_summary_lines:
            results_summary_lines.append("No tool results were generated.")

        plan_final = tool_plan.get("final_response") if isinstance(tool_plan, Mapping) else None
        summary_prompt = (
            f"{_SUMMARY_INSTRUCTIONS}\n\n"
            f"Latest user request:\n{self._latest_user_content(history)}\n\n"
            f"Tool outcomes:\n- " + "\n- ".join(results_summary_lines)
        )
        if plan_final:
            summary_prompt += f"\n\nPlanner suggestion:\n{plan_final}"

        messages = self._build_messages(trimmed_history, summary_prompt)
        chunks = list(self.llm.get_response(model=self.model, messages=messages, stream=False))
        final_response = "".join(chunks).strip()

        history.append({"role": "assistant", "content": final_response})
        if final_response:
            _emit(output_callback, f"[assistant] {final_response}")
        return final_response

    def _build_messages(
        self,
        history: List[Dict[str, Any]],
        summary_prompt: str,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for message in history:
            if message.get("role") == "system":
                continue
            messages.append(message)
        messages.append({"role": "user", "content": summary_prompt})
        return messages

    @staticmethod
    def _latest_user_content(history: List[Dict[str, Any]]) -> str:
        for message in reversed(history):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return "(no direct user input captured)"

    @staticmethod
    def _trim_history(
        history: List[Dict[str, Any]],
        *,
        max_chars: int = 6000,
        max_messages: int = 12,
    ) -> List[Dict[str, Any]]:
        trimmed: List[Dict[str, Any]] = []
        remaining_chars = max_chars
        for message in reversed(history):
            content = message.get("content", "")
            content_str = _stringify_payload(content)
            remaining_chars -= len(content_str)
            trimmed.append(message)
            if len(trimmed) >= max_messages or remaining_chars <= 0:
                break
        trimmed.reverse()
        return SummaryBuilder._ensure_history_coherence(history, trimmed)

    @staticmethod
    def _ensure_history_coherence(
        full_history: List[Dict[str, Any]],
        trimmed: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        coherent = list(trimmed)
        latest_user = None
        for message in reversed(full_history):
            if message.get("role") == "user":
                latest_user = message
                break
        if latest_user and latest_user not in coherent:
            coherent.insert(0, latest_user)

        idx = 0
        while idx < len(coherent):
            message = coherent[idx]
            if message.get("role") == "tool":
                if not SummaryBuilder._has_preceding_tool_parent(coherent, idx, message.get("tool_call_id")):
                    parent = SummaryBuilder._find_tool_parent(full_history, message.get("tool_call_id"))
                    if parent and parent not in coherent:
                        coherent.insert(idx, parent)
                        idx += 1
                        continue
            idx += 1
        return coherent

    @staticmethod
    def _has_preceding_tool_parent(
        messages: List[Dict[str, Any]],
        idx: int,
        tool_call_id: Any,
    ) -> bool:
        if idx <= 0:
            return False
        parent = messages[idx - 1]
        if parent.get("role") != "assistant":
            return False
        for call in parent.get("tool_calls") or []:
            if call.get("id") == tool_call_id:
                return True
        return False

    @staticmethod
    def _find_tool_parent(
        full_history: List[Dict[str, Any]],
        tool_call_id: Any,
    ) -> Optional[Dict[str, Any]]:
        if not tool_call_id:
            return None
        for message in reversed(full_history):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if call.get("id") == tool_call_id:
                    return message
        return None


class CodeAgentSession:
    """In-memory conversation session for the Code Agent CLI."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        llm_client: Any = None,
        max_iterations: int = 25,
        system_prompt: Optional[str] = None,
        environment: Optional[Mapping[str, Any]] = None,
        workspace: Optional[str | Path] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        self.registry = registry or create_default_registry(project_root=self.workspace)
        self.llm_client = llm_client or get_default_llm_client()
        self.max_iterations = max_iterations if max_iterations >= 1 else 1
        if environment is not None:
            self.environment = environment
        elif self.workspace is not None:
            self.environment = {"cwd": str(self.workspace)}
        else:
            self.environment = None
        cfg = get_config()
        default_model = cfg.llm.model
        self.tool_timeout_seconds = float(cfg.cli.tool_timeout_seconds)
        if system_prompt is None:
            resolved_prompt = build_code_agent_system_prompt(
                base_prompt=_BASE_SYSTEM_PROMPT,
                environment=self.environment,
            )
        else:
            resolved_prompt = system_prompt
        self.system_prompt = resolved_prompt
        self.planner = ToolPlanner(self.registry, model=default_model, llm_client=self.llm_client)
        self.executor = ToolExecutionRunner(
            self.registry,
            default_timeout_seconds=self.tool_timeout_seconds,
        )
        self.summarizer = SummaryBuilder(
            model=default_model,
            llm_client=self.llm_client,
            system_prompt=self.system_prompt,
        )
        self.history: List[Dict[str, Any]] = []
        self.tool_output_store = ToolOutputStore()

    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty")
        trimmed_input = user_input.strip()
        history = _prepare_history(self.history, self.system_prompt, trimmed_input)
        _emit(output_callback, f"[user] {trimmed_input}")

        tool_results: List[ToolOutput] = []
        tool_plan: Dict[str, Any] = {}
        iterations = 0

        while True:
            plan = self.planner.plan(history, output_callback)
            tool_plan = plan
            tool_calls = plan.get("tool_calls") or []
            if not tool_calls:
                break
            outputs = self.executor.run(
                tool_calls,
                history=history,
                output_callback=output_callback,
                store=self.tool_output_store,
                timeout_override=self.tool_timeout_seconds,
            )
            tool_results.extend(outputs)
            iterations += 1
            if iterations >= self.max_iterations:
                break

        final_response = self.summarizer.summarize(
            history,
            tool_results,
            tool_plan,
            output_callback,
        )

        result = {
            "final_response": final_response,
            "tool_results": list(tool_results),
            "tool_plan": tool_plan,
            "history": list(history),
        }
        self.history = [
            self._normalize_message(msg) for msg in history if self._is_valid_message(msg)
        ]
        return result

    def set_tool_timeout_seconds(self, seconds: Optional[float]) -> None:
        if seconds is None or seconds <= 0:
            return
        self.tool_timeout_seconds = float(seconds)
        self.executor.set_default_timeout(self.tool_timeout_seconds)

    def get_tool_output_store(self) -> ToolOutputStore:
        return self.tool_output_store

    @staticmethod
    def _is_valid_message(raw: Any) -> bool:
        return isinstance(raw, dict) and isinstance(raw.get("role"), str) and "content" in raw

    @staticmethod
    def _normalize_message(raw: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(raw)
        normalized["role"] = str(normalized.get("role"))
        content = normalized.get("content", "")
        if isinstance(content, str):
            normalized["content"] = content
        elif content is None:
            normalized["content"] = ""
        else:
            normalized["content"] = content
        return normalized


def run_code_agent_cli(
        *,
        session: Optional["CodeAgentSession"] = None,
        session_factory: Optional[Callable[[], "CodeAgentSession"]] = None,
        input_iter: Optional[Iterable[str]] = None,
        output_callback: Optional[Callable[[str], None]] = None,
        console: Optional["Console"] = None,
) -> int:
    factory = session_factory or (lambda: CodeAgentSession())
    return _run_code_agent_cli(
        session=session,
        session_factory=factory,
        input_iter=input_iter,
        output_callback=output_callback,
        console=console,
    )


def run_code_agent_once(
        prompt: str,
        *,
        session: Optional["CodeAgentSession"] = None,
        session_factory: Optional[Callable[[], "CodeAgentSession"]] = None,
        output_callback: Callable[[str], None],
) -> Dict[str, Any]:
    factory = session_factory or (lambda: CodeAgentSession())
    return _run_code_agent_once(
        prompt,
        session=session,
        session_factory=factory,
        output_callback=output_callback,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for running the code agent directly."""

    return _run_cli_main(
        argv,
        session_factory=lambda: CodeAgentSession(max_iterations=100),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CodeAgentSession",
    "build_code_agent_system_prompt",
    "create_rich_output",
    "run_code_agent_cli",
    "run_code_agent_once",
    "run_cli_main",
]
