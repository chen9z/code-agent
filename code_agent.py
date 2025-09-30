"""Code Agent flow and CLI orchestrator built on the Flow/Node runtime."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from rich.console import Console
from rich.text import Text

from __init__ import Flow, FlowCancelledError, Node
from clients.llm import get_default_llm_client
from configs.manager import get_config
from nodes.tool_execution import ToolExecutionBatchNode
from tools.registry import ToolRegistry, create_default_registry

_BASE_SYSTEM_PROMPT = (
    "You are Code Agent, an autonomous software assistant operating inside the user's "
    "current workspace. Stay within the provided project directory, avoid inspecting the "
    "filesystem root, and prefer targeted searches over broad scans. Maintain the "
    "conversation history, minimise redundant tool calls, and when finished produce a "
    "concise natural language answer that cites the evidence you gathered."
)

# JSON-plan fallback removed; we rely on native tool-calling.

_SUMMARY_INSTRUCTIONS = (
    "Provide the final answer to the user using the available context. Reference tool "
    "results when they exist and be explicit about any limitations."
)


_RICH_STYLE_MAP = {
    "assistant": "white",
    "assistant:planner": "white",
    "planner": "white",
    "plan": "green",
    "tool": "green",
    "user": "bold white",
    "system": "magenta",
    "warning": "yellow",
}


def create_rich_output(console: Optional[Console] = None) -> Callable[[Any], None]:
    """Return a callback that renders agent events using a Rich console."""

    active_console = console or Console()

    def emit(message: Any) -> None:
        if message is None:
            return
        if isinstance(message, Mapping):
            text = _stringify_payload(message)
        else:
            text = message if isinstance(message, str) else _stringify_payload(message)
        tag, body = _split_message_tag(text)
        if tag is None:
            active_console.print(Text(str(text)))
            active_console.print()
            return

        normalized_tag = tag.lower()
        if normalized_tag == "user":
            line = Text.assemble(
                Text("> ", style="bold white"),
                Text(body, style="bold white"),
            )
            active_console.print(line)
            active_console.print()
            return

        if normalized_tag == "system":
            style = _RICH_STYLE_MAP.get(normalized_tag, "magenta")
            active_console.print(Text(body, style=style))
            active_console.print()
            return

        if normalized_tag in {"assistant", "assistant:planner", "planner"}:
            _render_bullet(active_console, body, [], "white", "white")
            return

        if normalized_tag == "plan":
            return

        if normalized_tag == "tool":
            header, metadata = _parse_structured_body(body)
            status = _extract_metadata_value(metadata, "status") or "success"
            bullet_style = "green" if status.lower() == "success" else "red"
            header_style = "bold green" if status.lower() == "success" else "bold red"
            _render_bullet(active_console, header, metadata, bullet_style, header_style)
            return

        style = _RICH_STYLE_MAP.get(normalized_tag, "white")
        _render_bullet(active_console, body, [], style, style)

    return emit


def _split_message_tag(message: str) -> tuple[Optional[str], str]:
    stripped = message.strip()
    if not stripped.startswith("["):
        return None, message
    closing = stripped.find("]")
    if closing <= 1:
        return None, message
    suffix = stripped[closing + 1 :]
    if not suffix.startswith(" "):
        return None, message
    tag = stripped[1:closing]
    body = stripped[closing + 2 :]
    return tag, body


def _parse_structured_body(body: str) -> tuple[str, List[tuple[str, str]]]:
    parts = [segment.strip() for segment in body.split("|") if segment.strip()]
    if not parts:
        return body, []
    header = parts[0]
    metadata: List[tuple[str, str]] = []
    for segment in parts[1:]:
        if ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        metadata.append((key.strip(), value.strip()))
    return header, metadata


def _extract_metadata_value(metadata: List[tuple[str, str]], key: str) -> Optional[str]:
    for meta_key, value in metadata:
        if meta_key.lower() == key.lower():
            return value
    return None


def _render_bullet(
    console: Console,
    header: str,
    metadata: List[tuple[str, str]],
    bullet_style: str,
    header_style: str,
) -> None:
    bullet = Text("● ", style=bullet_style)
    header_text = Text(header, style=header_style)
    console.print(Text.assemble(bullet, header_text))

    meta_lines = _format_metadata(metadata)
    for idx, line in enumerate(meta_lines):
        prefix = Text("└ " if idx == 0 else "  ", style="dim")
        console.print(Text.assemble(prefix, line))

    console.print()


def _format_metadata(metadata: List[tuple[str, str]]) -> List[Text]:
    lines: List[Text] = []
    for key, value in metadata:
        normalized = key.lower()
        value_text = value
        if not value_text:
            continue
        if normalized == "status":
            continue
        if normalized == "args":
            line = Text(f"args: {value_text}", style="dim")
        elif normalized in {"output", "result"}:
            line = Text(value_text, style="white")
        elif normalized == "error":
            line = Text(value_text, style="red")
        else:
            line = Text(f"{key}: {value_text}", style="dim")
        lines.append(line)
    return lines


def _stringify_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _preview_payload(value: Any, limit: int) -> str:
    text = _stringify_payload(value)
    if limit <= 3 or len(text) <= limit:
        return text[:limit]
    return f"{text[: limit - 3]}..."


def _emit(shared: Dict[str, Any], message: str) -> None:
    cancel_event = shared.get("cancel_event") if isinstance(shared, dict) else None
    if cancel_event is not None:
        checker = getattr(cancel_event, "is_set", None)
        if callable(checker) and checker():
            return
        if not callable(checker) and cancel_event:
            return
    callback = shared.get("output_callback")
    if callable(callback):
        callback(message)


class ConversationContextNode(Node):
    """Initialises conversation history and appends the latest user input."""

    def __init__(self, system_prompt: str = _BASE_SYSTEM_PROMPT) -> None:
        super().__init__()
        self.system_prompt = system_prompt

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        history = list(shared.get("history") or self.params.get("history") or [])
        if not history:
            history = [{"role": "system", "content": self.system_prompt}]
        elif history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": self.system_prompt})

        user_input = self.params.get("user_input") or shared.get("user_input")
        if user_input:
            history.append({"role": "user", "content": str(user_input)})
        # Drop any additional 'system' messages beyond the first to reduce prompt injection surface.
        if history:
            filtered = [history[0]] + [m for m in history[1:] if m.get("role") != "system"]
        else:
            filtered = history
        shared["history"] = filtered
        return {"has_user_input": bool(user_input)}

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> str:
        user_input = shared.get("user_input")
        if user_input:
            _emit(shared, f"[user] {user_input}")
        return "plan"


class ToolPlanningNode(Node):
    """Uses the LLM to produce a tool execution plan."""

    def __init__(self, registry: ToolRegistry, model: str, llm_client=None) -> None:
        super().__init__()
        self.registry = registry
        self.model = model
        self.llm = llm_client or get_default_llm_client()

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        descriptors = self.registry.describe()
        history = list(shared.get("history") or [])
        return {"history": history, "descriptors": descriptors}

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        history = prep_res["history"]
        descriptors = prep_res["descriptors"]
        serialized_tools = json.dumps(descriptors, ensure_ascii=False, indent=2)
        planning_prompt = "Available tools:\n" + serialized_tools
        messages = history + [{"role": "system", "content": planning_prompt}]
        tools = self.registry.to_openai_tools()

        response = self.llm.create_with_tools(
            model=self.model,
            messages=messages,
            tools=tools,
            parallel_tool_calls=True,
        )
        plan = self._parse_tool_response(response)
        plan["raw_text"] = self._extract_raw_response(response)
        return plan

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        shared["tool_plan"] = {
            "tool_calls": exec_res.get("tool_calls", []),
            "final_response": exec_res.get("final_response"),
            "thoughts": exec_res.get("thoughts"),
            "raw_text": exec_res.get("raw_text"),
        }
        history = shared.setdefault("history", [])
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": exec_res.get("thoughts")
            or f"Tool plan: {json.dumps(exec_res.get('tool_calls', []), ensure_ascii=False)}",
        }

        tool_calls = exec_res.get("tool_calls") or []
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
            _emit(shared, f"[assistant:planner] {thoughts}")
        for call in assistant_message.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            raw_args = call.get("function", {}).get("arguments")
            parsed_args: Any = {}
            if isinstance(raw_args, str) and raw_args:
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = raw_args
            else:
                parsed_args = raw_args or {}
            args_preview = _preview_payload(parsed_args, 180)
            message_body = name or "tool"
            if args_preview and args_preview not in {"{}", "null"}:
                message_body += f" | args: {args_preview}"
            _emit(shared, f"[plan] {message_body}")
        return "execute" if tool_calls else "summarize"

    def _parse_tool_response(self, response: Any) -> Dict[str, Any]:
        message = self._extract_message(response)
        if not message:
            return {"tool_calls": [], "final_response": None, "thoughts": None}

        tool_calls = self._extract_tool_calls(message)
        content_text = self._message_content_to_str(self._get_attr(message, "content"))

        if tool_calls:
            normalized_calls: List[Dict[str, Any]] = []
            for idx, call in enumerate(tool_calls):
                function = self._get_attr(call, "function", {})
                name = self._get_attr(function, "name")
                arguments_str = self._get_attr(function, "arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    arguments = {}
                normalized_calls.append(
                    {
                        "id": self._get_attr(call, "id", f"call-{idx}"),
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

    # JSON-plan fallback removed

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
        tool_calls = ToolPlanningNode._get_attr(message, "tool_calls")
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


class SummaryNode(Node):
    """Creates the final natural language response using the LLM."""

    def __init__(self, model: str, llm_client=None) -> None:
        super().__init__()
        self.model = model
        self.llm = llm_client or get_default_llm_client()

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "history": list(shared.get("history") or []),
            "tool_results": list(shared.get("tool_results") or []),
            "tool_plan": shared.get("tool_plan") or {},
        }

    def exec(self, prep_res: Dict[str, Any]) -> str:
        history = self._trim_history(prep_res["history"])
        tool_results = prep_res["tool_results"]
        plan = prep_res["tool_plan"]

        results_summary_lines: List[str] = []
        for result in tool_results:
            key = result.get("key")
            error_payload = result.get("error")
            if error_payload:
                results_summary_lines.append(
                    f"Tool {key} failed with error: {error_payload}"
                )
            else:
                results_summary_lines.append(
                    f"Tool {key} succeeded with result: {_preview_payload(result.get('result'), 400)}"
                )
        if not results_summary_lines:
            results_summary_lines.append("No tool results were generated.")

        plan_final = plan.get("final_response")
        summary_prompt = (
            f"{_SUMMARY_INSTRUCTIONS}\n\n"
            f"Latest user request:\n{self._latest_user_content(history)}\n\n"
            f"Tool outcomes:\n- " + "\n- ".join(results_summary_lines)
        )
        if plan_final:
            summary_prompt += f"\n\nPlanner suggestion:\n{plan_final}"

        messages = self._build_messages(history, summary_prompt)
        chunks = list(self.llm.get_response(model=self.model, messages=messages, stream=False))
        return "".join(chunks).strip()

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: str) -> str:
        shared["final_response"] = exec_res
        shared.setdefault("history", []).append({"role": "assistant", "content": exec_res})
        if exec_res:
            _emit(shared, f"[assistant] {exec_res}")
        return "complete"

    @staticmethod
    def _latest_user_content(history: List[Dict[str, Any]]) -> str:
        for message in reversed(history):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return "(no direct user input captured)"

    @staticmethod
    def _trim_history(history: List[Dict[str, Any]], *, max_chars: int = 6000, max_messages: int = 12) -> List[Dict[str, Any]]:
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
        return SummaryNode._ensure_history_coherence(history, trimmed)

    @staticmethod
    def _ensure_history_coherence(
        full_history: List[Dict[str, Any]], trimmed: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        coherent = list(trimmed)

        # Always ensure the latest user message is present.
        latest_user = None
        for message in reversed(full_history):
            if message.get("role") == "user":
                latest_user = message
                break
        if latest_user and latest_user not in coherent:
            coherent.insert(0, latest_user)

        # Ensure each tool message is preceded by the assistant tool-call declaration.
        idx = 0
        while idx < len(coherent):
            message = coherent[idx]
            if message.get("role") == "tool":
                if not SummaryNode._has_preceding_tool_parent(coherent, idx, message.get("tool_call_id")):
                    parent = SummaryNode._find_tool_parent(full_history, message.get("tool_call_id"))
                    if parent and parent not in coherent:
                        coherent.insert(idx, parent)
                        idx += 1
                        continue
            idx += 1

        return coherent

    @staticmethod
    def _has_preceding_tool_parent(messages: List[Dict[str, Any]], idx: int, tool_call_id: Any) -> bool:
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
    def _find_tool_parent(full_history: List[Dict[str, Any]], tool_call_id: Any) -> Optional[Dict[str, Any]]:
        if not tool_call_id:
            return None
        for message in reversed(full_history):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if call.get("id") == tool_call_id:
                    return message
        return None

    def _build_messages(self, history: List[Dict[str, Any]], summary_prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": _BASE_SYSTEM_PROMPT}]
        for message in history:
            if message.get("role") == "system":
                continue
            messages.append(message)
        messages.append({"role": "user", "content": summary_prompt})
        return messages


class ToolAgentFlow(Flow):
    """End-to-end agent flow orchestrating tool planning, execution, and summarisation."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        llm_client=None,
        model: Optional[str] = None,
        *,
        max_iterations: int = 25,
    ) -> None:
        if model is None:
            cfg = get_config()
            model = cfg.llm.model
        self.registry = registry or create_default_registry()
        self.llm = llm_client or get_default_llm_client()
        self.model = model
        self.max_iterations = max_iterations if max_iterations >= 1 else 1

        super().__init__()
        self.context_node = ConversationContextNode()
        self.planning_node = ToolPlanningNode(self.registry, model=self.model, llm_client=self.llm)
        self.execution_node = ToolExecutionBatchNode(self.registry)
        self.summary_node = SummaryNode(model=self.model, llm_client=self.llm)

        self.execution_node.max_iterations = self.max_iterations

        self.start(self.context_node)
        self.context_node.next(self.planning_node, "plan")
        self.planning_node.next(self.execution_node, "execute")
        self.planning_node.next(self.summary_node, "summarize")
        self.execution_node.next(self.planning_node, "plan")
        self.execution_node.next(self.summary_node, "summarize")

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        history = self.params.get("history") or shared.get("history")
        if history:
            shared["history"] = list(history)
        user_input = self.params.get("user_input") or shared.get("user_input")
        if not user_input:
            raise ValueError("user_input is required for the tool agent flow")
        shared["user_input"] = user_input
        output_callback = self.params.get("output_callback") or shared.get("output_callback")
        if output_callback is not None:
            shared["output_callback"] = output_callback
        shared["tool_iterations_used"] = 0
        return {}

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> Dict[str, Any]:
        return {
            "final_response": shared.get("final_response"),
            "tool_results": shared.get("tool_results", []),
            "tool_plan": shared.get("tool_plan"),
            "history": shared.get("history", []),
        }


def create_tool_agent_flow(
    registry: Optional[ToolRegistry] = None,
    llm_client=None,
    model: Optional[str] = None,
    *,
    max_iterations: int = 25,
) -> ToolAgentFlow:
    return ToolAgentFlow(
        registry=registry,
        llm_client=llm_client,
        model=model,
        max_iterations=max_iterations,
    )


FlowFactory = Callable[[], ToolAgentFlow]


def create_code_agent_flow(
    *,
    registry: Optional[ToolRegistry] = None,
    llm_client: Any = None,
    max_parallel_workers: int = 4,
    model: Optional[str] = None,
    max_iterations: int = 25,
) -> ToolAgentFlow:
    """Return a `ToolAgentFlow` instance with all default tools registered."""

    flow = create_tool_agent_flow(
        registry=registry,
        llm_client=llm_client,
        model=model,
        max_iterations=max_iterations,
    )
    if max_parallel_workers < 1:
        raise ValueError("max_parallel_workers must be >= 1")
    if hasattr(flow, "execution_node") and hasattr(flow.execution_node, "max_parallel_workers"):
        flow.execution_node.max_parallel_workers = max_parallel_workers
    return flow


class CodeAgentSession:
    """In-memory conversation session for the Code Agent CLI."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        llm_client: Any = None,
        max_parallel_workers: int = 4,
        max_iterations: int = 25,
        flow_factory: Optional[FlowFactory] = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.llm_client = llm_client
        self.max_parallel_workers = max_parallel_workers
        self.max_iterations = max_iterations if max_iterations >= 1 else 1
        if flow_factory is not None:
            self._flow_factory = flow_factory
        else:
            default_model = get_config().llm.model
            self._flow_factory = lambda: create_code_agent_flow(
                registry=self.registry,
                llm_client=self.llm_client,
                max_parallel_workers=self.max_parallel_workers,
                max_iterations=self.max_iterations,
                model=default_model,
            )
        self.history: List[Dict[str, Any]] = []

    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty")
        token = cancel_event or threading.Event()
        token_setter = getattr(token, "is_set", None)
        if callable(token_setter) and token_setter():
            return {"cancelled": True, "history": list(self.history)}
        if not callable(token_setter) and token:
            return {"cancelled": True, "history": list(self.history)}
        flow = self._flow_factory()
        params: Dict[str, Any] = {"user_input": user_input.strip(), "cancel_event": token}
        if self.history:
            params["history"] = list(self.history)
        if output_callback is not None:
            params["output_callback"] = output_callback
        flow.set_params(params)
        shared: Dict[str, Any] = {"cancel_event": token}
        if output_callback is not None:
            shared["output_callback"] = output_callback
        try:
            result = flow._run(shared)
        except FlowCancelledError:
            return {"cancelled": True, "history": list(self.history)}
        updated = result.get("history")
        if isinstance(updated, list):
            self.history = [self._normalize_message(msg) for msg in updated if self._is_valid_message(msg)]
        else:
            self.history.extend(
                [
                    {"role": "user", "content": user_input.strip()},
                    {"role": "assistant", "content": result.get("final_response", "")},
                ]
            )
        return result

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


class _RunLoopNode(Node):
    def __init__(self, session: CodeAgentSession) -> None:
        super().__init__()
        self.session = session

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        console = self.params.get("console")
        output_callback = self.params.get("output_callback")
        if output_callback is None:
            output_callback = create_rich_output(console)
        return {
            "input_iter": self.params.get("input_iter"),
            "output_callback": output_callback,
            "console": console,
        }

    def exec(self, prep_res: Dict[str, Any]) -> int:
        custom_iter = prep_res.get("input_iter")
        interactive = custom_iter is None
        console: Optional[Console] = prep_res.get("console")
        iterator: Iterator[str] = (
            custom_iter if custom_iter is not None else _stdin_iterator(console)
        )
        output = prep_res["output_callback"]
        if interactive:
            output("[system] Entering Code Agent. Type 'exit' to quit. Press ESC to cancel the current request.")
        else:
            output("[system] Entering Code Agent. Type 'exit' to quit.")
        for raw in iterator:
            message = raw.strip()
            if not message:
                continue
            if message.lower() in {"exit", "quit"}:
                break
            result = (
                self._run_with_cancellation(message, output)
                if interactive
                else self.session.run_turn(message, output_callback=output)
            )
            if result.get("cancelled"):
                output("[system] 当前请求已取消。")
                continue
            _emit_result(result, output)
        return 0

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: int) -> int:
        shared["exit_code"] = exec_res
        return "complete"

    def _run_with_cancellation(self, message: str, output: Callable[[str], None]) -> Dict[str, Any]:
        cancel_event = threading.Event()
        done_event = threading.Event()
        result_box: Dict[str, Any] = {}
        error_box: Dict[str, BaseException] = {}

        def worker() -> None:
            try:
                result_box.update(
                    self.session.run_turn(
                        message,
                        output_callback=output,
                        cancel_event=cancel_event,
                    )
                )
            except BaseException as exc:  # pragma: no cover - defensive
                error_box["error"] = exc
            finally:
                done_event.set()

        runner = threading.Thread(target=worker, daemon=True)
        runner.start()
        try:
            self._monitor_escape(cancel_event, done_event, output)
        finally:
            done_event.wait()
            runner.join()
        if error_box:
            raise error_box["error"]
        if cancel_event.is_set() and not result_box.get("cancelled"):
            result_box["cancelled"] = True
        return result_box

    def _monitor_escape(
        self,
        cancel_event: threading.Event,
        done_event: threading.Event,
        output: Callable[[str], None],
    ) -> None:
        if cancel_event.is_set():
            return
        if done_event.wait(timeout=0):
            return
        if not sys.stdin.isatty():
            done_event.wait()
            return
        if os.name == "nt":  # pragma: no cover - Windows-only branch
            self._monitor_escape_windows(cancel_event, done_event, output)
        else:
            self._monitor_escape_posix(cancel_event, done_event, output)

    def _monitor_escape_posix(
        self,
        cancel_event: threading.Event,
        done_event: threading.Event,
        output: Callable[[str], None],
    ) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not done_event.is_set():
                read_list, _, _ = select.select([fd], [], [], 0.05)
                if not read_list:
                    continue
                char = os.read(fd, 1)
                if not char:
                    continue
                if char == b"\x1b":
                    cancel_event.set()
                    output("[system] 捕获 ESC，正在取消本次请求…")
                    break
            done_event.wait()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    def _monitor_escape_windows(
        self,
        cancel_event: threading.Event,
        done_event: threading.Event,
        output: Callable[[str], None],
    ) -> None:
        import msvcrt

        while not done_event.is_set():
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char == b"\x1b":
                    cancel_event.set()
                    output("[system] 捕获 ESC，正在取消本次请求…")
                    break
            if done_event.wait(0.05):
                break
        done_event.wait()


class CodeAgentCLIFlow(Flow):
    def __init__(self, session: Optional[CodeAgentSession] = None) -> None:
        super().__init__()
        self.session = session or CodeAgentSession()
        self.loop_node = _RunLoopNode(self.session)
        self.start(self.loop_node)

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> int:
        return shared.get("exit_code", 0)


def run_code_agent_cli(
    *,
    session: Optional[CodeAgentSession] = None,
    input_iter: Optional[Iterable[str]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    console: Optional[Console] = None,
) -> int:
    active_console = console or Console()
    emitter = output_callback or create_rich_output(active_console)
    flow = CodeAgentCLIFlow(session=session)
    flow.set_params(
        {
            "input_iter": input_iter,
            "output_callback": emitter,
            "console": active_console,
        }
    )
    return flow._run({})


def run_code_agent_once(
    prompt: str,
    *,
    session: Optional[CodeAgentSession] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Execute a single Code Agent turn and emit the summarised result."""

    active_session = session or CodeAgentSession()
    if output_callback is None:
        active_console = console or Console()
        emitter = create_rich_output(active_console)
    else:
        emitter = output_callback
    result = active_session.run_turn(prompt, output_callback=emitter)
    if not result.get("cancelled"):
        _emit_result(result, emitter)
    return result


def _emit_result(result: Mapping[str, Any], output_callback: Callable[[str], None]) -> None:
    plan = result.get("tool_plan") or {}
    thoughts = plan.get("thoughts")
    if thoughts:
        output_callback(f"[planner] {thoughts}")
    for call in plan.get("tool_calls") or []:
        key = call.get("key") or call.get("name") or "tool"
        args_preview = _preview_payload(call.get("arguments") or {}, 180)
        message = key
        if args_preview and args_preview not in {"{}", "null"}:
            message += f" | args: {args_preview}"
        output_callback(f"[plan] {message}")
    for tool_result in result.get("tool_results") or []:
        key = tool_result.get("label") or tool_result.get("key")
        arguments_preview = _preview_payload(tool_result.get("arguments") or {}, 180)
        error_payload = tool_result.get("error")
        if error_payload:
            error_preview = _preview_payload(error_payload, 160)
            if error_preview in {"{}", "null"}:
                error_preview = ""
            message = f"{key} | status: error"
            if arguments_preview and arguments_preview not in {"{}", "null"}:
                message += f" | args: {arguments_preview}"
            if error_preview:
                message += f" | error: {error_preview}"
            output_callback(f"[tool] {message}")
        else:
            preview = _preview_payload(tool_result.get("result"), 160)
            if preview in {"{}", "null"}:
                preview = ""
            message = f"{key} | status: success"
            if arguments_preview and arguments_preview not in {"{}", "null"}:
                message += f" | args: {arguments_preview}"
            if preview:
                message += f" | result: {preview}"
            output_callback(f"[tool] {message}")
    final = result.get("final_response")
    if final:
        output_callback(f"[assistant] {final}")


def _stdin_iterator(console: Optional[Console] = None) -> Iterable[str]:
    prompt = "You: "
    if console is not None:
        reader: Callable[[], str] = lambda: console.input(prompt)
    else:
        reader = lambda: input(prompt)
    while True:
        try:
            yield reader()
        except EOFError:
            break


def _create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code Agent module CLI")
    parser.add_argument(
        "-w",
        "--workspace",
        default=".",
        help="Workspace path to operate within during the session.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="+",
        help="Prompt to execute once before exiting the CLI.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for running the code agent directly."""

    parser = _create_cli_parser()
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    prompt_text = " ".join(args.prompt).strip() if args.prompt else ""
    console = Console()
    emitter = create_rich_output(console)

    original_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        session = CodeAgentSession(max_iterations=100)
        if prompt_text:
            run_code_agent_once(
                prompt_text,
                session=session,
                output_callback=emitter,
                console=console,
            )
            return 0
        return run_code_agent_cli(
            session=session,
            output_callback=emitter,
            console=console,
        )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
