from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ui.emission import OutputCallback, OutputMessage, create_emit_event

from tools.registry import ToolRegistry, ToolSpec


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class ToolOutput:
    """Standardized record for tool execution outcomes."""

    name: str
    arguments: Dict[str, Any]
    call_id: str
    label: str
    status: str = "unknown"
    content: str = ""
    data: Any = None

    @property
    def id(self) -> str:
        return self.call_id

    @property
    def result_text(self) -> str:
        return self.content or ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "inputs": {
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
                "label": self.label,
            },
            "result": {
                "status": self.status,
                "content": self.content,
                "data": self.data,
            },
        }

class ToolExecutionRunner:
    """Executes tool calls and streams console-friendly output."""

    def __init__(
        self,
        registry: ToolRegistry,
        max_parallel_workers: int = 4,
        *,
        default_timeout_seconds: Optional[float] = None,
    ) -> None:
        if max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be at least 1")
        self.registry = registry
        self.max_parallel_workers = max_parallel_workers
        self.default_timeout_seconds = (
            float(default_timeout_seconds)
            if default_timeout_seconds is not None and default_timeout_seconds > 0
            else None
        )

    def set_default_timeout(self, seconds: Optional[float]) -> None:
        if seconds is None or seconds <= 0:
            self.default_timeout_seconds = None
        else:
            self.default_timeout_seconds = float(seconds)

    def run(
        self,
        tool_calls: Iterable[Mapping[str, Any]],
        *,
        messages: List[Dict[str, Any]],
        output_callback: Optional[OutputCallback] = None,
        timeout_override: Optional[float] = None,
    ) -> List[ToolOutput]:
        if timeout_override is not None and timeout_override > 0:
            self.default_timeout_seconds = float(timeout_override)

        calls = [self._to_tool_call(tool, idx) for idx, tool in enumerate(tool_calls)]
        if not calls:
            return []

        results = self._run_parallel(calls)
        self._record_results(results, messages, output_callback)
        return results

    def _record_results(
        self,
        results: List[ToolOutput],
        history: List[Dict[str, Any]],
        output_callback: Optional[OutputCallback],
    ) -> None:
        for result in results:
            tool_name = result.name
            label = result.label or (tool_name.upper() if isinstance(tool_name, str) else str(tool_name))
            normalized_tool = str(tool_name).lower() if isinstance(tool_name, str) else ""
            is_bash_tool = normalized_tool == "bash"
            is_glob_tool = normalized_tool == "glob"
            has_error = result.status != "success"

            console_preview = ""
            truncated_output = False
            full_output_text = ""

            if not has_error and result.data is not None:
                full_output_text = _stringify_tool_output(result.data)
                console_preview, truncated_output = _build_console_preview(full_output_text)
                if (is_bash_tool or is_glob_tool) and console_preview:
                    history_content = _build_history_preview(console_preview)
                else:
                    history_content = ""
            elif not has_error and result.content:
                full_output_text = result.content
                console_preview, truncated_output = _build_console_preview(full_output_text)
                if (is_bash_tool or is_glob_tool) and console_preview:
                    history_content = _build_history_preview(console_preview)
                else:
                    history_content = ""
            elif has_error:
                error_text = str(result.content or "")
                history_content = _truncate_text(
                    error_text,
                    max_chars=MAX_HISTORY_PREVIEW_CHARS,
                    max_lines=MAX_HISTORY_PREVIEW_LINES,
                )[0]
            else:
                history_content = ""

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": result.id,
                    "name": tool_name,
                    "content": history_content,
                }
            )

            if not has_error:
                display = [("status", result.status)]
                display.extend(
                    _build_tool_result_display_entries(
                        normalized_tool,
                        result.data,
                        result.content,
                        console_preview,
                        full_output_text,
                    )
                )
                if truncated_output:
                    display.append(("note", "preview truncated"))
                payload = {
                    "tool": tool_name,
                    "tool_call_id": result.id,
                    "arguments": result.arguments,
                    "status": result.status,
                    "content": result.content,
                    "data": result.data,
                    "truncated_output": truncated_output,
                }
                if display:
                    payload["display"] = display
                _emit(
                    output_callback,
                    create_emit_event(
                        "tool",
                        label,
                        payload=payload,
                    ),
                )
            else:
                error_preview_text, trunc_err = _truncate_text(
                    str(result.content or ""),
                    max_chars=MAX_ERROR_PREVIEW_CHARS,
                    max_lines=MAX_ERROR_PREVIEW_LINES,
                )
                error_inline = error_preview_text.replace("\n", " ")
                display = [("status", "error")]
                if error_inline:
                    display.append(("error", error_inline))
                if error_preview_text:
                    display.append(("output", error_preview_text))
                if trunc_err:
                    display.append(("note", "preview truncated"))
                payload = {
                    "tool": tool_name,
                    "tool_call_id": result.id,
                    "arguments": result.arguments,
                    "status": result.status,
                    "content": result.content,
                    "data": result.data,
                }
                if display:
                    payload["display"] = display
                _emit(
                    output_callback,
                    create_emit_event(
                        "tool",
                        label,
                        payload=payload,
                    ),
                )

    def _to_tool_call(self, raw: Dict[str, Any], index: int) -> ToolCall:
        if not isinstance(raw, dict):
            raise TypeError("Tool call specification must be a dictionary")
        name = raw.get("name") or raw.get("key") or raw.get("tool")
        if not name:
            raise ValueError("Tool call is missing 'name'")
        arguments = raw.get("arguments") or raw.get("args") or {}
        if not isinstance(arguments, dict):
            raise TypeError("Tool call arguments must be a dictionary")
        call_id = raw.get("id") or f"call_{index}"
        return ToolCall(name=str(name), arguments=arguments, call_id=str(call_id))

    def _run_parallel(self, calls: List[ToolCall]) -> List[ToolOutput]:
        # Execute concurrently but preserve original call order in the returned list.
        ordered: List[Optional[ToolOutput]] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(self.max_parallel_workers, len(calls))) as executor:
            future_map = {executor.submit(self._execute_call, call): idx for idx, call in enumerate(calls)}
            for future in as_completed(future_map):
                idx = future_map[future]
                ordered[idx] = future.result()
        return [res for res in ordered if res is not None]

    def _execute_call(self, call: ToolCall) -> ToolOutput:
        try:
            spec: ToolSpec = self.registry.get(call.name)
        except Exception as exc:  # pragma: no cover - exercised via tests
            return ToolOutput(
                name=call.name,
                arguments=call.arguments,
                call_id=call.call_id,
                label=str(call.name),
                status="error",
                content=str(exc),
                data=None,
            )

        effective_arguments = self._apply_timeout_default(call.name, call.arguments)

        try:
            output = spec.tool.execute(**effective_arguments)
        except Exception as exc:  # pragma: no cover - exercised via tests
            return ToolOutput(
                name=call.name,
                arguments=effective_arguments,
                call_id=call.call_id,
                label=spec.name,
                status="error",
                content=str(exc),
                data=None,
            )

        return ToolOutput(
            name=call.name,
            arguments=effective_arguments,
            call_id=call.call_id,
            label=spec.name,
            status="success",
            content=_stringify_payload(output),
            data=output,
        )

    def _apply_timeout_default(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        normalized = str(name).lower() if isinstance(name, str) else ""
        if normalized not in _TIMEOUT_AWARE_TOOLS or self.default_timeout_seconds is None:
            return dict(arguments)
        if "timeout" in arguments and arguments["timeout"]:
            return dict(arguments)
        timeout_ms = max(int(self.default_timeout_seconds * 1000), 1)
        updated = dict(arguments)
        updated["timeout"] = timeout_ms
        return updated


def _build_tool_result_display_entries(
    tool_key: str,
    data: Any,
    result_content: Optional[str],
    console_preview: str,
    full_output_text: str,
) -> List[tuple[str, Optional[str]]]:
    entries: List[tuple[str, Optional[str]]] = []
    normalized = (tool_key or "").lower()

    if normalized == "read":
        summary = _format_read_summary(data)
        if summary:
            entries.append(("result", summary))
            return entries

    if normalized == "grep":
        entries.extend(_format_grep_matches(data))
        if entries:
            return entries

    if normalized == "todo_write":
        entries.extend(_format_todo_markdown(data))
        if entries:
            return entries

    fallback = console_preview or (result_content if result_content else full_output_text)
    fallback = fallback.strip()
    if fallback:
        entries.append(("result", fallback))
    return entries


def _format_read_summary(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    count = data.get("count")
    if isinstance(count, (int, float)):
        return f"Read {int(count)} lines"
    return "Read file"


def _format_grep_matches(data: Any) -> List[tuple[str, Optional[str]]]:
    entries: List[tuple[str, Optional[str]]] = []
    if not isinstance(data, dict):
        return entries
    matches = data.get("matches")
    if not isinstance(matches, list) or not matches:
        entries.append(("result", "No matches"))
        return entries

    display_limit = MAX_GREP_DISPLAY_MATCHES
    for match in matches[:display_limit]:
        if not isinstance(match, dict):
            continue
        path = match.get("path") or "[unknown path]"
        line_no = match.get("line")
        location = f"{path}:{line_no}" if isinstance(line_no, int) else str(path)
        snippet = (match.get("line_text") or "").strip()
        entry_value = f"{location} {snippet}".strip()
        entries.append(("match", entry_value))

    total = data.get("count")
    if isinstance(total, int) and total > display_limit:
        entries.append(("note", f"+{total - display_limit} more matches"))

    return entries


def _format_todo_markdown(data: Any) -> List[tuple[str, Optional[str]]]:
    entries: List[tuple[str, Optional[str]]] = []
    if not isinstance(data, dict):
        return entries
    todos = data.get("todos")
    if not isinstance(todos, list) or not todos:
        return entries

    lines: List[str] = []
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        status = str(todo.get("status") or "").lower()
        if status == "completed":
            marker = "[x]"
        elif status == "in_progress":
            marker = "[-]"
        else:
            marker = "[ ]"
        content = str(todo.get("content") or "").strip()
        active_form = str(todo.get("activeForm") or "").strip()
        suffix = f" ({active_form})" if active_form else ""
        lines.append(f"- {marker} {content}{suffix}".rstrip())

    if lines:
        entries.append(("todo", "\n".join(lines)))
    return entries


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


def _emit(output_callback: Optional[OutputCallback], message: OutputMessage) -> None:
    if callable(output_callback):
        output_callback(message)

def _truncate_text(text: str, *, max_chars: int, max_lines: int) -> tuple[str, bool]:
    if not text:
        return "", False
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        truncated = True
        lines = lines[:max_lines]
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        truncated = True
        joined = joined[: max_chars - 3] + "..."
    return joined, truncated


def _stringify_tool_output(data: Any) -> str:
    if isinstance(data, dict):
        stdout = data.get("stdout")
        stderr = data.get("stderr")
        if isinstance(stdout, str) or isinstance(stderr, str):
            segments: List[str] = []
            if isinstance(stdout, str) and stdout:
                segments.append(stdout.rstrip("\n"))
            if isinstance(stderr, str) and stderr:
                header = "STDERR:"
                segments.append(f"{header}\n{stderr.rstrip('\n')}")
            if segments:
                return "\n\n".join(seg for seg in segments if seg)
        return json.dumps(data, ensure_ascii=False, indent=2)
    if isinstance(data, list):
        preview = ", ".join(str(item) for item in data[:20])
        if len(data) > 20:
            preview += f", ... (+{len(data) - 20} more)"
        return preview
    return _stringify_payload(data)


def _build_console_preview(full_output: str) -> tuple[str, bool]:
    preview, truncated = _truncate_text(
        full_output,
        max_chars=MAX_TOOL_PREVIEW_CHARS,
        max_lines=MAX_TOOL_PREVIEW_LINES,
    )
    if not preview and not truncated:
        return "", False
    if truncated:
        if preview:
            preview = f"{preview}\n... (output truncated)"
        else:
            preview = "... (output truncated)"
    return preview, truncated


def _build_history_preview(preview_text: str) -> str:
    trimmed, _ = _truncate_text(
        preview_text,
        max_chars=MAX_HISTORY_PREVIEW_CHARS,
        max_lines=MAX_HISTORY_PREVIEW_LINES,
    )
    return trimmed
MAX_TOOL_PREVIEW_CHARS = 4000
MAX_TOOL_PREVIEW_LINES = 80
MAX_HISTORY_PREVIEW_CHARS = 800
MAX_HISTORY_PREVIEW_LINES = 8
MAX_ERROR_PREVIEW_CHARS = 800
MAX_ERROR_PREVIEW_LINES = 12
_TIMEOUT_AWARE_TOOLS = {"bash"}
MAX_GREP_DISPLAY_MATCHES = 5
