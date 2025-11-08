from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from tools.registry import ToolRegistry, ToolSpec


@dataclass
class ToolCall:
    key: str
    arguments: Dict[str, Any]
    call_id: str
    mode: str = "sequential"


@dataclass
class ToolInvocationInputs:
    """Canonical inputs used when invoking a tool."""

    key: str
    arguments: Dict[str, Any]
    call_id: str
    label: str


@dataclass
class ToolResultPayload:
    """Normalized result payload captured from a tool invocation."""

    status: str
    content: str
    data: Any


@dataclass
class ToolOutput:
    """Standardized record for tool execution outcomes."""

    inputs: ToolInvocationInputs
    result: Optional[ToolResultPayload]
    error: Optional[str]

    @property
    def key(self) -> str:
        return self.inputs.key

    @property
    def id(self) -> str:
        return self.inputs.call_id

    @property
    def status(self) -> str:
        if self.result and self.result.status:
            return self.result.status
        if self.error:
            return "error"
        return "unknown"

    @property
    def arguments(self) -> Dict[str, Any]:
        return self.inputs.arguments

    @property
    def result_text(self) -> str:
        return self.result.content if self.result else ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "inputs": {
                "key": self.inputs.key,
                "arguments": self.inputs.arguments,
                "call_id": self.inputs.call_id,
                "label": self.inputs.label,
            },
            "result": {
                "status": self.result.status,
                "content": self.result.content,
                "data": self.result.data,
            }
            if self.result
            else None,
            "error": self.error,
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
        history: List[Dict[str, Any]],
        output_callback: Optional[Callable[[str], None]] = None,
        timeout_override: Optional[float] = None,
    ) -> List[ToolOutput]:
        if timeout_override is not None and timeout_override > 0:
            self.default_timeout_seconds = float(timeout_override)

        raw_calls = list(tool_calls or [])
        calls = [self._to_tool_call(raw, idx) for idx, raw in enumerate(raw_calls)]
        if not calls:
            return []

        results = self._run_parallel(calls)
        self._record_results(results, history, output_callback)
        return results

    def _record_results(
        self,
        results: List[ToolOutput],
        history: List[Dict[str, Any]],
        output_callback: Optional[Callable[[str], None]],
    ) -> None:
        for result in results:
            key = result.key
            label = result.inputs.label or (key.upper() if isinstance(key, str) else str(key))
            arguments_preview = _preview_payload(result.inputs.arguments or {}, 180)
            if arguments_preview in {"{}", "null"}:
                arguments_preview = ""

            tool_key = str(key).lower() if isinstance(key, str) else ""
            is_bash_tool = tool_key == "bash"
            is_glob_tool = tool_key == "glob"
            has_error = bool(result.error)

            console_preview = ""
            truncated_output = False
            full_output_text = ""

            if not has_error and result.result is not None:
                full_output_text = _stringify_tool_output(result.result.data)
                console_preview, truncated_output = _build_console_preview(full_output_text)
                if (is_bash_tool or is_glob_tool) and console_preview:
                    history_content = _build_history_preview(console_preview)
                else:
                    history_content = ""
            elif has_error:
                error_text = str(result.error or "")
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
                    "name": key,
                    "content": history_content,
                }
            )

            if not has_error:
                message = f"{label} | status: {result.status}"
                if arguments_preview:
                    message += f" | args: {arguments_preview}"
                if truncated_output:
                    message += " | preview truncated"
                _emit(output_callback, f"[tool] {message}")
                if console_preview:
                    _emit(output_callback, f"[tool-output] {console_preview}")
            else:
                error_preview_text, trunc_err = _truncate_text(
                    str(result.error or ""),
                    max_chars=MAX_ERROR_PREVIEW_CHARS,
                    max_lines=MAX_ERROR_PREVIEW_LINES,
                )
                error_inline = error_preview_text.replace("\n", " ")
                message = f"{label} | status: error"
                if arguments_preview:
                    message += f" | args: {arguments_preview}"
                if error_inline:
                    message += f" | error: {error_inline}"
                _emit(output_callback, f"[tool] {message}")
                if trunc_err:
                    _emit(output_callback, f"[tool-output] {error_preview_text}")

    def _to_tool_call(self, raw: Dict[str, Any], index: int) -> ToolCall:
        if not isinstance(raw, dict):
            raise TypeError("Tool call specification must be a dictionary")
        key = raw.get("key") or raw.get("tool")
        if not key:
            raise ValueError("Tool call is missing 'key'")
        arguments = raw.get("arguments") or raw.get("args") or {}
        if not isinstance(arguments, dict):
            raise TypeError("Tool call arguments must be a dictionary")
        call_id = raw.get("id") or f"call_{index}"
        mode = raw.get("mode") or ("parallel" if raw.get("parallel") else "sequential")
        return ToolCall(key=str(key), arguments=arguments, call_id=str(call_id), mode=str(mode))

    def _run_sequential(self, calls: List[ToolCall]) -> List[ToolOutput]:
        results: List[ToolOutput] = []
        for call in calls:
            results.append(self._execute_call(call))
        return results

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
            spec: ToolSpec = self.registry.get(call.key)
        except Exception as exc:  # pragma: no cover - exercised via tests
            inputs = ToolInvocationInputs(
                key=call.key,
                arguments=call.arguments,
                call_id=call.call_id,
                label=str(call.key),
            )
            return ToolOutput(inputs=inputs, result=None, error=str(exc))

        effective_arguments = self._apply_timeout_default(call.key, call.arguments)

        inputs = ToolInvocationInputs(
            key=call.key,
            arguments=effective_arguments,
            call_id=call.call_id,
            label=spec.name,
        )

        try:
            output = spec.tool.execute(**effective_arguments)
            payload = ToolResultPayload(
                status="success",
                content=_stringify_payload(output),
                data=output,
            )
            return ToolOutput(inputs=inputs, result=payload, error=None)
        except Exception as exc:  # pragma: no cover - exercised via tests
            return ToolOutput(inputs=inputs, result=None, error=str(exc))

    def _apply_timeout_default(self, key: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        normalized = str(key).lower() if isinstance(key, str) else ""
        if normalized not in _TIMEOUT_AWARE_TOOLS or self.default_timeout_seconds is None:
            return dict(arguments)
        if "timeout" in arguments and arguments["timeout"]:
            return dict(arguments)
        timeout_ms = max(int(self.default_timeout_seconds * 1000), 1)
        updated = dict(arguments)
        updated["timeout"] = timeout_ms
        return updated


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


def _emit(output_callback: Optional[Callable[[str], None]], message: str) -> None:
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
