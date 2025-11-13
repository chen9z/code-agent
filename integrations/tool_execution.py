from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
        messages: List[Dict[str, Any]],
        output_callback: Optional[OutputCallback],
    ) -> None:
        for result in results:
            tool_name = str(result.name).lower()
            is_bash_tool = tool_name == "bash"
            is_glob_tool = tool_name == "glob"
            has_error = result.status != "success"

            console_preview = ""
            full_output_text = ""

            if not has_error and result.data is not None:
                full_output_text = _stringify_tool_output(result.data)
                console_preview = full_output_text
                if (is_bash_tool or is_glob_tool) and console_preview:
                    result_content = console_preview
                else:
                    result_content = ""
            elif not has_error and result.content:
                full_output_text = result.content
                console_preview = full_output_text
                if (is_bash_tool or is_glob_tool) and console_preview:
                    result_content = console_preview
                else:
                    result_content = ""
            elif has_error:
                error_text = str(result.content or "")
                result_content = error_text
            else:
                result_content = ""

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.id,
                    "name": tool_name,
                    "content": result_content,
                }
            )

            if not has_error:
                display = [("status", result.status)]
                display.extend(
                    _extract_display_entries(
                        result.data,
                        result.content,
                        console_preview,
                        full_output_text,
                    )
                )
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
                            tool_name,
                            payload=payload,
                        ),
                    )
            else:
                tool_display: List[tuple[str, Optional[str]]] = []
                if isinstance(result.data, Mapping):
                    tool_display.extend(_normalize_display_items(result.data.get("display")))
                    nested_data = result.data.get("data") if isinstance(result.data, Mapping) else None
                    if not tool_display and isinstance(nested_data, Mapping):
                        tool_display.extend(_normalize_display_items(nested_data.get("display")))

                if tool_display:
                    display = tool_display
                else:
                    error_preview_text = str(result.content or "")
                    error_inline = error_preview_text.replace("\n", " ")
                    display = []
                    if error_inline:
                        display.append(("error", error_inline))
                    if error_preview_text:
                        display.append(("output", error_preview_text))
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
                            tool_name,
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
                status="error",
                content=str(exc),
                data=None,
            )

        tool_status = str(output.get("status", "success")).lower()
        if tool_status not in {"success", "error"}:
            tool_status = "success"

        content_value = output.get("content")
        if content_value is None:
            content_value = ""

        return ToolOutput(
            name=call.name,
            arguments=effective_arguments,
            call_id=call.call_id,
            status=tool_status,
            content=str(content_value),
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


def _stringify_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _emit(output_callback: Optional[OutputCallback], message: OutputMessage) -> None:
    if callable(output_callback):
        output_callback(message)

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
    return _stringify_payload(data)


def _extract_display_entries(
    data: Any,
    result_content: Optional[str],
    console_preview: str,
    full_output_text: str,
) -> List[tuple[str, Optional[str]]]:
    entries: List[tuple[str, Optional[str]]] = []
    if isinstance(data, Mapping):
        raw_display = data.get("display")
        if raw_display is None:
            nested = data.get("data")
            if isinstance(nested, Mapping):
                raw_display = nested.get("display")
        entries.extend(_normalize_display_items(raw_display))
    if entries:
        return entries

    fallback = console_preview or (result_content if result_content else full_output_text)
    fallback = fallback.strip()
    if fallback:
        entries.append(("result", fallback))
    return entries


def _normalize_display_items(source: Any) -> List[tuple[str, Optional[str]]]:
    if source is None:
        return []

    def _append(target: List[tuple[str, Optional[str]]], key: Any, value: Any) -> None:
        name = str(key or "").strip()
        if not name:
            return
        if value is None:
            target.append((name, None))
            return
        text = str(value).strip()
        if text:
            target.append((name, text))
        else:
            target.append((name, None))

    normalized: List[tuple[str, Optional[str]]] = []
    candidates: Sequence[Any]
    if isinstance(source, Mapping):
        candidates = list(source.items())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        candidates = source
    else:
        candidates = (source,)

    for entry in candidates:
        if isinstance(entry, Mapping):
            for key, value in entry.items():
                _append(normalized, key, value)
            continue
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            if not entry:
                continue
            key = entry[0]
            value = entry[1] if len(entry) > 1 else None
            _append(normalized, key, value)
            continue
        _append(normalized, entry, None)

    return normalized
_TIMEOUT_AWARE_TOOLS = {"bash"}
