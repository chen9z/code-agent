from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tools.registry import ToolRegistry, ToolSpec
from ui.emission import OutputCallback, OutputMessage, create_emit_event


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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.id,
                    "name": tool_name,
                    "content": result.content,
                }
            )

            payload: Dict[str, Any] = {
                "tool": tool_name,
                "tool_call_id": result.id,
                "arguments": result.arguments,
                "status": result.status,
                "content": result.content,
                "data": result.data,
                "display": result.data["display"]
            }

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

        return ToolOutput(
            name=call.name,
            arguments=effective_arguments,
            call_id=call.call_id,
            status=output.get("status"),
            content=output.get("content"),
            data=output.get("data"),
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


def _emit(output_callback: Optional[OutputCallback], message: OutputMessage) -> None:
    if callable(output_callback):
        output_callback(message)


_TIMEOUT_AWARE_TOOLS = {"bash"}
