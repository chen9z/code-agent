from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from __init__ import Node
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

class ToolExecutionBatchNode(Node):
    """Executes a batch of tool calls sequentially or in parallel based on plan metadata."""

    def __init__(self, registry: ToolRegistry, max_parallel_workers: int = 4) -> None:
        super().__init__()
        if max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be at least 1")
        self.registry = registry
        self.max_parallel_workers = max_parallel_workers

    def prep(self, shared: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan = shared.get("tool_plan", {})
        tool_calls = plan.get("tool_calls", []) if isinstance(plan, dict) else []
        if not isinstance(tool_calls, list):
            raise ValueError("tool_plan.tool_calls must be a list")
        return tool_calls

    def exec(self, tool_calls: List[Dict[str, Any]]) -> List[ToolOutput]:
        calls = [self._to_tool_call(raw, idx) for idx, raw in enumerate(tool_calls or [])]
        if not calls:
            return []
        # Always execute in parallel for simplicity and throughput.
        return self._run_parallel(calls)

    def post(
        self,
        shared: Dict[str, Any],
        prep_res: List[Dict[str, Any]],
        exec_res: List[ToolOutput],
    ) -> str:
        existing = shared.setdefault("tool_results", [])
        existing.extend(exec_res)

        history = shared.setdefault("history", [])
        for result in exec_res:
            key = result.key
            is_bash_tool = isinstance(key, str) and key.lower() == "bash"
            error_payload = result.error
            has_error = bool(error_payload)
            if has_error:
                content_preview = error_payload or ""
            else:
                payload_text = result.result.content if result.result else ""
                content_preview = (
                    _preview_payload(payload_text, 2000) if is_bash_tool else ""
                )
            message = {
                "role": "tool",
                "tool_call_id": result.id,
                "name": key,
                # Keep history compact to protect context window; store a safe preview only.
                "content": content_preview,
            }
            history.append(message)
            arguments_preview = _preview_payload(result.inputs.arguments or {}, 180)
            if arguments_preview in {"{}", "null"}:
                arguments_preview = ""
            label = result.inputs.label or (key.upper() if isinstance(key, str) else str(key))
            if not has_error and result.result:
                snippet = _preview_payload(result.result.data, 200)
                if snippet in {"{}", "null"}:
                    snippet = ""
                message = f"{label} | status: {result.status}"
                if arguments_preview:
                    message += f" | args: {arguments_preview}"
                if snippet:
                    message += f" | result: {snippet}"
                _emit(shared, f"[tool] {message}")
            else:
                error_preview = _preview_payload(error_payload or "", 200)
                if error_preview in {"{}", "null"}:
                    error_preview = ""
                message = f"{label} | status: error"
                if arguments_preview:
                    message += f" | args: {arguments_preview}"
                if error_preview:
                    message += f" | error: {error_preview}"
                _emit(shared, f"[tool] {message}")

        used = shared.get("tool_iterations_used", 0) + 1
        shared["tool_iterations_used"] = used
        max_iterations = getattr(self, "max_iterations", 1)
        if used >= max_iterations:
            return "summarize"
        return "plan"

    # Mode batching removed: we always parallelize within a single planning step.

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

        inputs = ToolInvocationInputs(
            key=call.key,
            arguments=call.arguments,
            call_id=call.call_id,
            label=spec.name,
        )

        try:
            output = spec.tool.execute(**call.arguments)
            payload = ToolResultPayload(
                status="success",
                content=_stringify_payload(output),
                data=output,
            )
            return ToolOutput(inputs=inputs, result=payload, error=None)
        except Exception as exc:  # pragma: no cover - exercised via tests
            return ToolOutput(inputs=inputs, result=None, error=str(exc))


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
