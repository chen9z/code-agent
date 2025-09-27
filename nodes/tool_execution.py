from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from __init__ import Node
from tools.registry import ToolRegistry, ToolSpec


@dataclass
class ToolCall:
    key: str
    arguments: Dict[str, Any]
    call_id: str
    mode: str = "sequential"


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

    def exec(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls = [self._to_tool_call(raw, idx) for idx, raw in enumerate(tool_calls or [])]
        if not calls:
            return []
        # Always execute in parallel for simplicity and throughput.
        return self._run_parallel(calls)

    def post(self, shared: Dict[str, Any], prep_res: List[Dict[str, Any]], exec_res: List[Dict[str, Any]]) -> str:
        existing = shared.setdefault("tool_results", [])
        existing.extend(exec_res)

        history = shared.setdefault("history", [])
        for result in exec_res:
            payload = result.get("output") if result.get("status") == "success" else result.get("error")
            message = {
                "role": "tool",
                "tool_call_id": result.get("id"),
                "name": result.get("key"),
                # Keep history compact to protect context window; store a safe preview only.
                "content": _preview_payload(payload, 2000),
            }
            history.append(message)
        return "summarize"

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

    def _run_sequential(self, calls: List[ToolCall]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for call in calls:
            results.append(self._execute_call(call))
        return results

    def _run_parallel(self, calls: List[ToolCall]) -> List[Dict[str, Any]]:
        # Execute concurrently but preserve original call order in the returned list.
        ordered: List[Optional[Dict[str, Any]]] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(self.max_parallel_workers, len(calls))) as executor:
            future_map = {executor.submit(self._execute_call, call): idx for idx, call in enumerate(calls)}
            for future in as_completed(future_map):
                idx = future_map[future]
                ordered[idx] = future.result()
        return [res for res in ordered if res is not None]

    def _execute_call(self, call: ToolCall) -> Dict[str, Any]:
        try:
            spec: ToolSpec = self.registry.get(call.key)
            output = spec.tool.execute(**call.arguments)
            return {
                "id": call.call_id,
                "key": call.key,
                "status": "success",
                "output": output,
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            return {
                "id": call.call_id,
                "key": call.key,
                "status": "error",
                "error": str(exc),
            }


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
