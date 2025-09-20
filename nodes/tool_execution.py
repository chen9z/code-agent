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
        batches = self._build_batches(tool_calls)
        results: List[Dict[str, Any]] = []
        for mode, calls in batches:
            if not calls:
                continue
            if mode == "parallel":
                results.extend(self._run_parallel(calls))
            else:
                results.extend(self._run_sequential(calls))
        return results

    def post(self, shared: Dict[str, Any], prep_res: List[Dict[str, Any]], exec_res: List[Dict[str, Any]]) -> str:
        existing = shared.setdefault("tool_results", [])
        existing.extend(exec_res)

        history = shared.setdefault("history", [])
        for result in exec_res:
            message = {
                "role": "tool",
                "tool_call_id": result.get("id"),
                "name": result.get("key"),
                "content": json.dumps(result.get("output") or result.get("error"), ensure_ascii=False),
            }
            history.append(message)
        return "summarize"

    def _build_batches(self, raw_calls: Iterable[Dict[str, Any]]) -> List[Tuple[str, List[ToolCall]]]:
        batches: List[Tuple[str, List[ToolCall]]] = []
        current_parallel: List[ToolCall] = []
        for index, raw in enumerate(raw_calls):
            call = self._to_tool_call(raw, index)
            normalized_mode = call.mode.lower()
            if normalized_mode == "parallel":
                current_parallel.append(call)
                continue
            if current_parallel:
                batches.append(("parallel", current_parallel))
                current_parallel = []
            batches.append(("sequential", [call]))
        if current_parallel:
            batches.append(("parallel", current_parallel))
        return batches

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
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_parallel_workers, len(calls))) as executor:
            future_map = {executor.submit(self._execute_call, call): call for call in calls}
            for future in as_completed(future_map):
                results.append(future.result())
        return results

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
