"""Coverage for runtime primitives extension points used by the CLI agent."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from __init__ import Flow, Node
from nodes.tool_execution import ToolExecutionBatchNode, ToolOutput
from tools.base import BaseTool
from tools.registry import ToolRegistry


class _TrackingNode(Node):
    def __init__(self, label: str, next_action: str | None = None, result: str | None = None) -> None:
        super().__init__()
        self.label = label
        self.next_action = next_action
        self.result = result
        self.calls: List[str] = []
        self.params_seen: List[Dict[str, Any]] = []

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(f"{self.label}:prep")
        return {"payload": shared["payload"]}

    def exec(self, prep_res: Dict[str, Any]) -> str | None:
        self.calls.append(f"{self.label}:exec")
        return self.next_action

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: str | None) -> str | None:
        self.params_seen.append(dict(self.params))
        shared.setdefault("trace", []).append((self.label, self.result or exec_res))
        self.calls.append(f"{self.label}:post")
        return self.result or exec_res


def test_flow_branching_and_params_propagation() -> None:
    first = _TrackingNode("first", next_action="branch")
    second = _TrackingNode("second", result="done")
    third = _TrackingNode("third", result="done")

    flow = Flow(start=first)
    flow.set_params({"mode": "cli"})
    first >> second
    first - "branch" >> third

    shared = {"payload": "seed"}
    action = flow._run(shared)

    assert action == "done"
    assert shared["trace"] == [("first", "branch"), ("third", "done")]
    assert first.params_seen == [{"mode": "cli"}]
    assert third.params_seen == [{"mode": "cli"}]
    assert second.params_seen == []
    assert first.calls == ["first:prep", "first:exec", "first:post"]
    assert third.calls == ["third:prep", "third:exec", "third:post"]


class _FallbackNode(Node):
    def __init__(self, attempts: int) -> None:
        super().__init__(max_retries=attempts)
        self.attempts = attempts
        self.invocations: int = 0

    def exec(self, prep_res: Any) -> str:
        self.invocations += 1
        raise RuntimeError("boom")

    def exec_fallback(self, prep_res: Any, exc: Exception) -> str:
        return f"fallback:{self.invocations}:{exc}"


def test_node_exec_fallback_runs_after_retries() -> None:
    node = _FallbackNode(attempts=2)
    result = node._run({})
    assert result.startswith("fallback:2")


class _EchoTool(BaseTool):
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo arguments"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {"echo": kwargs}


@pytest.fixture()
def echo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool(), key="echo")
    return registry


def test_tool_execution_node_supports_parallel_flag(echo_registry: ToolRegistry) -> None:
    node = ToolExecutionBatchNode(echo_registry, max_parallel_workers=2)
    shared = {
        "tool_plan": {
            "tool_calls": [
                {"key": "echo", "arguments": {"value": 1}, "parallel": True},
                {"key": "echo", "arguments": {"value": 2}},
            ]
        }
    }

    action = node._run(shared)

    assert action == "summarize"
    results = shared["tool_results"]
    assert all(isinstance(result, ToolOutput) for result in results)
    assert {result.result.data["echo"]["value"] for result in results if result.result} == {1, 2}
    assert len(shared["history"]) == 2
    assert {entry["tool_call_id"] for entry in shared["history"]} == {"call_0", "call_1"}
