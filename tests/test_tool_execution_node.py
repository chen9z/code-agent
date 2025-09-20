"""Tests for ToolExecutionBatchNode sequential and parallel execution."""

import threading
import time

import pytest

from nodes.tool_execution import ToolExecutionBatchNode
from tools.base import BaseTool
from tools.registry import ToolRegistry


class _RecorderTool(BaseTool):
    def __init__(self, name: str, delay: float = 0.0) -> None:
        self._name = name
        self.delay = delay
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "records invocations"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.calls.append(kwargs)
        return {"echo": kwargs}


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_RecorderTool("Echo"), key="echo")
    reg.register(_RecorderTool("Slow", delay=0.05), key="slow")
    return reg


def test_sequential_execution_updates_shared(registry: ToolRegistry):
    node = ToolExecutionBatchNode(registry)
    shared = {
        "tool_plan": {
            "tool_calls": [
                {"key": "echo", "arguments": {"value": 1}},
                {"key": "echo", "arguments": {"value": 2}},
            ]
        }
    }

    action = node._run(shared)

    assert action == "summarize"
    assert len(shared["tool_results"]) == 2
    assert shared["tool_results"][0]["status"] == "success"
    assert shared["tool_results"][0]["output"]["echo"] == {"value": 1}
    assert shared["tool_results"][1]["output"]["echo"] == {"value": 2}
    history = shared["history"]
    assert history[-2]["role"] == "tool"
    assert history[-1]["role"] == "tool"


def test_parallel_execution_runs_all_calls(registry: ToolRegistry):
    node = ToolExecutionBatchNode(registry, max_parallel_workers=2)
    shared = {
        "tool_plan": {
            "tool_calls": [
                {"key": "slow", "arguments": {"value": "a"}, "mode": "parallel"},
                {"key": "slow", "arguments": {"value": "b"}, "mode": "parallel"},
            ]
        }
    }

    action = node._run(shared)

    assert action == "summarize"
    results = shared["tool_results"]
    assert len(results) == 2
    statuses = {result["output"]["echo"]["value"] for result in results}
    assert statuses == {"a", "b"}
    history = shared["history"]
    assert all(entry["role"] == "tool" for entry in history[-2:])


def test_missing_tool_returns_error(registry: ToolRegistry):
    node = ToolExecutionBatchNode(registry)
    shared = {"tool_plan": {"tool_calls": [{"key": "missing", "arguments": {}, "mode": "sequential"}]}}

    action = node._run(shared)

    assert action == "summarize"
    result = shared["tool_results"][0]
    assert result["status"] == "error"
    assert "missing" in result["error"].lower()
