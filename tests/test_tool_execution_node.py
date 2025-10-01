"""Tests for ToolExecutionBatchNode sequential and parallel execution."""

import json
import threading
import time

import pytest

from nodes.tool_execution import ToolExecutionBatchNode, ToolOutput
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


class _EmptyTool(BaseTool):
    @property
    def name(self) -> str:
        return "Empty"

    @property
    def description(self) -> str:
        return "returns an empty list"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return []


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_RecorderTool("Echo"), key="echo")
    reg.register(_RecorderTool("Slow", delay=0.05), key="slow")
    reg.register(_EmptyTool(), key="empty")
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
    first, second = shared["tool_results"]
    assert isinstance(first, ToolOutput)
    assert isinstance(second, ToolOutput)
    assert first.status == "success"
    assert second.status == "success"
    assert first.result and first.result.data["echo"] == {"value": 1}
    assert second.result and second.result.data["echo"] == {"value": 2}
    history = shared["history"]
    assert history[-2]["role"] == "tool"
    assert history[-1]["role"] == "tool"
    assert history[-2]["content"] == ""
    assert history[-1]["content"] == ""


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
    statuses = {result.result.data["echo"]["value"] for result in results if result.result}
    assert statuses == {"a", "b"}
    history = shared["history"]
    assert all(entry["role"] == "tool" for entry in history[-2:])
    assert all(entry["content"] == "" for entry in history[-2:])


def test_history_preserves_falsy_output(registry: ToolRegistry):
    node = ToolExecutionBatchNode(registry)
    shared = {
        "tool_plan": {
            "tool_calls": [
                {"key": "empty", "arguments": {}, "mode": "sequential"},
            ]
        }
    }

    node._run(shared)

    history = shared["history"]
    assert history[-1]["content"] == ""


def test_missing_tool_returns_error(registry: ToolRegistry):
    node = ToolExecutionBatchNode(registry)
    shared = {"tool_plan": {"tool_calls": [{"key": "missing", "arguments": {}, "mode": "sequential"}]}}

    action = node._run(shared)

    assert action == "summarize"
    result = shared["tool_results"][0]
    assert isinstance(result, ToolOutput)
    assert result.status == "error"
    assert result.error and "missing" in result.error.lower()
