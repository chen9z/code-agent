"""Tests for the ToolExecutionRunner sequential and parallel execution."""

import threading
import time
from typing import Any

import pytest

from integrations.tool_execution import ToolExecutionRunner, ToolOutput
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
        return {
            "content": f"{self._name}: {kwargs}",
            "data": {
                "echo": kwargs,
                "display": [("result", f"{self._name}: {kwargs}")],
            },
        }


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
        return {
            "content": "",
            "data": {
                "display": [("result", "(empty)")],
            },
        }


class _LongOutputTool(BaseTool):
    @property
    def name(self) -> str:
        return "Long"

    @property
    def description(self) -> str:
        return "returns lengthy stdout"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        lines = [f"line {i}" for i in range(200)]
        payload = "\n".join(lines)
        return {
            "content": payload,
            "data": {
                "stdout": payload,
                "command": "generate",
                "display": [("result", payload)],
            },
        }


class _TimeoutAwareTool(BaseTool):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return "expects timeout"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"timeout": {"type": "number"}}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "content": "done",
            "data": {
                "stdout": "done",
                "display": [("result", "done")],
            },
        }


class _DisplayTool(BaseTool):
    @property
    def name(self) -> str:
        return "Display"

    @property
    def description(self) -> str:
        return "emits custom display"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return {
            "status": "success",
            "content": "",
            "data": {
                "display": [("result", "custom display")],
            },
        }


class _ReportedErrorTool(BaseTool):
    @property
    def name(self) -> str:
        return "ReportedError"

    @property
    def description(self) -> str:
        return "reports error status in payload"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return {
            "status": "error",
            "content": "simulated failure",
            "data": {
                "detail": "boom",
                "display": [("error", "simulated failure")],
            },
        }


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_RecorderTool("Echo"), key="echo")
    reg.register(_RecorderTool("Slow", delay=0.05), key="slow")
    reg.register(_EmptyTool(), key="empty")
    return reg


def test_sequential_execution_updates_history(registry: ToolRegistry):
    runner = ToolExecutionRunner(registry, max_parallel_workers=1)
    history: list[dict[str, Any]] = []

    results = runner.run(
        [
            {"name": "echo", "arguments": {"value": 1}},
            {"name": "echo", "arguments": {"value": 2}},
        ],
        messages=history,
    )

    assert len(results) == 2
    first, second = results
    assert isinstance(first, ToolOutput)
    assert isinstance(second, ToolOutput)
    assert first.status == "success"
    assert second.status == "success"
    assert first.data["echo"] == {"value": 1}
    assert second.data["echo"] == {"value": 2}
    assert history[-2]["role"] == "tool"
    assert history[-1]["role"] == "tool"
    assert history[-2]["content"] == "Echo: {'value': 1}"
    assert history[-1]["content"] == "Echo: {'value': 2}"


def test_parallel_execution_runs_all_calls(registry: ToolRegistry):
    runner = ToolExecutionRunner(registry, max_parallel_workers=2)
    history: list[dict[str, Any]] = []

    results = runner.run(
        [
            {"name": "slow", "arguments": {"value": "a"}, "mode": "parallel"},
            {"name": "slow", "arguments": {"value": "b"}, "mode": "parallel"},
        ],
        messages=history,
    )

    assert len(results) == 2
    statuses = {result.data["echo"]["value"] for result in results if result.data}
    assert statuses == {"a", "b"}
    assert all(entry["role"] == "tool" for entry in history[-2:])
    assert {entry["content"] for entry in history[-2:]} == {
        "Slow: {'value': 'a'}",
        "Slow: {'value': 'b'}",
    }


def test_history_preserves_falsy_output(registry: ToolRegistry):
    runner = ToolExecutionRunner(registry)
    history: list[dict[str, Any]] = []

    runner.run(
        [{"name": "empty", "arguments": {}, "mode": "sequential"}],
        messages=history,
    )

    assert history[-1]["content"] == ""


def test_missing_tool_returns_error(registry: ToolRegistry):
    runner = ToolExecutionRunner(registry)
    history: list[dict[str, Any]] = []

    results = runner.run(
        [{"name": "missing", "arguments": {}, "mode": "sequential"}],
        messages=history,
    )

    result = results[0]
    assert isinstance(result, ToolOutput)
    assert result.status == "error"
    assert result.content and "missing" in result.content.lower()


def test_emits_full_tool_output_without_truncation():
    registry = ToolRegistry()
    registry.register(_LongOutputTool(), key="long")
    messages: list[str] = []
    runner = ToolExecutionRunner(registry)
    history: list[dict[str, Any]] = []

    runner.run(
        [{"name": "long", "arguments": {}}],
        messages=history,
        output_callback=messages.append,
    )

    preview_messages = [msg for msg in messages if msg.startswith("[tool]")]

    assert preview_messages, "Expected a tool message"
    assert all("preview truncated" not in msg for msg in preview_messages)
    assert any("line 0" in msg for msg in preview_messages)
    assert any("line 199" in msg for msg in preview_messages)


def test_runner_uses_tool_display_entries():
    registry = ToolRegistry()
    registry.register(_DisplayTool(), key="display")
    runner = ToolExecutionRunner(registry)
    emitted: list[str] = []

    runner.run(
        [{"name": "display", "arguments": {}}],
        messages=[],
        output_callback=emitted.append,
    )

    assert any("custom display" in message for message in emitted), "Expected custom display text"


def test_runner_honors_tool_reported_status():
    registry = ToolRegistry()
    registry.register(_ReportedErrorTool(), key="reported_error")
    runner = ToolExecutionRunner(registry)
    emitted: list[str] = []

    results = runner.run(
        [{"name": "reported_error", "arguments": {}}],
        messages=[],
        output_callback=emitted.append,
    )

    assert results[0].status == "error"
    assert any("simulated failure" in message for message in emitted)


def test_default_timeout_applies_to_bash_when_missing_argument():
    registry = ToolRegistry()
    bash_tool = _TimeoutAwareTool()
    registry.register(bash_tool, key="bash")
    runner = ToolExecutionRunner(registry, default_timeout_seconds=42)
    runner.run(
        [{"name": "bash", "arguments": {"command": "echo ok"}}],
        messages=[],
    )

    assert bash_tool.calls, "expected the tool to be invoked"
    call_kwargs = bash_tool.calls[0]
    assert call_kwargs.get("timeout") == 42000


def test_existing_timeout_is_preserved():
    registry = ToolRegistry()
    bash_tool = _TimeoutAwareTool()
    registry.register(bash_tool, key="bash")
    runner = ToolExecutionRunner(registry, default_timeout_seconds=99)
    runner.run(
        [{"name": "bash", "arguments": {"command": "echo ok", "timeout": 10}}],
        messages=[],
    )

    call_kwargs = bash_tool.calls[0]
    assert call_kwargs.get("timeout") == 10
