"""Tests for the simplified CodeAgentSession and CLI flow."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import threading

import pytest

from code_agent import (
    CodeAgentSession,
    create_code_agent_flow,
    run_code_agent_cli,
    run_code_agent_once,
)


class _DummyFlow:
    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}
        self.runs = 0

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = params

    def _run(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        self.runs += 1
        history = list(self.params.get("history") or [])
        history.append({"role": "user", "content": self.params["user_input"]})
        history.append({"role": "assistant", "content": "ack"})
        return {"history": history, "final_response": "ack", "tool_results": []}


def test_session_runs_flow_and_tracks_history() -> None:
    session = CodeAgentSession(flow_factory=lambda: _DummyFlow())

    first = session.run_turn("hello")
    second = session.run_turn("again")

    assert first["final_response"] == "ack"
    assert [msg["role"] for msg in session.history[-2:]] == ["user", "assistant"]
    assert any(msg["content"] == "again" for msg in session.history)
    assert second["final_response"] == "ack"


def test_create_code_agent_flow_sets_parallel_workers() -> None:
    flow = create_code_agent_flow(max_parallel_workers=2)
    assert flow.execution_node.max_parallel_workers == 2


class _StubSession(CodeAgentSession):
    def __init__(self, scripted: Iterable[str]) -> None:
        self.outputs: List[str] = list(scripted)
        super().__init__(flow_factory=lambda: _DummyFlow())

    def run_turn(
        self,
        message: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        result = super().run_turn(
            message,
            output_callback=output_callback,
            cancel_event=cancel_event,
        )
        result.setdefault("tool_plan", {"thoughts": "stub", "tool_calls": []})
        result.setdefault("tool_results", [])
        return result


def test_run_code_agent_cli_loops_until_exit(monkeypatch) -> None:
    session = CodeAgentSession(flow_factory=lambda: _DummyFlow())
    inputs = ["first", "second", "exit"]
    outputs: List[str] = []

    exit_code = run_code_agent_cli(session=session, input_iter=inputs, output_callback=outputs.append)

    assert exit_code == 0
    assert session.history[-4]["content"] == "first"
    assert outputs[0].startswith("[system] Entering Code Agent")
    assert outputs[-1].startswith("[assistant]")


def test_run_code_agent_cli_skips_empty_lines() -> None:
    session = CodeAgentSession(flow_factory=lambda: _DummyFlow())
    inputs = ["   ", "exit"]
    outputs: List[str] = []

    exit_code = run_code_agent_cli(session=session, input_iter=inputs, output_callback=outputs.append)

    assert exit_code == 0
    assert len(session.history) == 0


def test_run_turn_respects_pre_set_cancel_event() -> None:
    session = CodeAgentSession(flow_factory=lambda: _DummyFlow())
    token = threading.Event()
    token.set()

    result = session.run_turn("cancel me", cancel_event=token)

    assert result["cancelled"] is True
    assert session.history == []


def test_run_code_agent_once_executes_single_turn() -> None:
    session = CodeAgentSession(flow_factory=lambda: _DummyFlow())
    outputs: List[str] = []

    result = run_code_agent_once("single", session=session, output_callback=outputs.append)

    assert result["final_response"] == "ack"
    assert outputs
    assert outputs[-1].startswith("[assistant]")
