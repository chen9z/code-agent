from __future__ import annotations

from typing import Any, Callable, Dict

from rich.console import Console

from cli.code_agent_cli import _RunLoopNode
from cli.rich_output import create_rich_output
from core.tool_output_store import ToolOutputStore


class _DummySession:
    def __init__(self, store: ToolOutputStore) -> None:
        self.store = store

    def run_turn(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - not used
        raise AssertionError("run_turn should not be called in this test")

    def get_tool_output_store(self) -> ToolOutputStore:
        return self.store


def test_rich_output_renders_tool_output_tag():
    console = Console(record=True)
    emit = create_rich_output(console)

    emit("[tool-output] preview text")

    rendered = console.export_text()
    assert "preview text" in rendered


def test_cli_show_command_displays_full_tool_output():
    store = ToolOutputStore()
    store.record(
        call_id="call_0",
        label="Bash",
        status="success",
        arguments={"command": "echo"},
        output="full log contents",
        truncated=True,
    )
    session = _DummySession(store)
    node = _RunLoopNode(session, lambda *_args, **_kwargs: None)
    outputs: list[str] = []

    handled = node._handle_command(":show last", outputs.append)

    assert handled is True
    assert any("full log contents" in line for line in outputs)
