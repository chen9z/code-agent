from __future__ import annotations

from typing import Any, Dict

from rich.console import Console

from ui.emission import create_emit_event
from cli import _handle_cli_command
from ui.rich_output import create_rich_output


class _DummySession:
    def run_turn(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - not used
        raise AssertionError("run_turn should not be called in this test")


def test_rich_output_renders_tool_output_tag():
    console = Console(record=True)
    emit = create_rich_output(console)

    emit("[tool-output] preview text")

    rendered = console.export_text()
    assert "preview text" in rendered


def test_rich_output_handles_structured_events():
    console = Console(record=True)
    emit = create_rich_output(console)

    event = create_emit_event(
        "tool",
        "Echo",
        payload={
            "hello": "world",
            "display": [("status", "success"), ("args", "value: 1")],
        },
    )
    emit(event)

    rendered = console.export_text()
    assert "Echo" in rendered
    assert "args" in rendered



def test_cli_help_command_emits_message():
    session = _DummySession()
    outputs: list[str] = []

    handled = _handle_cli_command(":help", session, outputs.append)

    assert handled is True
    assert outputs
    assert "help" in outputs[-1].lower()
