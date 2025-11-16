"""Code Agent root exports and CLI helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from cli import (
    run_cli_main as _run_cli_main,
    run_code_agent_cli as _run_code_agent_cli,
)
from ui.emission import OutputCallback, create_emit_event

from agent.session import CodeAgentSession

try:  # pragma: no cover - rich is optional for type checking
    from rich.console import Console
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[misc, assignment]

__all__ = ["CodeAgentSession", "run_code_agent_cli", "main"]


def run_code_agent_cli(
    *,
    session: Optional["CodeAgentSession"] = None,
    session_factory: Optional[Callable[[], "CodeAgentSession"]] = None,
    input_iter: Optional[Iterable[str]] = None,
    output_callback: Optional[OutputCallback] = None,
    console: Optional["Console"] = None,
) -> int:
    """Launch the interactive CLI using a CodeAgentSession factory."""

    factory = session_factory or (lambda: CodeAgentSession())
    return _run_code_agent_cli(
        session=session,
        session_factory=factory,
        input_iter=input_iter,
        output_callback=output_callback,
        console=console,
    )


def _emit_result(result: Mapping[str, Any], output_callback: OutputCallback) -> None:
    final = result.get("content")
    if not final:
        tool_plan = result.get("tool_plan") or {}
        if isinstance(tool_plan, Mapping):
            final = tool_plan.get("content")
    if not final:
        return

    history = result.get("messages") or result.get("history")
    already_emitted = False
    if isinstance(history, list):
        for message in reversed(history):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content == final:
                already_emitted = True
            break

    if not already_emitted:
        output_callback(create_emit_event("assistant", final))


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Shared helper used by cli.py to spin up a session."""

    return _run_cli_main(
        argv,
        session_factory=lambda: CodeAgentSession(max_iterations=100),
        emit_result=_emit_result,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for running the code agent directly."""

    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
