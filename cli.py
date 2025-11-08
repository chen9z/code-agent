#!/usr/bin/env python3
"""Code Agent CLI utilities and executable entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

from rich.console import Console

from ui.rich_output import create_rich_output, stringify_payload


class AgentSessionProtocol(Protocol):
    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        ...


def _handle_cli_command(
    command: str,
    session: AgentSessionProtocol,
    output_callback: Callable[[str], None],
) -> bool:
    normalized = command.strip()
    if normalized in {":help", ":?"}:
        _emit_help(output_callback)
        return True
    return False


def _emit_help(output_callback: Callable[[str], None]) -> None:
    output_callback("[system] Commands: :help to show this message; type exit to quit.")


def run_code_agent_cli(
    *,
    session: Optional[AgentSessionProtocol] = None,
    session_factory: Optional[Callable[[], AgentSessionProtocol]] = None,
    input_iter: Optional[Iterable[str]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    console: Optional[Console] = None,
    emit_result: Optional[Callable[[Mapping[str, Any], Callable[[str], None]], None]] = None,
) -> int:
    active_session = _resolve_session(session, session_factory)
    active_console = console or Console()
    emitter = output_callback or create_rich_output(active_console)
    iterator: Iterator[str]
    if input_iter is not None:
        iterator = iter(input_iter)
    else:
        iterator = _stdin_iterator(active_console)

    emit_hook = emit_result or _emit_result
    emitter("[system] Entering Code Agent. Type exit to quit.")
    for raw in iterator:
        message = raw.strip()
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        if message.startswith(":") and _handle_cli_command(message, active_session, emitter):
            continue
        result = active_session.run_turn(message, output_callback=emitter)
        emit_hook(result, emitter)
    return 0


def run_code_agent_once(
    prompt: str,
    *,
    session: Optional[AgentSessionProtocol] = None,
    session_factory: Optional[Callable[[], AgentSessionProtocol]] = None,
    output_callback: Callable[[str], None],
    emit_result: Optional[Callable[[Mapping[str, Any], Callable[[str], None]], None]] = None,
) -> Dict[str, Any]:
    """Execute a single Code Agent turn and emit the result."""

    active_session = _resolve_session(session, session_factory)
    result = active_session.run_turn(prompt, output_callback=output_callback)
    (emit_result or _emit_result)(result, output_callback)
    return result


def run_cli_main(
    argv: Optional[Sequence[str]] = None,
    *,
    session_factory: Callable[[], AgentSessionProtocol],
    emit_result: Optional[Callable[[Mapping[str, Any], Callable[[str], None]], None]] = None,
) -> int:
    parser = _create_cli_parser()
    args = parser.parse_args(argv)
    tool_timeout = args.tool_timeout

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    prompt_text = " ".join(args.prompt).strip() if args.prompt else ""
    console = Console()
    emitter = create_rich_output(console)

    original_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        session = session_factory()
        if tool_timeout is not None:
            setter = getattr(session, "set_tool_timeout_seconds", None)
            if callable(setter):
                setter(tool_timeout)
        if prompt_text:
            run_code_agent_once(
                prompt_text,
                session=session,
                output_callback=emitter,
                emit_result=emit_result,
            )
            return 0
        return run_code_agent_cli(
            session=session,
            output_callback=emitter,
            console=console,
            emit_result=emit_result,
        )
    finally:
        os.chdir(original_cwd)


def _resolve_session(
    session: Optional[AgentSessionProtocol],
    session_factory: Optional[Callable[[], AgentSessionProtocol]],
) -> AgentSessionProtocol:
    if session is not None:
        return session
    if session_factory is None:
        raise ValueError("session or session_factory must be provided")
    return session_factory()


def _emit_result(result: Mapping[str, Any], output_callback: Callable[[str], None]) -> None:
    final = result.get("final_response")
    if not final:
        return

    history = result.get("history")
    already_emitted = False
    if isinstance(history, list):
        for message in reversed(history):
            if message.get("role") != "assistant":
                continue
            content = stringify_payload(message.get("content", ""))
            if content == stringify_payload(final):
                already_emitted = True
            break

    if not already_emitted:
        output_callback(f"[assistant] {stringify_payload(final)}")


def _stdin_iterator(console: Optional[Console] = None) -> Iterator[str]:
    prompt = "You: "
    if console is not None:
        reader: Callable[[], str] = lambda: console.input(prompt)
    else:
        reader = lambda: input(prompt)
    while True:
        try:
            yield reader()
        except EOFError:
            break


def _create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code Agent module CLI")
    parser.add_argument(
        "-w",
        "--workspace",
        default=".",
        help="Workspace path to operate within during the session.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="+",
        help="Prompt to execute once before exiting the CLI.",
    )
    parser.add_argument(
        "--tool-timeout",
        type=float,
        dest="tool_timeout",
        help="Override the default tool timeout in seconds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI entrypoint so `python cli.py` just works."""

    parser = argparse.ArgumentParser(description="Code Agent command-line interface")
    parser.add_argument(
        "-w",
        "--workspace",
        default=".",
        help="Workspace path to operate in during the agent session.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="+",
        help="Prompt to run in a one-off non-interactive session.",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    prompt = " ".join(args.prompt).strip() if args.prompt else ""

    from code_agent import CodeAgentSession  # Local import to avoid circular dependency.

    original_cwd = Path.cwd()
    console = Console()
    emitter = create_rich_output(console)
    try:
        os.chdir(workspace)
        session = CodeAgentSession(max_iterations=100, workspace=workspace)
        if prompt:
            run_code_agent_once(
                prompt,
                session=session,
                output_callback=emitter,
            )
            return 0
        return run_code_agent_cli(
            session=session,
            output_callback=emitter,
            console=console,
        )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentSessionProtocol",
    "create_rich_output",
    "run_cli_main",
    "run_code_agent_cli",
    "run_code_agent_once",
    "stringify_payload",
]
