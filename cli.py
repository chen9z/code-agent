#!/usr/bin/env python3
"""Code Agent CLI utilities and executable entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

from rich.console import Console

from agent.session import CodeAgentSession
from ui.emission import OutputCallback, create_emit_event
from ui.rich_output import create_rich_output, stringify_payload


class AgentSessionProtocol(Protocol):
    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[OutputCallback] = None,
    ) -> Dict[str, Any]:
        ...


def _handle_cli_command(
    command: str,
    session: AgentSessionProtocol,
    output_callback: OutputCallback,
) -> bool:
    normalized = command.strip()
    if normalized in {":help", ":?"}:
        _emit_help(output_callback)
        return True
    return False


def _emit_help(output_callback: OutputCallback) -> None:
    output_callback(
        create_emit_event(
            "system",
            "Commands: :help to show this message; type exit to quit.",
        )
    )


def run_interactive_session(
    session: AgentSessionProtocol,
    *,
    input_iter: Optional[Iterable[str]] = None,
    output_callback: Optional[OutputCallback] = None,
    console: Optional[Console] = None,
) -> int:
    """Run the interactive loop for the agent session."""
    active_console = console or Console()
    emitter = output_callback or create_rich_output(active_console)
    
    iterator: Iterator[str]
    if input_iter is not None:
        iterator = iter(input_iter)
    else:
        iterator = _stdin_iterator(active_console)

    emitter(create_emit_event("system", "Entering Code Agent. Type exit to quit."))
    
    for raw in iterator:
        message = raw.strip()
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        if message.startswith(":") and _handle_cli_command(message, session, emitter):
            continue
            
        result = session.run_turn(message, output_callback=emitter)
        _emit_result_if_needed(result, emitter)
        
    return 0


def _emit_result_if_needed(result: Mapping[str, Any], output_callback: OutputCallback) -> None:
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
            content = stringify_payload(message.get("content", ""))
            if content == stringify_payload(final):
                already_emitted = True
            break

    if not already_emitted:
        output_callback(
            create_emit_event(
                "assistant",
                stringify_payload(final),
            )
        )


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
    parser.add_argument(
        "--tool-timeout",
        type=float,
        dest="tool_timeout",
        help="Override the default tool timeout in seconds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for the CLI."""
    parser = _create_cli_parser()
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    prompt = " ".join(args.prompt).strip() if args.prompt else ""
    
    original_cwd = Path.cwd()
    console = Console()
    emitter = create_rich_output(console)
    
    try:
        os.chdir(workspace)
        # Create session directly
        session = CodeAgentSession(max_iterations=100, workspace=workspace)
        
        if args.tool_timeout is not None:
            session.set_tool_timeout_seconds(args.tool_timeout)

        if prompt:
            session.run_turn(prompt, output_callback=emitter)
            return 0
            
        return run_interactive_session(
            session=session,
            output_callback=emitter,
            console=console,
        )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
