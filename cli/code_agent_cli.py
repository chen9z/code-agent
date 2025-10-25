from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

from rich.console import Console

from __init__ import Flow, Node
from cli.rich_output import create_rich_output, stringify_payload


class AgentSessionProtocol(Protocol):
    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        ...


class _RunLoopNode(Node):
    def __init__(
        self,
        session: AgentSessionProtocol,
        emit_result: Callable[[Mapping[str, Any], Callable[[str], None]], None],
    ) -> None:
        super().__init__()
        self.session = session
        self.emit_result = emit_result

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        console = self.params.get("console")
        output_callback = self.params.get("output_callback")
        if output_callback is None:
            output_callback = create_rich_output(console)
        return {
            "input_iter": self.params.get("input_iter"),
            "output_callback": output_callback,
            "console": console,
        }

    def exec(self, prep_res: Dict[str, Any]) -> int:
        custom_iter = prep_res.get("input_iter")
        console: Optional[Console] = prep_res.get("console")
        iterator: Iterator[str] = (
            custom_iter if custom_iter is not None else _stdin_iterator(console)
        )
        output = prep_res["output_callback"]
        output("[system] Entering Code Agent. Type 'exit' to quit.")
        for raw in iterator:
            message = raw.strip()
            if not message:
                continue
            if message.lower() in {"exit", "quit"}:
                break
            result = self.session.run_turn(message, output_callback=output)
            self.emit_result(result, output)
        return 0

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: int) -> int:
        shared["exit_code"] = exec_res
        return "complete"


class CodeAgentCLIFlow(Flow):
    def __init__(
        self,
        session: AgentSessionProtocol,
        *,
        emit_result: Optional[Callable[[Mapping[str, Any], Callable[[str], None]], None]] = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.emit_result = emit_result or _emit_result
        self.loop_node = _RunLoopNode(self.session, self.emit_result)
        self.start(self.loop_node)

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> int:
        return shared.get("exit_code", 0)


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
    flow = CodeAgentCLIFlow(active_session, emit_result=emit_result)
    flow.set_params(
        {
            "input_iter": input_iter,
            "output_callback": emitter,
            "console": active_console,
        }
    )
    return flow._run({})


def run_code_agent_once(
    prompt: str,
    *,
    session: Optional[AgentSessionProtocol] = None,
    session_factory: Optional[Callable[[], AgentSessionProtocol]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    console: Optional[Console] = None,
    emit_result: Optional[Callable[[Mapping[str, Any], Callable[[str], None]], None]] = None,
) -> Dict[str, Any]:
    """Execute a single Code Agent turn and emit the summarised result."""

    active_session = _resolve_session(session, session_factory)
    if output_callback is None:
        active_console = console or Console()
        emitter = create_rich_output(active_console)
    else:
        emitter = output_callback
    result = active_session.run_turn(prompt, output_callback=emitter)
    (emit_result or _emit_result)(result, emitter)
    return result


def run_cli_main(
    argv: Optional[Sequence[str]] = None,
    *,
    session_factory: Callable[[], AgentSessionProtocol],
    emit_result: Optional[Callable[[Mapping[str, Any], Callable[[str], None]], None]] = None,
) -> int:
    parser = _create_cli_parser()
    args = parser.parse_args(argv)

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
        if prompt_text:
            run_code_agent_once(
                prompt_text,
                session=session,
                output_callback=emitter,
                console=console,
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


def _stdin_iterator(console: Optional[Console] = None) -> Iterable[str]:
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
    return parser


__all__ = [
    "AgentSessionProtocol",
    "CodeAgentCLIFlow",
    "create_rich_output",
    "run_cli_main",
    "run_code_agent_cli",
    "run_code_agent_once",
    "stringify_payload",
]
