#!/usr/bin/env python3
"""Unified entrypoint for the code-agent CLI suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from rich.console import Console

from code_agent import (
    CodeAgentSession,
    create_rich_output,
    run_code_agent_cli,
    run_code_agent_once,
)


def main(argv: Sequence[str] | None = None) -> int:
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

    original_cwd = Path.cwd()
    console = Console()
    emitter = create_rich_output(console)
    try:
        os.chdir(workspace)
        session = CodeAgentSession(max_iterations=100)
        if prompt:
            run_code_agent_once(
                prompt,
                session=session,
                output_callback=emitter,
                console=console,
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
