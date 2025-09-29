#!/usr/bin/env python3
"""Unified entrypoint for the code-agent CLI suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from code_agent import CodeAgentSession, run_code_agent_cli


def _stream_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


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
    try:
        os.chdir(workspace)
        if prompt:
            session = CodeAgentSession()
            # Run a single turn via the CLI loop to reuse logging behaviour.
            inputs = iter([prompt, "exit"])
            return run_code_agent_cli(
                session=session,
                input_iter=inputs,
                output_callback=_stream_print,
            )
        return run_code_agent_cli(output_callback=_stream_print)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
