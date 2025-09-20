#!/usr/bin/env python3
"""Unified entrypoint for the code-agent CLI suite."""

from __future__ import annotations

import argparse
from typing import Sequence

from code_agent import run_code_agent_cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Code Agent command-line interface")
    parser.add_argument(
        "command",
        nargs="?",
        default="agent",
        help="Command to execute (only 'agent' is supported).",
    )
    args = parser.parse_args(argv)

    if args.command == "agent":
        return run_code_agent_cli()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
