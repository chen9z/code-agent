#!/usr/bin/env python3
"""Unified entrypoint for the code-agent CLI suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from code_agent import create_tool_agent_flow, run_code_agent_cli
from tools.registry import create_default_registry


def run_embedding_model_demo(workspace: Path, model: str = "deepseek-chat") -> int:
    """Run a one-off agent turn to enumerate embedding models in the workspace."""

    target = workspace.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Workspace does not exist: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {target}")

    registry = create_default_registry(include=["glob", "grep", "read"])
    flow = create_tool_agent_flow(registry=registry, model=model)

    prompt = (
        "在当前代码仓库中，列出与 Embedding model 相关的实现文件。"
        " 可以结合 glob、grep、read 等工具来定位并总结结果，"
        "最终请输出文件路径及对应的模型名称。"
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(target)
        shared = {"user_input": prompt, "output_callback": print}
        result = flow.run(shared)
    finally:
        os.chdir(original_cwd)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Code Agent command-line interface")
    parser.add_argument(
        "command",
        nargs="?",
        default="agent",
        help="Command to execute ('agent' for CLI, 'embedding-demo' for the walkthrough).",
    )
    parser.add_argument(
        "--workspace",
        default="/Users/looper/workspace/spring-ai",
        help="Workspace path used by example commands.",
    )
    args = parser.parse_args(argv)

    if args.command == "agent":
        return run_code_agent_cli()

    if args.command == "embedding-demo":
        return run_embedding_model_demo(Path(args.workspace))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
