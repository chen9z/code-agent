#!/usr/bin/env python3
"""
Minimal entrypoint for the code-agent with RAG integration.

This script intentionally avoids a full CLI to keep the codebase tidy.
Use the Python API via `rag_flow.run_rag_workflow` in your own scripts.
"""

from __future__ import annotations

import os


def main() -> None:
    print("Code Agent (RAG)")
    print("-----------------")
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found. Using stub LLM (offline mode).")
    print("Use rag_flow.run_rag_workflow(action=..., **kwargs) for programmatic use.")


if __name__ == "__main__":
    main()
