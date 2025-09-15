#!/usr/bin/env python3
"""
Typer-based CLI for the code-agent RAG utilities.

Subcommands:
- index:  Index a project directory
- search: Semantic search within an indexed project
- query:  Ask a question using RAG
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from tools.rag_tool import create_rag_tool

app = typer.Typer(add_completion=False, help="Code Agent with RAG integration")


def _print_json(data) -> None:
    try:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        typer.echo(str(data))


@app.command(help="Index a project directory for search and RAG")
def index(project_path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)):
    tool = create_rag_tool()
    res = tool.execute(action="index", project_path=str(project_path))
    if res.get("status") != "success":
        typer.secho(f"Index failed: {res.get('message', 'Unknown error')}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(res.get("message", "Indexed."), fg=typer.colors.GREEN)


@app.command(help="Semantic search in an indexed project")
def search(
    project_name: str = typer.Argument(..., help="Project name (basename of indexed path)"),
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-k", help="Max results"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON output"),
):
    tool = create_rag_tool()
    res = tool.execute(action="search", project_name=project_name, query=query, limit=limit)
    if res.get("status") != "success":
        typer.secho(f"Search failed: {res.get('message', 'Unknown error')}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if json_out:
        _print_json(res)
        return
    total = res.get("total_results", 0)
    typer.secho(f"Found {total} result(s)", fg=typer.colors.GREEN)
    for i, m in enumerate(res.get("matches", []), start=1):
        header = f"{i}. {m.get('file')} (score: {m.get('score', 0.0):.3f})"
        rng = m.get("start_line"), m.get("end_line")
        if all(isinstance(x, int) and x > 0 for x in rng):
            header += f"  lines {rng[0]}-{rng[1]}"
        typer.echo(header)
        snippet = (m.get("content") or "").strip().splitlines()
        preview = " ".join(snippet[:4])
        if len(snippet) > 4:
            preview += " ..."
        if preview:
            typer.echo(f"   {preview}")


@app.command(help="Ask a question using RAG over the codebase")
def query(
    project_name: str = typer.Argument(..., help="Project name (basename of indexed path)"),
    question: str = typer.Argument(..., help="Question to ask"),
    limit: int = typer.Option(5, "--limit", "-k", help="Context results to retrieve"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON output"),
):
    tool = create_rag_tool()
    res = tool.execute(action="query", project_name=project_name, question=question, limit=limit)
    if res.get("status") != "success":
        typer.secho(f"Query failed: {res.get('message', 'Unknown error')}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if json_out:
        _print_json(res)
        return
    typer.secho("Answer:", fg=typer.colors.GREEN)
    typer.echo(res.get("answer", ""))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
