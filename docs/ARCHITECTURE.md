# Architecture Overview

This project is a code agent focused on local repository understanding. Flow/Node abstractions have been trimmed back in favour of a lightweight session layer living in `agent/`.

## Layers
- entrypoints: `cli.py`, `code_agent.py`, `codebase_retrieval.py` wire together the lower layers.
- agent: Session + prompt orchestration (`agent/session.py`), shared by CLI and benchmarks.
- runtime: Tool execution runner (`runtime/tool_runner.py`) that fans out OpenAI tool calls in parallel.
- retrieval: Semantic indexing/search pipeline plus chunking helpers (`retrieval/index.py`, `retrieval/codebase_indexer.py`, `retrieval/splitter.py`).
- adapters: External integrations – `adapters/llm/` for OpenAI-compatible chat clients, `adapters/workspace/` for Tree-sitter parsing and Qdrant-backed vector storage.
- config: Centralized settings and prompt pieces in `config/`.
- tools & ui: Tool implementations (`tools/`) and CLI emitters (`ui/`).

## Module Responsibilities
- codebase_retrieval.py: Lightweight helpers for indexing/search built on `retrieval.index.Index`.
- retrieval/index.py: Project adapter exposing `index_project`, `search`, and formatting utilities.
- retrieval/codebase_indexer.py: Chunking + embedding pipeline backed by LiteLLM embeddings and the local Qdrant store.
- adapters/llm/llm.py: Unified LLM client (litellm/OpenAI) with optional Opik tracking.

## Data & Indexing
- Project key: `<project_name>` (default basename of path). Storage under `./storage/` (ignored by VCS).
- Document model: `{path, content, chunk_id, start_line, end_line, score}`. Chunks are non-overlapping and max ~RAG_CHUNK_SIZE lines (default 200).

## Usage
- Programmatic API: direct helpers `index_project`, `search_project`.

## Extensibility Guidelines
- Tool implementations remain stateless and idempotent; wire them into `runtime.ToolExecutionRunner`.
- Retrieval helpers should route filesystem and vector-store concerns through `adapters/workspace`.
- Config is read via `config.config.get_config()`; avoid direct `os.environ` access in feature code.

## Roadmap (excerpt)
- adapters/llm caching; replace ad-hoc client instantiations.
- Typer CLI with subcommands for `index|search|query` and agent runners.
- PR reviewer agent: diff ingestion, risk categorization, suggestions.
