# Architecture Overview

This project is a code-agent built on a Flow/Node runtime. The Flow/Node code, exposed via the package `__init__`, remains unchanged; higher-level features (RAG, cursor-like agent, PR reviewer) are assembled from nodes and tools.

## Layers
- flow runtime: Flow/Node execution, retries, routing. No changes for now.
- agents: Agent entrypoints and flows (e.g., code_chat, cursor_like, pr_reviewer). Each agent wires nodes into a task-specific pipeline.
- tools: Node implementations and utilities (e.g., `rag_nodes`). Nodes avoid global state and are idempotent when possible.
- integrations: External adapters (code repository index/search, VCS, CI). `integrations/repository.py` provides a functional fallback `ChatRepository` for local indexing/search.
- clients: Service adapters (LLM, Vector DB). Centralize API usage, retries, and streaming.
- config: Centralized settings via Pydantic with env prefixes: `LLM_*` for LLM, `RAG_*` for embedding/rerank. Other settings use sensible defaults.
- ui: CLI/TUI entrypoints and HTTP adapters.

## Module Responsibilities
- code_rag.py: RAG index/search/query flows reusing `integrations.repository` and LLM clients via nodes.
- integrations/repository.py: `ChatRepository` implements local indexing (line-chunked) and simple search scoring; bridges to external chat-codebase when available via `CHAT_CODEBASE_PATH`.
- clients/llm.py (planned): Unified LLM client (litellm), streaming, caching; `model.py` to be replaced gradually.

## Data & Indexing
- Project key: `<project_name>` (default basename of path). Storage under `./storage/` (ignored by VCS).
- Document model: `{path, content, chunk_id, start_line, end_line, score}`. Chunks are non-overlapping and max ~RAG_CHUNK_SIZE lines (default 200).

## Usage
- Programmatic API: `code_rag.run_rag_workflow(action=...)`.

## Extensibility Guidelines
- New nodes extend `Node`; keep side effects isolated and parameters explicit.
- Compose flows from small nodes; avoid global state.
- Config is read via `configs.manager.get_config()`; avoid direct `os.environ` in nodes.

## Roadmap (excerpt)
- clients/llm.py + caching; replace direct `model.call_llm`.
- Typer CLI with subcommands for `index|search|query` and agent runners.
- PR reviewer agent: diff ingestion, risk categorization, suggestions.
