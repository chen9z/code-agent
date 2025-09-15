# Architecture Overview

This project is a code-agent built on a Flow/Node runtime. The core Flow/Node code remains unchanged; higher-level features (RAG, cursor-like agent, PR reviewer) are assembled from nodes and tools.

## Layers
- core (existing): Flow/Node execution, retries, routing. No changes for now.
- agents: Agent entrypoints and flows (e.g., code_chat, cursor_like, pr_reviewer). Each agent wires nodes into a task-specific pipeline.
- tools: Reusable actions with JSON-schema parameters (e.g., `rag_tool`). Tools avoid global state and are idempotent when possible.
- integrations: External adapters (code repository index/search, VCS, CI). `integration/repository.py` provides a functional fallback `ChatRepository` for local indexing/search.
- clients: Service adapters (LLM, Vector DB). Centralize API usage, retries, and streaming.
- config: Centralized settings via Pydantic with env prefixes: `LLM_*` for LLM, `RAG_*` for embedding/rerank. Other settings use sensible defaults.
- ui: CLI/TUI entrypoints and HTTP adapters.

## Module Responsibilities
- agents/code_chat: RAG index/search/query flows reusing `tools.rag_tool` and `integration.repository`.
- tools/rag_tool.py: Action router for `index|search|query`, formatting of results, optional LLM usage.
- integration/repository.py: `ChatRepository` implements local indexing (line-chunked) and simple search scoring; bridges to external chat-codebase when available via `CHAT_CODEBASE_PATH`.
- clients/llm.py (planned): Unified LLM client (litellm), streaming, caching; `model.py` to be replaced gradually.

## Data & Indexing
- Project key: `<project_name>` (default basename of path). Storage under `./storage/` (ignored by VCS).
- Document model: `{path, content, chunk_id, start_line, end_line, score}`. Chunks are max ~200 lines to keep context focused.

## CLI & Usage
- Entry: `code-agent` (mapped to `main:main`).
- RAG examples: `rag_flow.run_rag_workflow(action=...)` or use `tools.rag_tool` directly.

## Extensibility Guidelines
- New tools subclass `BaseTool` and expose `name`, `description`, `parameters`, `execute`.
- New agents compose existing nodes/tools; keep side-effecting nodes isolated.
- Config is read via `config.manager.get_config()`; avoid direct `os.environ` in tools/nodes.

## Roadmap (excerpt)
- clients/llm.py + caching; replace direct `model.call_llm`.
- Typer CLI with subcommands for `index|search|query` and agent runners.
- PR reviewer agent: diff ingestion, risk categorization, suggestions.
