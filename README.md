# Code Agent (RAG)

Lightweight codebase search and RAG utilities. Index a repository, semantically search code, and ask questions grounded in relevant snippets. Works offline with a stub LLM; optionally uses OpenAI‑compatible APIs when configured.

## Quick Start

Prereqs: Python >= 3.11 and uv.

- Setup env
  - `uv venv && source .venv/bin/activate && uv sync`
  - Install (editable + tests): `uv pip install -e '.[test]'`
- Optional API config
  - `export OPENAI_API_BASE=https://api.deepseek.com`
  - `export OPENAI_API_KEY=...`
- Optional chat-codebase bridge
  - `export CHAT_CODEBASE_PATH=/path/to/chat-codebase` (fallback repo is used if absent)

## Notes
- Vector store defaults to `./storage` (ignored by VCS).

## Python API

Minimal RAG examples:

```python
from rag_flow import run_rag_workflow

# Index
run_rag_workflow(action="index", project_path="/path/to/project")

# Search
res = run_rag_workflow(action="search", project_name="project", query="function definition", limit=5)

# Query
res = run_rag_workflow(action="query", project_name="project", question="How does it work?", limit=5)
```

## Test

- Run tests: `uv run pytest` (uses `-v --tb=short`).
- Focused: `uv run pytest tests/test_rag.py::test_rag_flow -q`.
