# Code Agent

Composable agent runtime focused on code understanding. Index repositories, run semantic search, and answer questions with retrieved context. Ships with a fallback stub LLM and optional OpenAI-compatible clients.

## Project Layout

- Flow/node runtime primitives exported directly via top-level `__init__`.
- Agent modules (e.g., `code_rag.py`, `tool_agent.py`) built from reusable nodes and tools.
- `integrations/` – adapters for repositories, vector stores, external bridges.
- `configs/` – environment-driven configuration helpers.
- `clients/` / `tools/` – LLM clients and tool abstractions.
- `tests/` – pytest suite mirroring runtime modules.

Vector data persists in `storage/` (gitignored).

## Quick Start

Prerequisites: Python ≥ 3.11 and `uv`.

```bash
uv venv && source .venv/bin/activate
uv sync
# optional extras
uv pip install -e '.[test]'
```

Optional configuration:

```bash
export OPENAI_API_BASE=https://api.deepseek.com
export OPENAI_API_KEY=sk-...
export CHAT_CODEBASE_PATH=/path/to/chat-codebase  # optional deep integration
```

## Python API

Minimal Code-RAG examples:

```python
from rag_flow import run_rag_workflow

# Index
run_rag_workflow(action="index", project_path="/path/to/project")

# Search
res = run_rag_workflow(action="search", project_name="project", query="function definition", limit=5)

# Query
res = run_rag_workflow(action="query", project_name="project", question="How does it work?", limit=5)
```

## Development

- Run tests: `uv run pytest`
- Focused flow test: `uv run pytest tests/test_rag.py::test_rag_flow -q`
- Demo CLI banner: `uv run python main.py`

Refer to `AGENTS.md` for contributor guidance and coding conventions.
