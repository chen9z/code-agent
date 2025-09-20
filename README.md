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

Optional configuration (env vars consumed via `configs/manager.py`):

```bash
export LLM_MODEL=deepseek-chat
export LLM_API_KEY=sk-...
export LLM_API_BASE=https://api.deepseek.com
export OPENAI_API_KEY=sk-...            # compatibility fallback
export OPENAI_API_BASE=https://api.openai.com/v1
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

## Agent CLI

Launch the code agent from the terminal and interact conversationally:

```bash
uv run python main.py          # starts the agent
# or explicitly
uv run python main.py agent
```

Type your prompts when the CLI shows `You: ` and enter `exit` (or `quit`) to finish. The agent streams planning thoughts, tool activity, and final answers inline.

## Development

- Run tests: `uv run pytest`
- Focused flow test: `uv run pytest tests/test_rag.py::test_rag_flow -q`
- Agent CLI smoke test: `uv run python main.py`

Refer to `AGENTS.md` for contributor guidance and coding conventions.
