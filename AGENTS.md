# Repository Guidelines

## Project Structure & Modules
- `main.py`: CLI/demo entrypoint for the agent.
- `rag_flow.py`: Orchestrates Node/Flow RAG workflow.
- `tools/`: Tooling (`rag_tool.py`) plus bases (`base.py`, `base_node.py`).
- `config/`: Centralized settings (`manager.py`, `rag_config.py`) via Pydantic; env prefixes `RAG_`, `LLM_`, `VECTORDB_`, `APP_`.
- `integration/`: External adapters (Qdrant + chat-codebase bridge).
- `tests/`: Pytest suite (`test_*.py`, `conftest.py`).
- `storage/`: Local vector store (Qdrant) data.
- Docs: `requirements.md`, `design.md`, `tasks.md`, `CLAUDE.md`, `docs/ARCHITECTURE.md`.

## Build, Test, and Development
- Python: `>=3.11`. Package manager: `uv`.
- Setup venv: `uv venv && source .venv/bin/activate && uv sync`
- Install (editable + tests): `uv pip install -e '.[test]'`
- Run tests: `uv run pytest` (uses `pytest.ini` → `-v --tb=short`).
- Run demo: `uv run python main.py`
- Tip: `OPENAI_API_BASE`/`OPENAI_API_KEY` and optional `CHAT_CODEBASE_PATH` are read (also via `.env`). Vector store defaults to `./storage`.

## Coding Style & Naming
- Follow PEP 8; 4‑space indentation; add type hints where practical.
- Names: modules and functions `snake_case`; classes `PascalCase` (e.g., `RAGIndexNode`).
- Docstrings: triple double quotes; keep functions small and cohesive.
- JSON‑schema for tool params lives on the tool (`parameters` property).

## Testing Guidelines
- Framework: Pytest. Place tests under `tests/` as `test_*.py`; test functions `test_*`.
- Run focused tests: `uv run pytest tests/test_rag.py::test_rag_flow -q`.
- Optional coverage (if installed): `uv run pytest --cov`.

## Commit & Pull Requests
- Commits: imperative mood, concise scope, link issues (e.g., `Fix: handle empty search results (#123)`).
- PRs must include:
  - Clear description and rationale; linked issue(s).
  - Test plan/results (commands used and output summary).
  - Updates to docs if behavior/config changes (`requirements.md`, `design.md`, `tasks.md`).
  - Screenshots/logs when helpful (CLI output for flows/tests).

## Security & Configuration
- Configure via env vars: `RAG_*`, `LLM_*`, `VECTORDB_*`, `APP_*`, plus `OPENAI_API_BASE`, `OPENAI_API_KEY`, and optional `CHAT_CODEBASE_PATH`.
- Example:
  - `export OPENAI_API_BASE=https://api.deepseek.com`
  - `export OPENAI_API_KEY=...`
- Avoid committing secrets or large files in `storage/`.

## Minimal RAG Examples
```python
from rag_flow import run_rag_workflow
# Index
run_rag_workflow(action="index", project_path="/path/to/project")
# Search
run_rag_workflow(action="search", project_name="project", query="function definition")
# Query
run_rag_workflow(action="query", project_name="project", question="How does it work?")
```

## Agent-Specific Notes
- New tools should subclass `tools.base.BaseTool`; expose `name`, `description`, `parameters`, `execute`.
- New flows/nodes should extend `Node`/`Flow` and route by action strings (see `rag_flow.py`).
