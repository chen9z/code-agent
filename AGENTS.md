# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI/demo entrypoint for the agent.
- `rag_flow.py`: Orchestrates Node/Flow RAG workflow.
- `tools/`: Tooling (`rag_tool.py`) and bases (`base.py`, `base_node.py`).
- `config/`: Centralized settings via Pydantic (`manager.py`). Env prefixes: `LLM_*` (LLM), `RAG_*` (embedding/rerank).
- `integration/`: External adapters (e.g., Qdrant, chat‑codebase bridge).
- `tests/`: Pytest suite (`test_*.py`, `conftest.py`).
- `storage/`: Local vector store data (Qdrant). Avoid committing large files/secrets.
- Docs: `requirements.md`, `design.md`, `tasks.md`, `docs/ARCHITECTURE.md`.

## Build, Test, and Development
- Requires Python `>=3.11` and `uv`.
- Setup: `uv venv && source .venv/bin/activate && uv sync`.
- Install (editable + tests): `uv pip install -e '.[test]'`.
- Run tests: `uv run pytest` (uses `pytest.ini` → `-v --tb=short`).
- Focused test: `uv run pytest tests/test_rag.py::test_rag_flow -q`.
- Run demo: `uv run python main.py`.

## Coding Style & Naming Conventions
- Follow PEP 8; 4‑space indentation; add type hints where practical.
- Names: modules/functions `snake_case`; classes `PascalCase` (e.g., `RAGIndexNode`).
- Docstrings: triple double quotes; keep functions small and cohesive.
- Tool params JSON‑schema lives on the tool (`parameters` property).

## Testing Guidelines
- Framework: Pytest. Place tests under `tests/` as `test_*.py`; functions `test_*`.
- Keep tests fast and deterministic; favor unit coverage for nodes/tools/flows.
- Optional coverage (if installed): `uv run pytest --cov`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope; link issues (e.g., `Fix: handle empty search results (#123)`).
- PRs must include: clear description + rationale, linked issue(s), test plan/results (commands used + output summary), and docs updates for behavior/config changes. Add CLI logs/screenshots when helpful.

## Security & Configuration
- Configure via env vars: `RAG_*`, `LLM_*`, `VECTORDB_*`, `APP_*`, plus `OPENAI_API_BASE`, `OPENAI_API_KEY`, optional `CHAT_CODEBASE_PATH`.
- Example: `export OPENAI_API_BASE=https://api.deepseek.com`; `export OPENAI_API_KEY=...`.
- Vector store defaults to `./storage`; avoid committing secrets or large artifacts.

## Agent‑Specific Notes
- New tools: subclass `tools.base.BaseTool`; expose `name`, `description`, `parameters`, `execute`.
- New flows/nodes: extend `Node`/`Flow`; route by action strings (see `rag_flow.py`).
- Use `config.manager` for settings; prefer dependency injection for testability.
