# Repository Guidelines

## Project Structure & Module Organization
- `core/`: Runtime primitives for flows and nodes; import from `core.base` or top-level `__init__`.
- `agents/`: Ready-to-run agent flows (e.g., `agents/code_rag.py`). Use `register_agent` to expose new flows.
- `integrations/`: Adapters for external systems such as repositories and vector stores (`integrations/repository.py`).
- `configs/`: Environment-driven configuration (`configs/manager.py`).
- `clients/` & `tools/`: LLM clients and reusable tool abstractions.
- `tests/`: Pytest suite mirroring runtime layout.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate`: create and activate the local virtualenv.
- `uv sync` or `uv pip install -e '.[test]'`: install editable dependencies.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_rag.py::test_rag_flow -q`: focused flow test.
- `uv run python main.py`: demo the CLI entrypoint.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, PEP 8-aligned naming (`snake_case` for functions/modules, `PascalCase` for classes).
- Keep node/flow logic small and composable; prefer dependency injection for integrations.
- Use concise comments only where non-obvious orchestration occurs.

## Testing Guidelines
- Tests live under `tests/` with `test_*.py` / `test_*` functions.
- Prefer unit coverage for nodes, tools, and agent flows; integrate stubs for external services.
- Run `uv run pytest --maxfail=1 -q` before submitting to ensure the fast path passes.

## Commit & Pull Request Guidelines
- Commits: imperative mood, scoped messages (e.g., `Refactor: reorganize runtime`).
- PRs: include summary, rationale, linked issue if applicable, and `Test Plan` section with commands executed. Attach screenshots/logs for UX changes.

## Security & Configuration Tips
- Configure runtime via environment variables (`LLM_*`, `RAG_*`, `OPENAI_API_KEY`, `CHAT_CODEBASE_PATH`).
- Avoid committing secrets or large artifacts; local vector data stays in `storage/` (gitignored).

## Agent-Specific Notes
- New agents should subclass existing nodes or compose tools via `core.Flow`. Register with `agents.register_agent("my_agent", factory)` and document entrypoints under `agents/`.
