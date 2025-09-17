# Repository Guidelines

## Project Structure & Module Organization
- `core/`: Runtime primitives and shared node scaffolding; import flows via `core.base` to keep dependencies predictable.
- `agents/`: Ready-to-run agent flows such as `agents/code_rag.py`; every new flow must be registered with `agents.register_agent` and documented in-module.
- `integrations/`: Connectors for repos, embeddings, and other backends (e.g., `integrations/repository.py`); rely on dependency injection from flows.
- `configs/`: Environment-driven settings managed through `configs/manager.py`; read via the provided helpers instead of `os.environ` directly.
- Supporting directories: `clients/` & `tools/` host LLM adapters and reusable tool abstractions, `nodes/` contains reusable flow nodes, while `tests/` mirrors this tree for pytest coverage.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate`: create and enter the project virtualenv.
- `uv sync` or `uv pip install -e '.[test]'`: install editable dependencies plus test extras.
- `uv run pytest`: execute the full automated test suite.
- `uv run pytest tests/test_rag.py::test_rag_flow -q`: target the primary RAG flow regression test.
- `uv run python main.py`: launch the CLI demo from the configured workspace.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, and PEP 8 naming (`snake_case` functions/modules, `PascalCase` classes).
- Keep node/flow units short and composable; prefer constructor injection for integrations to ease testing.
- Limit comments to non-obvious orchestration; rely on descriptive function names instead of inline narration.

## Testing Guidelines
- Pytest is the default harness; tests reside under `tests/` with `test_*.py` modules mirroring runtime packages.
- Add focused unit tests for nodes, tools, and integrations; stub external services with fixtures under `tests/fixtures/` when needed.
- Run `uv run pytest --maxfail=1 -q` before submitting to surface fast failures locally.

## Commit & Pull Request Guidelines
- Write commits in imperative tense with scoped prefixes (e.g., `Add: register new retriever`).
- PRs must include a summary, rationale, linked issue when applicable, and a `Test Plan` enumerating commands executed; attach logs or screenshots for UX or CLI output changes.

## Security & Configuration Tips
- Read secrets via environment variables such as `OPENAI_API_KEY`, `LLM_*`, and `RAG_*`; never hardcode credentials.
- Keep local vector artifacts under `storage/` (gitignored) and exclude large binaries from commits.

## Agent-Specific Instructions
- Compose new agents by wiring reusable nodes within `core.Flow`; extend existing patterns in `agents/code_rag.py` for reference.
- Register each agent with `register_agent("your_agent", factory)` to expose it through the CLI and document entrypoints in the agent module docstring.
