# Repository Guidelines

## Project Structure & Module Organization
- Core orchestrators — `code_agent.py`, `codebase_retrieval.py`, `main.py` — assemble flows from shared logic in `core/`, `nodes/`, and `tools/`.
- CLI entry points stay in `cli/`; client adapters and config helpers live in `clients/`, `integrations/`, and `configs/`.
- Benchmarks and transcripts belong in `benchmarks/`; user-facing docs reside in `docs/`. Tests mirror runtime modules inside `tests/`, while gitignored embeddings land in `storage/`.

## Build, Test, and Development Commands
- `uv sync` — install dependencies; rerun after editing `pyproject.toml` or `uv.lock`.
- `uv run python main.py` — launch the interactive agent. Add `-w <repo>` to target another workspace or `-p "List TODOs"` for a one-off prompt.
- `uv run pytest` — execute the suite; focus with `uv run pytest tests/test_tool_agent_flow.py -q`.
- `uv run python benchmarks/code_agent_benchmark.py --config benchmarks/examples/embedding_models.json` — regenerate embedding performance reports under `benchmarks/results/`.

## Coding Style & Naming Conventions
- Python 3.11+, four-space indentation, type annotations by default, and double-quoted strings unless another delimiter reduces escaping.
- Keep functions focused; move shared operations into `nodes/` or `tools/`. Prefer descriptive flow/node names (`SemanticCodeIndexer`, `ToolPlanningNode`), and align new test modules with their targets.

## Testing Guidelines
- Provide pytest coverage for every behavior change; share fixtures through `tests/conftest.py`.
- Combine scenario tests (CLI, planner) with narrow unit checks when touching runtime logic (see `tests/test_cli_output.py`).
- Run `uv run pytest --cov=core --cov=nodes --cov=tools` before submitting; treat coverage drops as blockers.

## Commit & Pull Request Guidelines
- Use `<Verb>: <summary>` commit headers consistent with history (`Refactor: simplify codebase retrieval helpers`). Expand body text only when rationale or rollout steps need clarification.
- PRs must outline scope, manual verification, and linked issues. Attach transcripts or screenshots if CLI UX changes.
- Update `README.md` or `docs/` for user-facing shifts and surface configuration updates explicitly.

## Security & Configuration Tips
- Inject secrets through environment variables consumed by `configs/manager.py`; never commit `.env` files.
- Clear `storage/` before sharing reproductions and document migrations when embedding schemas change.
- Document new external tool requirements (scopes, rate limits) in `docs/` and expose toggle flags in configuration.
