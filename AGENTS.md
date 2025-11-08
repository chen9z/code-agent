# Repository Guidelines

## Project Structure & Module Organization

Runtime entry points (`cli.py`, `code_agent.py`, `codebase_retrieval.py`) wire together reusable building blocks.
Domain logic is split across `core/` (prompts, security, config), `nodes/` (tool planners/executors), `tools/` and
`clients/` (LLM + external integrations). The CLI shell in `cli.py` handles interactive UX, while `integrations/` carries
adapters such as Tree-sitter grammars. Persisted embeddings live under `storage/` (gitignored), and parity tests mirror
the runtime layout inside `tests/`.

## Build, Test, and Development Commands

- `uv venv && source .venv/bin/activate && uv sync` – create/update the managed environment; add
  `uv pip install -e '.[test]'` when working on coverage.
- `uv run python cli.py -w /path/to/repo` – launch the conversational agent against a workspace; omit `-w` to target
  the current tree.
- `uv run pytest` – execute the full regression suite; append `-k retrieval` for focused debugging.
- `uv run pytest tests/test_codebase_retrieval.py -q` – fast signal for the semantic indexer pipeline.
-
`uv run python benchmarks/code_agent_benchmark.py --config benchmarks/examples/embedding_models.json --output benchmarks/results/latest.json` –
compare embedding/model settings before shipping planner changes.

## Coding Style & Naming Conventions

Target Python 3.11, four-space indentation, and PEP 8 line widths (~100 chars). Prefer explicit type hints (see
`code_agent.py`) and docstrings for public helpers. Modules, files, and functions stay `snake_case`; classes are
PascalCase. Use `pathlib.Path` for filesystem work and keep most side effects inside CLI wrappers or nodes.

## Testing Guidelines

All tests run through `pytest`; place mirrors of new modules inside `tests/<module_name>/` and name files
`test_<feature>.py`. Within a file, order fixtures → helpers → `test_*` functions or parameterized classes. Run
`uv run pytest --cov=core --cov=tools --cov=nodes` before pushing and capture flaky seeds in the PR description.

## Commit & Pull Request Guidelines

Follow the existing history pattern: a concise imperative subject with an optional scope prefix (e.g.,
`Refactor: simplify CLI output wiring`). Document behavioral changes, linked issues, and local test/benchmark results in
the PR body; include CLI transcripts or screenshots if UX output shifts. Each PR should stay focused on one feature/fix,
update relevant docs (`README.md`, `docs/`, this guide), and ensure storage artifacts stay untracked.

## Security & Configuration Tips

Secrets are read via `configs/manager.py`; set `LLM_MODEL`, `LLM_API_KEY`, `OPENAI_API_KEY`, and related endpoints in
your shell, never in code. Keep local `.env` files out of Git, and double-check that `storage/` snapshots or benchmark
logs do not leak proprietary code. Use the `CLI_TOOL_TIMEOUT_SECONDS` override sparingly and document why longer tool
windows are required.

要遵守的原则：
1. Development Spirit: Remember: Our development should follow the spirit of Linus.
2. Output with Chinese.
