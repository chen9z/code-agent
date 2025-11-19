# Repository Guidelines

## Project Structure & Module Organization

- 入口：`cli.py`, `code_agent.py`, `codebase_retrieval.py`，负责拼装 Agent/runtime。
- 领域模块：`agent/`（会话 + 提示）、`runtime/`（工具调度）、`retrieval/`（索引/搜索）、`tools/`（Read/Grep、`dataset_log.py` 等）。
- 适配层：`adapters/llm`（LLM/Embedding 客户端）、`adapters/workspace`（Tree-sitter、Qdrant、本地文件视图）。
- 配置：`config/` 统一维护安全提示与默认参数。
- 基准/数据集：`benchmarks/` 包含 planner benchmark 以及 `benchmarks/dataset/` 数据集 orchestrator；流程细节写在 `docs/dataset_synthesis_plan.md`。
- CLI 输出与壳层仍在 `cli.py` + `ui/`，持久化 embedding 存 `storage/`（gitignored），测试在 `tests/` 下与模块镜像。

## Build, Test, and Development Commands

- `uv venv && source .venv/bin/activate && uv sync` – create/update the managed environment; add
  `uv pip install -e '.[test]'` when working on coverage.
- `uv run python cli.py -w /path/to/repo` – launch the conversational agent against a workspace; omit `-w` to target
  the current tree.
- `uv run pytest` – execute the full regression suite; append `-k retrieval` for focused debugging.
- `uv run pytest tests/test_codebase_retrieval.py -q` – fast signal for the semantic indexer pipeline.
- `uv run python benchmarks/dataset/runner.py --queries demo.jsonl` – 运行 DatasetSynthesisAgent 流水线（参见 `docs/dataset_synthesis_plan.md`）。
-
`uv run python benchmarks/code_agent_benchmark.py --config benchmarks/examples/embedding_models.json --output benchmarks/results/latest.json` –
compare embedding/model settings before shipping planner changes.

## Coding Style & Naming Conventions

Target Python 3.11, four-space indentation, and PEP 8 line widths (~100 chars). Prefer explicit type hints (see
`code_agent.py`) and docstrings for public helpers. Modules, files, and functions stay `snake_case`; classes are
PascalCase. Use `pathlib.Path` for filesystem work and keep most side effects inside CLI wrappers or nodes.

## Testing Guidelines

All tests run through `pytest`; place mirrors of new modules inside `tests/<module_name>/` and name files
`test_<feature>.py`. Within a file,保持“fixtures → helpers → tests”的顺序。临近提交流程时执行
`uv run pytest --cov=agent --cov=retrieval --cov=runtime --cov=tools` 并记录任何 flaky seed。Dataset 流水线相关变更需补充/更新 `tests/benchmarks/test_dataset_pipeline.py` 或其它对应测试。

## Commit & Pull Request Guidelines

Follow the existing history pattern: a concise imperative subject with an optional scope prefix (e.g.,
`Refactor: simplify CLI output wiring`). Document behavioral changes, linked issues, and local test/benchmark results in
the PR body; include CLI transcripts or screenshots if UX output shifts. Each PR should stay focused on one feature/fix,
update relevant docs (`README.md`, `docs/`, this guide), and ensure storage artifacts stay untracked.

## Security & Configuration Tips

Secrets are read via `config.config.get_config()`; set `LLM_MODEL`, `LLM_API_KEY`, `OPENAI_API_KEY`, and related
endpoints in your shell, never in code. Keep local `.env` files out of Git, and double-check that `storage/` snapshots
or benchmark logs do not leak proprietary code. Use the `CLI_TOOL_TIMEOUT_SECONDS` override sparingly and document why
 longer tool windows are required。对 DatasetSynthesisAgent 来说，`dataset_log_write_chunk` 仅负责单 chunk 校验（raw_samples 持久化已暂时禁用），仍需禁止未经授权的路径输出。

要遵守的原则：
1. **Development Spirit: Remember: Our development should follow the spirit of Linus.**
2. Output with Chinese.
