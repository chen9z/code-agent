# Architecture Overview

This project is a code agent focused on local repository understanding. Flow/Node abstractions have been trimmed back in favour of a lightweight session layer living in `agent/`.

## Layers
- **Entrypoints**：`cli.py`, `code_agent.py`, `codebase_retrieval.py` 负责解析参数并装配后端模块。
- **Agent**：会话与提示协调在 `agent/`，`agent/session.py` 提供共享的 turn loop。
- **Runtime**：`runtime/` 提供工具调度（`runtime/tool_runner.py`）。
- **Retrieval**：语义索引、切片与搜索逻辑位于 `retrieval/index.py`, `retrieval/codebase_indexer.py`, `retrieval/splitter.py`。
- **Adapters**：`adapters/llm`（LLM/Embedding 客户端）、`adapters/workspace`（Tree-sitter、Qdrant、本地文件视图）。
- **Tools & UI**：`tools/` 中实现 Read/Grep 等检索工具以及 `tools/dataset_log.py`；`ui/` 负责 CLI 输出格式。
- **Benchmarks & Dataset**：`benchmarks/code_agent_benchmark.py` 评估 planner/embedding，`benchmarks/dataset/` 编排 DatasetSynthesisAgent（参见 `docs/dataset_synthesis_plan.md`）。
- **Config**：集中式设置与安全提示在 `config/`。

## Module Responsibilities
- `codebase_retrieval.py`：面向 CLI/benchmark 的索引与搜索助手，封装 `retrieval.index.Index`。
- `retrieval/index.py`：暴露 `index_project`、`search`、格式化工具，负责与 Qdrant/Tree-sitter 协作。
- `retrieval/codebase_indexer.py`：切片与 embedding pipeline，依赖 `adapters/llm.embedding` 与 `adapters/workspace.vector_store`。
- Dataset runner 直接配置 `CodeAgentSession`（受限工具/温度 0）用于数据集生成。
- `tools/dataset_log.py`：`dataset_log_write_chunk` 工具实现，逐条验证 golden_chunk 并写入 raw_samples JSONL。
- `benchmarks/dataset/runner.py`：数据集 orchestrator，串联快照→Agent→聚合→过滤全流程。
- `adapters/llm/llm.py`：统一的 OpenAI/Opik 兼容客户端，提供 tracing hook。

## Data & Indexing
- Project key：`<project_name>`（默认为工作区 basename）。快照及向量集合写入 `./storage/`（gitignored）。
- 检索 chunk 模型：`{path, start_line, end_line, chunk_id, score}`。`retrieval.splitter` 控制最大行数（默认约 2048 行）。
- 数据集 golden_chunk：`{path, start_line, end_line, confidence}`，由 `dataset_log_write_chunk` 在静态快照上校验并写入 raw_samples JSONL；`build_dataset_from_raw` 内联聚合逻辑负责后续整合。

## Usage
- 程序化入口：示例脚本可通过 `codebase_retrieval.main` 调用 `index_project`；复杂集成走 `retrieval.index.Index`。
- 数据集流水线：运行 `uv run python benchmarks/dataset/runner.py --queries demo.jsonl`，详见 `docs/dataset_synthesis_plan.md`。

## Extensibility Guidelines
- 工具实现保持无状态/幂等，统一通过 `runtime.tool_runner.ToolExecutionRunner` 调度。
- Retrieval 逻辑应借助 `adapters/workspace` 处理文件与向量存储；不要直接操作 Qdrant API。
- DatasetSynthesisAgent 工具需保持单 chunk 校验逻辑；如需调整 raw_samples schema，记得同步 `tools/dataset_log.py` 与 `docs/dataset_synthesis_plan.md`。
- 配置通过 `config.config.get_config()` 读取，勿在功能代码中直接访问环境变量。

## Roadmap (excerpt)
- adapters/llm caching; replace ad-hoc client instantiations.
- Typer CLI with subcommands for `index|search|query` and agent runners.
- PR reviewer agent: diff ingestion, risk categorization, suggestions.
