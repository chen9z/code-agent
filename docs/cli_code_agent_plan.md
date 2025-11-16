# CLI Code Agent Plan

## 实施计划（基于现有精简架构）
- 梳理当前核心：阅读 `agent/session.py`（会话与消息链路）、`runtime/tool_runner.py`（工具并发执行）、`config/prompt.py` 与 `config/config.py`（提示与配置）、`retrieval/index.py`（索引/搜索适配器）、`adapters/llm/llm.py`（LLM 客户端）。
- 设计 CLI 交互：根目录保留 `cli.py` 与 `code_agent.py`，由 `CodeAgentSession` 直接驱动工具调用，无 Flow/Node 框架。
- 集成工具与 LLM：继续用 `tools.registry.create_default_registry()` 注册全部工具，LLM 客户端统一由 `adapters/llm/llm.py` 构造。
- 输出与体验：`cli.py` 内联解析与循环，借助 `ui/emission.py`、`ui/rich_output.py` 输出用户/助手/工具事件。
- 测试与文档：补充/维护 `tests/agent`、`tests/runtime`、`tests/retrieval` 等对应层级的用例；同步 README、AGENTS、ARCHITECTURE 的目录描述。

## 待办事项
- [x] 复核会话、工具执行、检索链路（见上）。
- [ ] 增补 `tests/agent/test_session.py` 等覆盖并行工具、超时、消息保留等路径。
- [ ] 检查并更新示例/脚本（如 benchmarks）导入路径，防止遗留 `integrations*`/`clients*`。
- [ ] 文档对齐：确保 README、AGENTS、ARCHITECTURE 仅引用新的目录名与职责。

### CLI 接入概要
- Session：`CodeAgentSession` 使用系统提示（`config/prompt.py`）与工具注册表执行 LLM 工具调用，默认迭代上限 25。
- 入口：`cli.py` 解析 `-w/--workspace`、`-p/--prompt`，调用 `CodeAgentSession.run_turn`；`code_agent.py` 提供 `run_code_agent_cli`/`main` 包装。
- 并发：`runtime/tool_runner.py` 采用 `ThreadPoolExecutor` 并发执行多个工具调用并保持调用顺序。

### 运行时扩展点速记
- 工具返回格式：`{"status", "content", "data": {"display": ...}}`；未提供 `display` 会降级为 content/status。
- 超时注入：`ToolExecutionRunner` 默认将 `timeout` 毫秒传给支持的工具（如 Bash）。
- 检索管线：`retrieval/codebase_indexer.py` 负责 chunk/embedding/Qdrant 落盘；`retrieval/index.py` 暴露 `Index.index_project/search/format_search_results`。
