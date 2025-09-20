# CLI Code Agent Plan

## 实施计划
- 梳理现有代理框架：通读 `tool_agent.py`, `code_rag.py`, `nodes/tool_execution.py`, `tools/registry.py`，确认多工具并行/串行调度、`Flow` 生命周期以及配置注入方式。
- 设计 CLI Code Agent 交互：在根目录定义 `code_agent.py`，通过 `Flow`/`Node` 驱动最简交互循环。
- 拓展可复用节点：复用 `ToolAgentFlow`，必要时包装单节点实现提示输出。
- 集成工具与 LLM：使用默认 `create_default_registry()` 注册全部工具，保留配置中心化。
- CLI 驱动与输出：在 `main.py` 中内联解析逻辑，调用 `run_code_agent_cli`，实时输出规划、工具和总结。
- 测试与文档：编写 `tests/test_code_agent_session.py` 覆盖会话/CLI 循环，更新文档说明使用方式。

## 待办事项
- [x] 阅读并记录 `__init__.py`, `Flow`, `Node`, `ToolExecutionBatchNode` 的扩展点。
- 覆盖测试：`tests/runtime/test_flow_primitives.py`
- [x] 勾勒 `code_agent.py` 简化交互循环，实现 Flow/Node 调用。
- [x] 定义 `create_code_agent_flow` 复用 `ToolAgentFlow`。
- 覆盖测试：`tests/test_code_agent_session.py`
- [x] 保证默认注册全部工具。
- [x] 在 `main.py` 内联 CLI 解析并调用 `run_code_agent_cli`。
- [x] 更新文档说明使用与测试命令。

### CLI 接入概要
- Factory：`code_agent.create_code_agent_flow` 直接包装 `ToolAgentFlow`，默认注册目录下全部工具。
- 会话/CLI：`CodeAgentSession` 在内存中维护历史，`CodeAgentCLIFlow` 借助 `_RunLoopNode` 输出规划 (`[planner]`)、工具 (`[tool]`) 与最终回答 (`[assistant]`)。
- 入口：`main.py` 内联解析，默认执行 `agent` 命令；`README` 提供运行示例。

### 运行时扩展点速记
- `BaseNode`/`Node`：通过 `prep -> exec -> post` 钩子串联；`Node` 增加重试与 `exec_fallback`，适合处理瞬时失败；`BatchNode` 在 `_exec` 层批处理条目。
- `Flow`：以 `start()` 注册起点，通过 `node >> next` 和 `node - "action" >> branch` 管理分支；`params` 在循环中注入到每个节点。
- `Async*`：异步节点/Flow 复用同一决策逻辑，重写 `*_async` 钩子即可并行化开销节点。
- `ToolExecutionBatchNode`：`tool_plan.tool_calls` 驱动执行；`mode`/`parallel` 控制批次；`post` 将结果写入 `shared["tool_results"]` 与历史，为后续总结节点提供输入；`max_parallel_workers` 可调节并发限额。
