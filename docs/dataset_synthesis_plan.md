# CodeAgent RAG 评测数据集方案（DatasetSynthesisAgent 核心版）

## 背景与目标
- 现有线上真实 query 及其 repo url/branch/commit，并运行着一个多路召回（BM25 + 语义向量检索 + Query 改写）+ rerank 的代码检索系统，需评估代码检索链路本身（命中文档、行号、召回率等）。
- 希望借助 CodeAgent 项目生成评估集，以用于该外部代码检索系统的效果对比与回归测试。
- 目标：构建独立于 CodeAgent 核心的评测数据集流水线，批量回放 query、提取检索证据、生成可复现的 ground truth，并输出检索指标。
- 引入专用的 DatasetSynthesisAgent：复用 CodeAgent 的工具栈，但以“只调工具、不写花哨回复”为原则；为评测合成新增的工具仅负责一次性写出样本 JSON，其余日志/回放由现有 opik 记录承担，保持核心流程最小化。

## 设计原则
- **确定性**：会话运行固定模型、温度、工具集和最大步数；opik 已记录全量请求/响应，本文档聚焦合成所需的关键输入输出。
- **分层复用**：DatasetSynthesisAgent 位于 `runtime/`，只 orchestrate；检索工具沿用现有实现。新增的 `dataset_log_tool` 负责对单条 golden_chunk 做校验并落盘：它使用静态快照（`repo_url + branch + commit_id`）检查文件、行号、内容，每次调用只写入一条 golden_chunk，Agent 按需多次调用累积样本。
- **最小改动**：忽略额外日志、回放管线，先把样本生成链路打通；后续如需更丰富审计，再基于 opik 日志补充。
- **脱敏已完成**：用于合成的数据（query、repo）已完成脱敏/去标识，本文档不再单独处理隐私字段。

> **2025-11-19 更新**：为减小磁盘占用，raw_samples 持久化暂时关闭；`dataset_log_write_chunk` 仅执行快照校验，不再写入 JSONL。下述涉及 raw_samples 的流程可在恢复持久化时重新启用。

## 总体流程
1. **仓库快照 & 索引**
   - 输入：`{repo_url, branch, commit}`。
   - 操作：拉取只读快照（如 `storage/dataset/snapshots/<repo>/<commit>`），校验 commit SHA/GPG、同步子模块，并通过离线索引脚本针对快照构建 Qdrant collection，记录 chunk 数、构建时间与索引参数。
   - 输出：快照目录 + 索引元数据缓存 + `snapshot_metadata.json`（含 repo 授权、rag 配置、磁盘占用）。

2. **Dataset Agent 回放**
   - 输入：query 列表 + 快照路径 + agent 配置。
   - 操作：DatasetRunner 直接配置 `CodeAgentSession`（受限工具、温度 0）并执行 `run_turn`。会话细节由 opik 侧统一记录，数据集流水线仅需关心 agent 成功生成了足够证据。
   - 输出：内存中的 golden_chunks（供下一步使用），无需额外本地日志。

3. **合成样本写出工具**
   - 输入：Agent 每次挑选出的一条 golden_chunk；query_id/query/repo 等静态信息由 orchestrator 注入（快照与 `{repo_url, branch, commit_id}` 一一对应，数据视为静态）。
   - 操作：`dataset_log_tool`（`tools/dataset_log.py`）暴露 `write_chunk(payload)` 并在调用前校验 `path`/行号/内容。目前 raw_samples 持久化已关闭，工具只做校验与回显。
   - 输出：即时返回的校验结果，供 Agent 判断是否继续（无本地 JSONL 累积）。

4. **证据抽取与补全**
   - 输入：`raw_samples/*.jsonl`（当前阶段禁用；若重新打开持久化即可复用）。
   - 操作：extractor 聚合同一 `query_id` 的所有 chunk、去重、可选地补充内容 hash/额外特征，并输出统一格式。
   - 输出：`golden_chunks[]`、`validation_status`（启用落盘后生效）。

5. **样本构建**
   - 输入：query、repo meta（由 orchestrator 注入）、golden_chunks。
   - 操作：生成统一 schema，保留必要版本字段：
     ```json
     {
       "query_id": "...",
       "query": "...",
       "repo": {"url": "...", "commit": "..."},
       "golden_chunks": [...],
       "schema_version": "2025.11"
     }
     ```
   - 输出：评测数据集文件（JSON/Parquet），附 `dataset_version` 及生成日期。

6. **指标与过滤**
   - 当前阶段先不记录/汇总指标；仅执行基础过滤：剔除无命中、写入失败、命中文件过多或 snippet 高重复的样本，并按仓库/语言/任务类型分桶配额。
   - 输出：过滤后的数据集；如需指标再开启后处理计算。

7. **自动化与质量控制**
   - 在 `benchmarks/dataset/` 下实现 orchestrator CLI，串联“快照→回放→抽取→评估”，支持并发 job、断点续跑、单仓库增量刷新，所有输出采用原子写（tmp→rename）。
   - CLI 提供 `--dry-run`、`--resume query_id`、`--refresh-snapshot repo@commit` 等能力；运行时将异常（索引失败、超时）写入 `storage/dataset/<date>/anomalies.jsonl`。
   - 提供抽样审查工具：random spot-check golden_chunks，并允许人工标注“通过/退回”。
   - 每次 pipeline 完成后触发 `uv run pytest tests/benchmarks/test_dataset_pipeline.py -q` 的冒烟测试，确保存储 schema 没被破坏。

## 目录与模块建议
```
benchmarks/dataset/
  snapshot_manager.py       # 拉取仓库 + 索引缓存 + 元数据校验
  runner.py                 # DatasetSynthesisAgent 批量回放
  extractor.py              # raw_samples(jsonl) → golden_chunks（聚合 + 补齐字段，当前未启用）
  dataset_builder.py        # 组合 schema + 版本管理
  metrics.py                # （暂缓）指标计算，如需开启在此实现
agent/prompts/dataset.md    # DatasetSynthesisAgent system prompt
  tools/dataset_log.py        # dataset_log_tool，仅做快照校验（raw_samples 已禁用）
 storage/dataset/<date>/           # 输出数据/异常
```

## DatasetSynthesisAgent & dataset_log_tool
- **Agent 行为**：复用 `CodeAgentSession` 的对话循环（仅配置受限工具/温度 0），只允许调用检索/读文件相关工具，禁止自然语言长回复；每步检查 token 数与工具状态，异常立即中断并打标签。
- **Prompt**：强调“每当检索到能够支撑答案的片段，就立即调用 `dataset_log_write_chunk(payload)` 提交单条 golden_chunk；同一 `query_id` 可多次调用，结束前确保覆盖所有证据”。
- **Tool schema**：
  ```json
  {
    "tool": "dataset_log_write_chunk",
    "arguments": {
      "path": "src/foo.py",
      "start_line": 120,
      "end_line": 135,
      "confidence": 0.9
    }
  }
  ```
  工具在内部合并 orchestrator 注入的 `query_id/query/repo/...`，先校验 path/行号/片段与静态快照一致；raw_samples 落盘暂未开启，但仍需保证单次调用幂等（同一 chunk 重复写入需检测并拒绝）。
- **版本记录**：工具输出字段变化时 bump `schema_version` 并更新 `tests/tools/test_dataset_log.py`。

（opik 已记录完整请求/响应，此处无需另外定义本地日志格式）

## 过滤策略
- 命中数为 0 → 丢弃或标记 `low_confidence`。
- 工具 `status=error` 或超时 → 记入异常集合。
- 命中文件数 > 10 或 chunk 区间高度重复 → 视为噪声。
- 分桶配额：按仓库、语言、任务类型限制样本占比，避免热门仓库垄断。

## 指标（暂缓）
- 当前阶段不输出额外指标，先把数据集合成与基础过滤打通；后续若需要趋势或 diff，再补充统计逻辑。

## 任务拆分

| 阶段 | 任务 | 产出 |
| --- | --- | --- |
| Snapshot & Index | 拉取仓库、构建 Qdrant 索引 | 快照目录 + index metadata |
| Runner | DatasetSynthesisAgent 批量回放脚本 | raw golden_chunks（仅校验，无 `raw_samples`） |
| Evidence Extractor | 解析 raw_samples 补齐 golden_chunks | `raw_samples` 暂未生成 |
| Dataset Builder | 组合 query/meta/golden_chunks → 数据集 | `datasets/rag_gt.jsonl` |
| Metrics & Filter | （暂缓）指标计算、过滤策略实现 | filtered dataset |
| Automation & QC | Orchestrator + 抽样审查工具 | CLI/CI pipeline + QC 报告 |

## 风险与缓解
- **仓库获取失败**：配置镜像、重试、离线缓存。
- **索引耗时**：记录 build_time，支持增量/复用。
- **样本质量参差**：`dataset_log_tool` 在校验阶段即验证文件/行号/内容，落盘恢复后 extractor 再补充聚合信息。
- **指标偏差**：明确定义 schema 与过滤规则，定期人工抽查，迭代阈值。

## 下一步
1. 确定 query 数据源与仓库访问方式，完成快照管理 PoC，并输出 `snapshot_metadata` 模板。
2. 实现 DatasetSynthesisAgent + dataset_log_tool MVP，编写最小集成测试（mock 工具 + 校验输出）。
3. 打通 orchestrator：`uv run python benchmarks/dataset/runner.py --queries demo.jsonl`，当前仅验证 Agent 回放（raw_samples/样本写出仍关闭）。
4. 设计 CI 任务：每日/每周回放随机子集，产出 trend 报告。
