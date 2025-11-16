# CodeAgent RAG 检索评测 Harness 方案

## 背景与目标
- 现有线上真实 query 及其 repo url/branch/commit，并运行着一个多路召回（BM25 + 语义向量检索 + Query 改写）+ rerank 的代码检索系统，需评估代码检索链路本身（命中文档、行号、召回率等）。
- 希望借助 CodeAgent 项目生成评估集，以用于该外部代码检索系统的效果对比与回归测试。
- 目标：构建独立于 CodeAgent 核心的外部 Harness，批量回放 query、提取检索证据、生成可复现的 ground truth，并输出检索指标。

## 总体流程
1. **仓库快照 & 索引**
   - 输入：`{repo_url, branch, commit}`。
   - 操作：拉取只读快照（如 `storage/harness/<repo>/<commit>`），运行 `RepositoryAdapter.index_project` 预建 Qdrant collection，记录 chunk 数/构建时间。
   - 输出：快照目录 + 索引元数据缓存。

2. **回放执行**
   - 输入：query 列表 + 快照路径。
   - 操作：初始化 `CodeAgentSession(workspace=快照)`，调用 `run_turn`，记录 `messages`、`tool_results`、耗时、状态。
   - 输出：结构化日志 JSONL（含 `query_id`、tool 调用详情、耗时等）。

3. **证据抽取**
   - 输入：`tool_results` 日志、索引数据。
   - 操作：筛选检索类工具（Read/Grep/Glob），抽取 `{path, start_line, end_line, snippet, tool_call_id}`；可与 `RepositoryAdapter.search` top-k 对比。
   - 输出：`evidence[]`、`retrieved_chunks[]`，附命中数、命中率等特征。

4. **样本构建**
   - 输入：query、repo meta、evidence、retrieved_chunks。
   - 操作：生成统一 schema：
     ```json
     {
       "query_id": "...",
       "query": "...",
       "repo": {"url": "...", "commit": "..."},
       "evidence": [...],
       "retrieved_chunks": [...],
       "metrics": {...},
       "logs_ref": "..."
     }
     ```
   - 输出：评测数据集文件（JSON/Parquet）。

5. **指标与过滤**
   - 计算 Recall@K、Jaccard（evidence 文件集合 vs top-k）、平均命中文件数等。
   - 应用过滤规则：剔除无命中、执行失败、命中文件过多或重复度高的样本；按仓库/语言/任务类型分桶抽样。
   - 输出：最终数据集 + 指标报告。

6. **自动化与质量控制**
   - 在 `benchmarks/harness/` 下实现 orchestrator CLI，串联“快照→回放→抽取→评估”，支持批处理/增量更新。
   - 提供抽样审查工具：随机展示 evidence 供人工 spot-check。
   - 记录异常（索引失败、超时），避免污染数据集。

## 目录与模块建议
```
benchmarks/harness/
  snapshot_manager.py    # 拉取仓库 + 索引缓存
  runner.py              # CodeAgentSession 批量回放
  extractor.py           # tool_results → evidence
  dataset_builder.py     # 组合 schema
  metrics.py             # 指标计算
  cli.py                 # 主 orchestrator
artifacts/<date>/        # 输出数据/日志
```

## 日志格式示例
```json
{
  "query_id": "uuid",
  "query": "How to ...",
  "repo": {"url": "...", "commit": "..."},
  "session_config": {"model": "...", "max_iterations": 25},
  "tool_results": [
    {"tool": "grep", "status": "success", "arguments": {...}, "data": {"matches": [...]}, "duration_ms": 120},
    ...
  ],
  "elapsed_ms": 9500,
  "status": "ok"
}
```

## 过滤策略
- 命中数为 0 → 丢弃或标记 `low_confidence`。
- 工具 `status=error` 或超时 → 记入异常集合。
- 命中文件数 > 10 或 snippet 高重复 → 视为噪声。
- 分桶配额：按仓库、语言、任务类型限制样本占比，避免热门仓库垄断。

## 指标
- **Recall@K**：evidence 与 top-k 检索结果重叠率。
- **Jaccard**：evidence 文件集合 vs top-k 文件集合。
- **Avg Hits**：平均命中文件/行段数量。
- 可选：执行耗时分布、工具调用次数、检索成功率等。

## 任务拆分

| 阶段 | 任务 | 产出 |
| --- | --- | --- |
| Snapshot & Index | 拉取仓库、构建 Qdrant 索引 | 快照目录 + index metadata |
| Runner | CodeAgentSession 批量回放脚本 | `logs/session_run.jsonl` |
| Evidence Extractor | 解析日志生成 evidence/retrieved chunks | `evidence/*.json` |
| Dataset Builder | 组合 query/meta/evidence → 数据集 | `datasets/rag_gt.jsonl` |
| Metrics & Filter | 指标计算、过滤策略实现 | 指标报表 + filtered dataset |
| Automation & QC | Orchestrator + 抽样审查工具 | CLI/CI pipeline + QC 报告 |

## 风险与缓解
- **仓库获取失败**：配置镜像、重试、离线缓存。
- **索引耗时**：记录 build_time，支持增量/复用。
- **日志体积大**：JSONL 分片 + 压缩；按 query_id 存档。
- **指标偏差**：明确定义 schema 与过滤规则，定期人工抽查，迭代阈值。

## 下一步
1. 确定 query 数据源与仓库访问方式，完成快照管理 PoC。
2. 开发 runner & extractor 的最小实现，验证日志结构与 evidence 抽取可行。
3. 搭建初版 orchestrator，跑一小批样本，生成原始指标并审查质量。
