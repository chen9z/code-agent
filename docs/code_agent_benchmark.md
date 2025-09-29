# Code Agent Benchmark

`benchmarks/code_agent_benchmark.py` 提供了一个可扩展的基准测试框架，用于针对多组仓库/Prompt 场景衡量 Code Agent 的准确率与效率。

## 配置格式

配置文件为 JSON，顶层包含 `scenarios` 数组。每个场景支持以下字段：

| 字段 | 说明 | 必填 |
| --- | --- | --- |
| `name` | 场景名称（用于输出和日志文件命名） | 否 |
| `workspace` | 需要运行的仓库目录（绝对路径或相对路径） | 是 |
| `prompt` | 提供给代理的指令 | 是 |
| `max_iterations` | 单次调用允许的最大循环次数，默认使用 CLI 传入的 `--max-iterations` | 否 |
| `validators` | 校验规则数组，决定最终通过/失败 | 否 |

### 校验器类型

- `contains`：确保最终回答中包含给定字符串。
- `regex`：使用正则表达式在最终回答中匹配文本。
- `regex_set`：从最终回答中提取一组匹配项，并与期望集合比较。支持 `ignore` 列表用于忽略不相关的匹配。
- `transcript_contains`：从代理输出的逐行日志中匹配关键消息。

每个校验器可定义 `name` 字段，自定义失败时的标识。脚本会在结果中记录缺失/多余项等细节，便于回归分析。

## 示例

`benchmarks/examples/embedding_models.json` 演示了如何验证 Spring AI 仓库中所有 `EmbeddingModel` 实现均被正确枚举：

```json
{
  "scenarios": [
    {
      "name": "spring_ai_embedding_models",
      "workspace": "/Users/looper/workspace/spring-ai",
      "prompt": "找出项目使用的所有的 Embedding 模型",
      "max_iterations": 25,
      "validators": [
        {
          "type": "regex_set",
          "name": "embedding_model_set",
          "pattern": "\\\b([A-Za-z]+EmbeddingModel)\\\b",
          "expected": [
            "AzureOpenAiEmbeddingModel",
            "BedrockCohereEmbeddingModel",
            "BedrockTitanEmbeddingModel",
            "MiniMaxEmbeddingModel",
            "MistralAiEmbeddingModel",
            "OCIEmbeddingModel",
            "OllamaEmbeddingModel",
            "OpenAiEmbeddingModel",
            "PostgresMlEmbeddingModel",
            "TransformersEmbeddingModel",
            "VertexAiMultimodalEmbeddingModel",
            "VertexAiTextEmbeddingModel",
            "ZhiPuAiEmbeddingModel"
          ],
          "ignore": [
            "AbstractEmbeddingModel",
            "DocumentEmbeddingModel",
            "TitanEmbeddingModel"
          ]
        }
      ]
    }
  ]
}
```

## 运行方式

```bash
uv run python benchmarks/code_agent_benchmark.py \
  --config benchmarks/examples/embedding_models.json \
  --transcript-dir benchmarks/logs \
  --output benchmarks/results/latest.json
```

脚本会输出：

- 每个场景的耗时、工具调用次数、使用的工具列表
- 校验器逐项通过/失败详情
- 可选的原始 transcript（按行记录 `[planner]`、`[tool]`、`[assistant]` 等日志）

全部场景通过时退出码为 0，否则为 1，方便集成至 CI/CD 或长期回归测试中。
