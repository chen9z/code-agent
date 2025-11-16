# Code Agent

Composable agent runtime focused on code understanding. Index repositories, run semantic search, and answer questions with retrieved context. Ships with a fallback stub LLM and optional OpenAI-compatible clients.

## Project Layout

- Root entrypoints (`cli.py`, `code_agent.py`, `codebase_retrieval.py`) orchestrate the runtime.
- `agent/` – conversation/session orchestration and system prompt assembly.
- `config/` – environment-driven settings and prompt fragments.
- `retrieval/` – indexing/search pipeline plus file chunking helpers.
- `adapters/` – `llm/` for model clients, `workspace/` for tree-sitter + vector store plumbing.
- `runtime/` – tool execution runner and concurrency helpers.
- `tools/` / `ui/` – builtin tool implementations and CLI output emitters.
- `tests/` – pytest suite mirroring the directories above.

Vector data persists in `storage/` (gitignored).

## Quick Start

Prerequisites: Python ≥ 3.11 and `uv`.

```bash
uv venv && source .venv/bin/activate
uv sync
# optional extras
uv pip install -e '.[test]'
```

Optional configuration (env vars consumed via `config.config`):

```bash
export LLM_MODEL=deepseek-chat
export LLM_API_KEY=sk-...
export LLM_API_BASE=https://api.deepseek.com
export OPENAI_API_KEY=sk-...            # compatibility fallback
export OPENAI_API_BASE=https://api.openai.com/v1
export CHAT_CODEBASE_PATH=/path/to/chat-codebase  # optional deep integration
export CLI_TOOL_TIMEOUT_SECONDS=120     # optional: override default tool timeout in seconds
```

## Python API

```python
from retrieval.index import create_index

index = create_index()
index_info = index.index_project("/path/to/project")
hits = index.search(index_info["project_name"], "function definition", limit=5)

print(index_info["chunk_count"], index_info["chunk_size"], len(hits))
```

### Chunking & Retrieval Flow
- Files are tokenized by `SemanticCodeIndexer` and split into ~`chunk_size` lines (default 200) with symbol-aware boundaries provided by Tree-sitter.
- Each chunk is embedded via the configured model and stored in a local Qdrant collection keyed by project name under `storage/`.
- Index responses now surface `chunk_count`, `file_count`, and `chunk_size` so you can verify coverage and tune chunking upfront.
- Search embeds the query once, filters hits by optional directory patterns, and returns scored chunks with file paths and line spans for downstream tooling.

## Agent CLI

Launch the code agent from the terminal and interact conversationally:

```bash
uv run python cli.py                  # starts the agent in the current directory
uv run python cli.py -w /path/to/repo  # target a different workspace
uv run python cli.py -p "List TODOs"   # run a single prompt then exit
uv run python cli.py --tool-timeout 180  # allow tools up to 3 minutes by default
```

Type your prompts when the CLI shows `You: ` and enter `exit` (or `quit`) to finish. The agent streams planning thoughts, tool activity, and final answers inline.

Notes (concise): the agent uses native tool-calling; if multiple tools are planned in a turn they execute in parallel, and outputs are shown in the original plan order.
Long Bash/Glob logs are truncated for readability. When you see `preview truncated` or `... (output truncated)` the console has already shown as much as it will retain.

## Development

- Run tests: `uv run pytest`
- Focused retrieval test: `uv run pytest tests/test_codebase_retrieval.py -q`
- Agent CLI smoke test: `uv run python cli.py`
- Benchmark scenarios: `uv run python benchmarks/code_agent_benchmark.py --config benchmarks/examples/embedding_models.json --transcript-dir benchmarks/logs --output benchmarks/results/latest.json`

Refer to `AGENTS.md` for contributor guidance and coding conventions.
See `docs/code_agent_benchmark.md` for benchmark configuration details.
