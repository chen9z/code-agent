# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Overview

A composable agent runtime focused on code understanding with these key components:
- **Entry Points** (`cli.py`, `code_agent.py`, `codebase_retrieval.py`) – wire the conversation session and retrieval helpers.
- **Agent Layer** (`agent/`) – maintains conversational state (`CodeAgentSession`) and system prompts.
- **Runtime** (`runtime/`) – tool execution runner that orchestrates OpenAI tool calls and concurrency.
- **Retrieval** (`retrieval/`) – indexing/search adapters plus chunking utilities.
- **Adapters** (`adapters/llm`, `adapters/workspace`) – OpenAI/LiteLLM clients, Tree-sitter grammars, and Qdrant vector store helpers.
- **Configuration** (`config/`) – environment-driven settings and prompt fragments.
- **Tools & UI** (`tools/`, `ui/`) – builtin tool implementations and CLI emitters.
- **Testing** (`tests/`) – Pytest suite mirroring the directories above.

Vector data and local artifacts persist in `storage/` (gitignored). Key gitignored items:
- `.venv/` - Python virtual environment
- `storage/` - Vector embeddings and parsed symbols
- `.env` - Environment variables with secrets
- `.pytest_cache/` - Test cache
- `.qdrant/` - Local vector database

### Storage Structure
- `storage/tree_sitter/` - Parsed code symbols in JSONL format
- `storage/` - Vector embeddings and semantic indexes
- Project-specific subdirectories for multi-project support

## Core Architecture

### Flow/Node System
- **BaseNode** - Minimal synchronous node with lifecycle hooks (prep/exec/post)
- **Flow** - Orchestrates node execution with conditional transitions
- **AsyncNode/AsyncFlow** - Asynchronous variants for parallel execution
- **BatchNode/BatchFlow** - Batch processing capabilities

### Key Agent Flows
- **RAGFlow** (`code_rag.py`) - Index/search/query workflows for codebases
- **ToolAgentFlow** (`code_agent.py`) - Tool-based conversational agent with planning/execution/summary nodes
- **CodeAgentSession** - In-memory conversation session for CLI interactions

### Tool System
- **ToolRegistry** - Central registry for available tools with OpenAI-compatible schemas
- **ToolExecutionBatchNode** - Parallel tool execution with ThreadPoolExecutor
- **Built-in Tools** - File operations (read/write/edit), search (grep/glob), bash execution, todo management, codebase search
- **ToolSpec** - Immutable tool metadata with standardized parameter schemas
- **ToolOutput** - Standardized execution results with error handling
- **BaseTool** - Abstract base class requiring `name`, `description`, `parameters`, and `execute()` methods

## Development Commands

### Setup
```bash
uv venv && source .venv/bin/activate
uv sync
uv pip install -e '.[test]'  # optional test dependencies
```

### Testing
```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_rag.py::test_rag_flow -q  # Run specific test
uv run pytest --maxfail=1 -q    # Fast failure mode
```

### Running the Agent
```bash
uv run python main.py                  # Start agent in current directory
uv run python main.py -w /path/to/repo  # Target different workspace
uv run python main.py -p "List TODOs"   # Run single prompt then exit
```

### Benchmarking
```bash
uv run python benchmarks/code_agent_benchmark.py \
  --config benchmarks/examples/embedding_models.json \
  --transcript-dir benchmarks/logs \
  --output benchmarks/results/latest.json
```

## Key Environment Variables
```bash
export LLM_MODEL=deepseek-chat
export LLM_API_KEY=sk-...
export LLM_API_BASE=https://api.deepseek.com
export OPENAI_API_KEY=sk-...            # compatibility fallback
export OPENAI_API_BASE=https://api.openai.com/v1
export CHAT_CODEBASE_PATH=/path/to/chat-codebase  # optional deep integration

# RAG Configuration
export RAG_EMBEDDING_MODEL=openai-like
export RAG_RERANK_MODEL=api
export RAG_CHUNK_SIZE=200

# LLM Configuration
export LLM_TEMPERATURE=0.1
export LLM_MAX_TOKENS=2000
export OPIK_PROJECT_NAME=your-project  # optional observability
export OPIK_ENABLED=true

# Embedding Configuration
export EMBEDDING_API_BASE=http://127.0.0.1:8000/v1  # default local endpoint
```

## Key Dependencies
- **langchain** - LLM orchestration and tool abstractions
- **httpx** - HTTP client used for embedding requests
- **qdrant-client** - Local vector database for semantic search
- **tree-sitter** - Code parsing with language grammars
- **diskcache** - Persistent caching for parsed symbols
- **grep-ast** - Language detection and file filtering
- **rich** - Rich console output for CLI interface

## Python API Examples

### Code RAG Workflow
```python
from code_rag import run_rag_workflow

# Index a project
run_rag_workflow(action="index", project_path="/path/to/project")

# Search code
res = run_rag_workflow(action="search", project_name="project", query="function definition", limit=5)

# Query the agent
res = run_rag_workflow(action="query", project_name="project", question="How does it work?", limit=5)
```

### Tool Agent Session
```python
from code_agent import CodeAgentSession

session = CodeAgentSession(max_iterations=100)
result = session.run_turn("Find all TODO comments in the codebase")
print(result["final_response"])
```

## Development Guidelines

### Code Organization
- Runtime primitives exported via top-level `__init__`
- Agent modules provide ready-to-run flows with documented entrypoints
- Integrations use dependency injection for testability
- Configuration managed through `config.config.get_config()` helpers

### Testing Strategy
- Pytest with fixtures in `tests/conftest.py`
- Tests mirror runtime package structure
- Mock external services for unit testing
- Focus on node/flow unit tests and integration tests

### Tool Development
- Extend `tools.base.BaseTool` for new tools
- Register tools via `tools.registry.ToolRegistry`
- Tools support parallel execution and error handling
- Include comprehensive tests for new tools

### Python Version
- Requires Python 3.12+ (specified in `.python-version`)
- Uses `uv` for dependency management and virtual environments

### Code Patterns
- Extensive use of `@dataclass` for immutable data structures
- `@lru_cache` for configuration and client caching
- Type hints throughout for better IDE support and maintainability
- Dependency injection patterns for testability
- Async/await support via `AsyncNode` and `AsyncFlow` classes
- ThreadPoolExecutor for parallel tool execution

## Integration Points

### Tree-sitter Integration
- `adapters/workspace/tree_sitter/` - Code parsing and symbol extraction
- Supports multiple programming languages via tree-sitter grammars
- Uses `grep_ast` for language detection and file filtering
- Extracts symbols with `TagKind.DEF` (definitions) and `TagKind.REF` (references)
- Caches parsed results using `diskcache` for performance
- Exports parsed symbols to JSONL format

### Vector Store Integration
- `adapters/workspace/vector_store.py` - Embedding storage and retrieval using Qdrant
- `retrieval/index.py` - Project indexing and search
- `retrieval/codebase_indexer.py` - Semantic code indexing with embeddings
- Uses `httpx` to call OpenAI-compatible embedding endpoints
- Local Qdrant storage with configurable collection names and vector sizes
- Configurable embedding models via environment variables

### LLM Client Integration
- `adapters/llm/llm.py` - Unified LLM client interface
- Supports OpenAI-compatible APIs
- Fallback stub LLM for testing
- Optional Opik observability integration

### System Prompts
- `config/prompt.py` - Centralized prompt management
- Security constraints for defensive operations only
- Concise response guidelines to minimize token usage
- Context-aware conversation history management

## Benchmarking Framework

### Scenario Configuration
- JSON-based scenario definitions
- Validators for automated result checking
- Transcript logging for debugging
- CI/CD integration support

### Validation Types
- `contains` - String matching in final response
- `regex` - Regular expression matching
- `regex_set` - Set-based pattern matching with ignore lists
- `transcript_contains` - Log-level message matching

## Development Workflow

### Debugging Agent Sessions
- Use `--transcript-dir` in benchmarks to capture detailed execution logs
- Check `storage/tree_sitter/` for parsed symbol exports
- Monitor tool execution via rich console output in CLI mode

### Adding New Tools
1. Create tool class extending `tools.base.BaseTool`
2. Implement `execute()` method with proper error handling
3. Define `name`, `description`, and `parameters` schema
4. Register in `tools.registry.create_default_registry()`
5. Add comprehensive tests in `tests/test_*_tool.py`

### Extending Flows
- Compose new flows by wiring existing nodes
- Use conditional transitions (`node.next(other_node, "action")`)
- Implement custom nodes by extending `Node` or `AsyncNode`
- Test flows with focused pytest cases

### Performance Optimization
- Use `diskcache` for expensive tree-sitter parsing operations
- Configure `max_parallel_workers` for tool execution throughput
- Set appropriate `max_iterations` to prevent infinite loops
- Use `ThreadPoolExecutor` for parallel tool execution
- Cache configuration with `@lru_cache` to avoid repeated environment lookups
