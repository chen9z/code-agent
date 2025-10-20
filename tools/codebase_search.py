from __future__ import annotations

"""Semantic code search tool backed by a shared embedding indexer."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from integrations.codebase_indexer import EmbeddingClient, SemanticCodeIndexer
from tools.base import BaseTool


class CodebaseSearchTool(BaseTool):
    """Semantic search across the local repository using cached embeddings."""

    def __init__(
        self,
        *,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
        max_snippet_chars: int = 800,
        semantic_indexer: Optional[SemanticCodeIndexer] = None,
    ) -> None:
        super().__init__()
        if semantic_indexer is not None:
            self._indexer = semantic_indexer
        else:
            self._indexer = SemanticCodeIndexer(
                embedding_client=embedding_client,
                batch_size=batch_size,
                max_snippet_chars=max_snippet_chars,
            )

    # Metadata ---------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "codebase_search"

    @property
    def description(self) -> str:
        return '''Find snippets of code from the codebase most relevant to the search query.
This is a semantic search tool, so the query should ask for something semantically matching what is needed.
If it makes sense to only search in particular directories, please specify them in the target_directories field.
Unless there is a clear reason to use your own search query, please just reuse the user's exact query with their wording.
Their exact wording/phrasing can often be helpful for the semantic search query. Keeping the same exact question format can also be helpful.'''

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "properties": {
                "query": {
                    "description": "The search query to find relevant code. You should reuse the user's exact query/most recent message with their wording unless there is a clear reason not to.",
                    "type": "string",
                },
                "target_directories": {
                    "description": "Glob patterns for directories to search over",
                    "items": {"type": "string"},
                    "type": "array",
                },
            },
            "required": ["query"],
            "type": "object",
        }

    # Execution --------------------------------------------------------------------
    def execute(
        self,
        *,
        query: str,
        limit: float | int | None = None,
        project_root: Optional[str] = None,
        refresh_index: Optional[bool] = None,
        target_directories: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not query or not isinstance(query, str) or not query.strip():
            return {"error": "query must be a non-empty string", "query": query}

        try:
            limit_val = int(limit) if limit is not None else 5
        except (TypeError, ValueError):
            return {"error": "limit must be an integer", "limit": limit}
        limit_val = max(1, min(limit_val, 20))

        root = Path(project_root).expanduser().resolve() if project_root else Path.cwd()
        if not root.exists():
            return {"error": f"project_root does not exist: {root}"}
        if not root.is_dir():
            return {"error": f"project_root is not a directory: {root}"}

        try:
            hits, index = self._indexer.search(
                root,
                query,
                limit=limit_val,
                target_directories=target_directories,
                refresh_index=bool(refresh_index),
            )
        except Exception as exc:  # pragma: no cover - 网络/缓存异常
            return {"error": f"Failed to execute semantic search: {exc}"}

        if not hits:
            return {
                "status": "success",
                "query": query,
                "project_root": str(root),
                "project_name": index.project_name,
                "results": [],
                "count": 0,
                "result": "[no semantic matches]",
                "index": self._indexer.index_metadata(index),
                "target_directories": list(target_directories or []),
            }

        formatted = self._indexer.format_hits(hits)
        return {
            "status": "success",
            "query": query,
            "project_root": str(root),
            "project_name": index.project_name,
            "count": len(formatted["results"]),
            "result": formatted["summary"] or "[no semantic matches]",
            "index": self._indexer.index_metadata(index),
            "target_directories": list(target_directories or []),
        }
