from __future__ import annotations

"""Semantic code search tool backed by a shared embedding indexer."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from retrieval.codebase_indexer import EmbeddingClient, SemanticCodeIndexer
from runtime.tool_types import ToolResult
from tools.base import BaseTool


class CodebaseSearchArgs(BaseModel):
    query: str = Field(..., description="The natural language search query to run against the semantic index.")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of matches to return (1-20).",
    )
    project_root: Optional[str | Path] = Field(
        default=None,
        description="Optional project root override. Defaults to the configured workspace root.",
    )
    refresh_index: Optional[bool] = Field(
        default=None,
        description="Force a refresh of the semantic index before searching.",
    )
    target_directories: Optional[Sequence[str]] = Field(
        default=None,
        description="Restrict the search to the provided glob directory patterns.",
    )

    model_config = ConfigDict(extra="forbid")


class CodebaseSearchTool(BaseTool):
    """Semantic search across the local repository using cached embeddings."""

    ArgumentsModel = CodebaseSearchArgs

    def __init__(
        self,
        *,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
        semantic_indexer: Optional[SemanticCodeIndexer] = None,
        default_project_root: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self._default_root = (
            Path(default_project_root).expanduser().resolve()
            if default_project_root
            else None
        )
        if semantic_indexer is not None:
            self._indexer = semantic_indexer
        else:
            self._indexer = SemanticCodeIndexer(
                embedding_client=embedding_client,
                batch_size=batch_size,
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

    # Execution --------------------------------------------------------------------
    def execute(
        self,
        *,
        query: str,
        limit: int | None = None,
        project_root: Optional[str] = None,
        refresh_index: Optional[bool] = None,
        target_directories: Optional[Sequence[str]] = None,
    ) -> ToolResult:
        if not query or not isinstance(query, str) or not query.strip():
            message = "query must be a non-empty string"
            return _error_response(message, {"query": query})

        limit_val = limit if limit is not None else 5

        candidate_root: Optional[str | Path]
        if project_root:
            candidate_root = project_root
        elif self._default_root is not None:
            candidate_root = self._default_root
        else:
            candidate_root = Path.cwd()

        root = Path(candidate_root).expanduser().resolve()
        if not root.exists():
            message = f"project_root does not exist: {root}"
            return _error_response(message)
        if not root.is_dir():
            message = f"project_root is not a directory: {root}"
            return _error_response(message)

        try:
            hits, index = self._indexer.search(
                root,
                query,
                limit=limit_val,
                target_directories=target_directories,
                refresh_index=bool(refresh_index),
            )
        except Exception as exc:  # pragma: no cover - 网络/缓存异常
            message = f"Failed to execute semantic search: {exc}"
            return _error_response(message)

        if not hits:
            data = {
                "query": query,
                "project_root": str(root),
                "project_name": index.project_name,
                "results": [],
                "count": 0,
                "index": self._indexer.index_metadata(index),
                "target_directories": list(target_directories or []),
            }
            return _success_response("[no semantic matches]", data)

        # Group hits by file path
        from itertools import groupby
        
        # Group hits by file path, but we want to present files in order of relevance (best score first).
        # 1. Group hits by file
        hits_by_file = {}
        for hit in hits:
            path = hit.chunk.relative_path
            if path not in hits_by_file:
                hits_by_file[path] = []
            hits_by_file[path].append(hit)
            
        # 2. Determine file order by the maximum score of any hit in that file
        # (Assuming 'hits' is already sorted by score descending from the searcher? Yes, usually.)
        # But let's be safe and compute max score per file.
        file_scores = []
        for path, file_hits in hits_by_file.items():
            max_score = max(h.score for h in file_hits)
            file_scores.append((path, max_score))
            
        # Sort files by max score descending
        file_scores.sort(key=lambda x: x[1], reverse=True)
        
        formatted_results = []
        raw_results = []
        
        for relative_path, _ in file_scores:
            file_hits = hits_by_file[relative_path]
            
            # Sort hits within the file by line number for display
            file_hits.sort(key=lambda h: h.chunk.start_line)
            if not file_hits:
                continue
                
            # Merge logic
            file_content_parts = []
            last_end_line = -1
            
            file_content_parts.append(f"## File: {relative_path}")
            
            for hit in file_hits:
                chunk = hit.chunk
                
                # Add gap if needed
                if last_end_line != -1 and chunk.start_line > last_end_line + 1:
                    file_content_parts.append("\n# ...\n")
                elif last_end_line != -1:
                    # Ensure newline between adjacent chunks if not already present
                    file_content_parts.append("\n")
                
                # Add context annotation
                context_list = chunk.metadata.get("context")
                if context_list and isinstance(context_list, list):
                    context_str = " > ".join(context_list)
                    file_content_parts.append(f"# Context: {context_str}")
                
                # Add chunk content
                file_content_parts.append(chunk.content)
                
                last_end_line = chunk.end_line
                
                # Collect raw result for data payload
                raw_results.append({
                    "path": chunk.relative_path,
                    "absolute_path": chunk.absolute_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "score": hit.score,
                    "language": chunk.language,
                    "snippet": chunk.content,
                    "context": context_list,  # Pass context to model data
                })
            
            formatted_results.append("\n".join(file_content_parts))

        final_content = "\n\n".join(formatted_results)
        
        data = {
            "query": query,
            "project_root": str(root),
            "project_name": index.project_name,
            "results": raw_results,
            "count": len(raw_results),
            "index": self._indexer.index_metadata(index),
            "target_directories": list(target_directories or []),
        }
        
        return _success_response(final_content, data)


def _error_display(message: str) -> str:
    text = str(message or "").strip()
    return text or "Unknown error"


def _error_response(message: str, extra: Optional[Dict[str, Any]] = None) -> ToolResult:
    payload = dict(extra or {})
    payload["error"] = message
    payload["display"] = _error_display(message)
    return ToolResult(status="error", content=message, data=payload)


def _success_response(content: str, data: Dict[str, Any]) -> ToolResult:
    return ToolResult(status="success", content=content, data=data)
