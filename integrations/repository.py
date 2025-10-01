from __future__ import annotations

"""Repository adapter built on the shared semantic code indexer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from diskcache import Cache

from integrations.codebase_indexer import EmbeddingClient, SemanticCodeIndexer
_PROJECT_ROOTS: Dict[str, Path] = {}


@dataclass
class Document:
    path: str
    content: str
    chunk_id: Optional[str] = None
    score: float = 0.0
    start_line: int = 0
    end_line: int = 0


class RepositoryAdapter:
    """Adapter exposing semantic indexing/search for RAG flows."""

    def __init__(
        self,
        *,
        cache: Optional[Cache] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self._indexer = SemanticCodeIndexer(
            cache=cache,
            embedding_client=embedding_client,
            batch_size=batch_size,
        )
        self._project_paths: Dict[str, Path] = {}

    def index_project(self, project_path: str) -> Dict[str, int | str]:
        root = Path(project_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Project directory not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        index = self._indexer.ensure_index(root, refresh=True)
        self._project_paths[index.project_name] = root
        _PROJECT_ROOTS[index.project_name] = root
        return {
            "project_name": index.project_name,
            "project_path": str(root),
            "total_chunks": index.chunk_count,
        }

    def search(
        self,
        project_name: str,
        query: str,
        limit: int = 5,
        *,
        target_directories: Optional[Sequence[str]] = None,
        refresh_index: bool = False,
    ) -> List[Document]:
        root = self._resolve_project_root(project_name)
        hits, _ = self._indexer.search(
            root,
            query,
            limit=max(1, limit),
            target_directories=target_directories,
            refresh_index=refresh_index,
        )
        documents: List[Document] = []
        for idx, hit in enumerate(hits):
            chunk = hit.chunk
            documents.append(
                Document(
                    path=chunk.relative_path,
                    content=chunk.content,
                    chunk_id=f"{chunk.relative_path}:{idx}",
                    score=hit.score,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
            )
        return documents

    def format_search_results(self, documents: List[Document]) -> str:
        if not documents:
            return "No results found."

        formatted: List[str] = []
        for i, doc in enumerate(documents, 1):
            line = f"{i}. {doc.path}"
            if doc.score:
                line += f" (score: {doc.score:.3f})"
            if doc.start_line:
                line += f" lines {doc.start_line}-{doc.end_line}"
            formatted.append(line)
            preview = (doc.content[:200].replace("\n", " ") + ("..." if len(doc.content) > 200 else ""))
            formatted.append(f"   {preview}")
            formatted.append("")
        return "\n".join(formatted).rstrip()

    def _resolve_project_root(self, project_name: str) -> Path:
        local = self._project_paths.get(project_name)
        if local is not None:
            return local
        shared = _PROJECT_ROOTS.get(project_name)
        if shared is not None:
            self._project_paths[project_name] = shared
            return shared
        raise KeyError(f"Project '{project_name}' has not been indexed yet.")


def create_repository() -> RepositoryAdapter:
    return RepositoryAdapter()
