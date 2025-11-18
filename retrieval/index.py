from __future__ import annotations

"""Project indexing/search adapter built on the shared semantic code indexer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from retrieval.codebase_indexer import CodebaseIndex, EmbeddingClient, SemanticCodeIndexer
_PROJECT_ROOTS_BY_KEY: Dict[str, Path] = {}


@dataclass
class Document:
    path: str
    content: str
    chunk_id: Optional[str] = None
    score: float = 0.0
    start_line: int = 0
    end_line: int = 0


class Index:
    """Adapter exposing semantic indexing/search for RAG flows."""

    def __init__(
        self,
        *,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self._indexer = SemanticCodeIndexer(
            embedding_client=embedding_client,
            batch_size=batch_size,
        )
        self._project_paths_by_key: Dict[str, Path] = {}

    def index_project(
        self,
        project_path: str,
        *,
        refresh: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, int | str]:
        root = Path(project_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Project directory not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        index = self._indexer.ensure_index(root, refresh=refresh, show_progress=show_progress)
        self._register_index(index)
        return {
            "project_name": index.project_name,
            "project_path": str(root),
            "collection_name": index.collection_name,
            "chunk_size": self._indexer.chunk_size,
            "project_key": index.project_key,
        }

    def search(
        self,
        project_key: str,
        query: str,
        limit: int = 5,
        *,
        project_path: Optional[str] = None,
        target_directories: Optional[Sequence[str]] = None,
        refresh_index: bool = False,
    ) -> List[Document]:
        root = self._resolve_project_root(project_key, project_path)
        hits, index = self._indexer.search(
            root,
            query,
            limit=max(1, limit),
            target_directories=target_directories,
            refresh_index=refresh_index,
        )
        self._register_index(index)
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

    def _resolve_project_root(
        self,
        project_key: str,
        project_path: Optional[str],
    ) -> Path:
        if project_path:
            root = Path(project_path).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"Project directory not found: {root}")
            if not root.is_dir():
                raise ValueError(f"Path is not a directory: {root}")
            self._project_paths_by_key[project_key] = root
            _PROJECT_ROOTS_BY_KEY[project_key] = root
            return root

        local_key = self._project_paths_by_key.get(project_key)
        if local_key is not None:
            return local_key
        shared_key = _PROJECT_ROOTS_BY_KEY.get(project_key)
        if shared_key is not None:
            self._project_paths_by_key[project_key] = shared_key
            return shared_key
        raise KeyError(f"Project key '{project_key}' has not been indexed yet.")

    def _register_index(self, index: CodebaseIndex) -> None:
        root = index.project_root
        self._project_paths_by_key[index.project_key] = root
        _PROJECT_ROOTS_BY_KEY[index.project_key] = root


def create_index() -> Index:
    return Index()


# Backwards compatibility ------------------------------------------------------
def create_repository() -> Index:
    return create_index()
