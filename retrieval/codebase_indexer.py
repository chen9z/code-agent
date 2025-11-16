from __future__ import annotations

"""Shared semantic code indexing and search utilities for tools and flows."""

import hashlib
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config.config import get_config
from retrieval.splitter import chunk_code_file, iter_repository_files
from adapters.llm.embedding import (
    EmbeddingClientProtocol,
    HttpxEmbeddingClient,
    create_embedding_client,
)
from adapters.workspace.tree_sitter.parser import TagKind, TreeSitterProjectParser
from adapters.workspace.vector_store import LocalQdrantStore, QdrantConfig, QdrantPoint

EmbeddingClient = HttpxEmbeddingClient


def _project_identifier(root: Path) -> str:
    digest = hashlib.sha1(str(root).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


_COLLECTION_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")


def _collection_name(project_name: str, project_key: str) -> str:
    """Derive a deterministic, unique Qdrant collection name."""

    base = project_name or "project"
    safe = _COLLECTION_SANITIZE_RE.sub("_", base).strip("_") or "project"
    return f"{safe}_{project_key}"


@dataclass(frozen=True)
class CodeChunkEmbedding:
    relative_path: str
    absolute_path: str
    start_line: int
    end_line: int
    language: Optional[str]
    symbol: Optional[str]
    content: str
    vector: Tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CodebaseIndex:
    project_root: Path
    project_key: str
    project_name: str
    collection_name: str
    entries: Tuple[CodeChunkEmbedding, ...]
    file_count: int
    chunk_count: int
    indexed_at: datetime
    build_time_seconds: float


@dataclass(frozen=True)
class SemanticSearchHit:
    score: float
    chunk: CodeChunkEmbedding


class SemanticCodeIndexer:
    """集中式语义索引构建与检索实现。"""

    def __init__(
        self,
        *,
        embedding_client: Optional[EmbeddingClientProtocol] = None,
        batch_size: Optional[int] = None,
        vector_store: Optional[LocalQdrantStore] = None,
        qdrant_config: Optional[QdrantConfig] = None,
    ) -> None:
        cfg = get_config()
        rag_cfg = getattr(cfg, "rag", None)
        chunk_size = 200
        if rag_cfg is not None:
            chunk_size = max(32, int(getattr(rag_cfg, "chunk_size", 200)))

        batch = int(batch_size or os.getenv("CODEBASE_EMBEDDING_BATCH", "16"))
        api_timeout = float(os.getenv("CODEBASE_EMBEDDING_TIMEOUT", "120"))

        store_path = os.getenv("CODEBASE_QDRANT_PATH", "storage")
        store_root = Path(store_path).expanduser()
        store_root.mkdir(parents=True, exist_ok=True)
        config = qdrant_config or QdrantConfig(path=str(store_root))
        self._store = vector_store or LocalQdrantStore(config)
        self._embedder: EmbeddingClientProtocol = embedding_client or create_embedding_client(
            batch_size=batch,
            timeout=api_timeout,
        )
        self._chunk_size = chunk_size
        self._batch_size = max(1, batch)
        self._indices: Dict[str, CodebaseIndex] = {}
        self._lock = threading.Lock()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def ensure_index(
        self,
        project_root: Path | str,
        *,
        refresh: bool = False,
        show_progress: bool = False,
    ) -> CodebaseIndex:
        root = Path(project_root).expanduser().resolve()
        key = str(root)
        project_key = _project_identifier(root)
        collection_name = _collection_name(root.name, project_key)
        with self._lock:
            self._store.use_collection(collection_name)
            cached = self._indices.get(key)
            collection_missing = not self._store.collection_exists(collection_name)
            if not refresh and cached is not None and not collection_missing:
                return cached
            index = self._build_index(
                root,
                project_key=project_key,
                collection_name=collection_name,
                show_progress=show_progress,
            )
            self._indices[key] = index
            return index

    def search(
        self,
        project_root: Path | str,
        query: str,
        *,
        limit: int = 5,
        target_directories: Optional[Sequence[str]] = None,
        refresh_index: bool = False,
    ) -> Tuple[List[SemanticSearchHit], CodebaseIndex]:
        index = self.ensure_index(project_root, refresh=refresh_index)
        self._store.use_collection(index.collection_name)
        embeddings = self._embedder.embed_batch([query])
        if not embeddings:
            return [], index
        query_vector = tuple(float(v) for v in embeddings[0])

        fetch_limit = max(limit, 1)
        scored_points = self._store.search(
            vector=query_vector,
            project_key=index.project_key,
            limit=fetch_limit,
        )

        pattern_matcher = self._compile_directory_matcher(target_directories)
        hits: List[SemanticSearchHit] = []
        for point in scored_points:
            payload = point.payload or {}
            relative_path = payload.get("relative_path")
            if pattern_matcher and (not isinstance(relative_path, str) or not pattern_matcher(relative_path)):
                continue
            chunk = CodeChunkEmbedding(
                relative_path=str(relative_path or ""),
                absolute_path=str(payload.get("absolute_path") or ""),
                start_line=int(payload.get("start_line") or 0),
                end_line=int(payload.get("end_line") or 0),
                language=payload.get("language"),
                symbol=payload.get("symbol"),
                content=str(payload.get("snippet") or ""),
                vector=(),
            )
            hits.append(
                SemanticSearchHit(
                    score=float(point.score or 0.0),
                    chunk=chunk,
                )
            )
            if len(hits) >= limit:
                break
        return hits, index

    def close(self) -> None:
        try:
            self._store.close()
        except Exception:
            pass

    def format_hits(self, hits: Sequence[SemanticSearchHit]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        summary_blocks: List[str] = []

        hits_by_path: Dict[str, List[SemanticSearchHit]] = {}
        for hit in hits:
            hits_by_path.setdefault(hit.chunk.relative_path, []).append(hit)

        for path, path_hits in hits_by_path.items():
            path_hits.sort(key=lambda h: (h.chunk.start_line, h.chunk.end_line))
            top_hit = path_hits[0].chunk
            symbol = top_hit.symbol or "(anonymous)"
            preview_lines = top_hit.content.strip().splitlines()[:5]
            preview = " ".join(line.strip() for line in preview_lines)
            if len(top_hit.content.splitlines()) > 5:
                preview += " ..."
            summary_blocks.append(
                f"{path}:{top_hit.start_line}-{top_hit.end_line} [{symbol}] {preview}"
            )

        for hit in hits:
            entry = hit.chunk
            results.append(
                {
                    "path": entry.relative_path,
                    "absolute_path": entry.absolute_path,
                    "start_line": entry.start_line,
                    "end_line": entry.end_line,
                    "score": hit.score,
                    "language": entry.language,
                    "symbol": entry.symbol,
                    "snippet": entry.content,
                }
            )
        summary = "\n".join(f"- {block}" for block in summary_blocks)
        return {"results": results, "summary": summary}

    @staticmethod
    def index_metadata(index: CodebaseIndex) -> Dict[str, Any]:
        return {
            "project_key": index.project_key,
            "project_root": str(index.project_root),
            "collection_name": index.collection_name,
            "chunk_count": index.chunk_count,
            "file_count": index.file_count,
            "indexed_at": index.indexed_at.isoformat().replace("+00:00", "Z"),
            "build_time_seconds": round(index.build_time_seconds, 3),
        }

    def _build_index(
        self,
        root: Path,
        *,
        project_key: str,
        collection_name: str,
        show_progress: bool = False,
    ) -> CodebaseIndex:
        start = time.perf_counter()
        self._store.use_collection(collection_name)
        resolved_entries: List[CodeChunkEmbedding] = []
        file_paths: set[str] = set()

        parser = TreeSitterProjectParser()
        try:
            symbols = parser.parse_project(root)
        finally:
            parser.close()

        covered_paths: set[str] = set()
        reported_paths: set[str] = set()
        pending_entries: List[CodeChunkEmbedding] = []
        for symbol in symbols:
            if symbol.kind is not TagKind.DEF:
                continue
            snippet = symbol.metadata.get("code_snippet")
            if not isinstance(snippet, str):
                continue
            snippet_text = snippet.rstrip("\n")
            if not snippet_text.strip():
                continue
            item = CodeChunkEmbedding(
                relative_path=symbol.relative_path,
                absolute_path=symbol.absolute_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                language=symbol.language,
                symbol=symbol.metadata.get("identifier") or symbol.name,
                content=snippet_text,
            )
            if show_progress and symbol.absolute_path not in reported_paths:
                try:
                    rel = Path(symbol.absolute_path).relative_to(root).as_posix()
                except ValueError:
                    rel = symbol.relative_path
                reported_paths.add(symbol.absolute_path)
            covered_paths.add(symbol.absolute_path)
            file_paths.add(symbol.absolute_path)
            pending_entries.append(item)

        for file_path in iter_repository_files(root):
            absolute = str(file_path)
            if absolute in covered_paths:
                continue
            relative_path = file_path.relative_to(root).as_posix()
            try:
                chunks = chunk_code_file(file_path, chunk_size=self._chunk_size)
            except Exception:
                continue

            if show_progress and absolute not in reported_paths:
                reported_paths.add(absolute)

            for chunk in chunks:
                snippet_text = chunk.content.rstrip("\n")
                if not snippet_text.strip():
                    continue
                item = CodeChunkEmbedding(
                    relative_path=relative_path,
                    absolute_path=absolute,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language,
                    symbol=chunk.symbol,
                    content=snippet_text,
                )
                pending_entries.append(item)
                file_paths.add(item.absolute_path)

        existing_records = self._store.list_point_ids(project_key)
        if existing_records:
            remove_ids = [str(record.id) for record in existing_records.values()]
            self._store.delete_points(remove_ids)

        normalized_vectors: List[Tuple[float, ...]] = []
        if pending_entries:
            texts = [item.content for item in pending_entries]
            labels = [f"{item.relative_path}:{item.start_line}-{item.end_line}" for item in pending_entries]
            embeddings = self._embedder.embed_batch(texts, labels=labels)
            if len(embeddings) != len(pending_entries):
                raise RuntimeError("Embedding service returned mismatched vector count")
            if embeddings:
                self._store.use_collection(
                    collection_name,
                    vector_size=len(embeddings[0]),
                )

            points: List[QdrantPoint] = []
            for item, vector in zip(pending_entries, embeddings):
                normalized = tuple(float(v) for v in vector)
                normalized_vectors.append(normalized)
                payload = {
                    "project_key": project_key,
                    "project_name": root.name,
                    "project_root": str(root),
                    "relative_path": item.relative_path,
                    "absolute_path": item.absolute_path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "language": item.language,
                    "symbol": item.symbol,
                    "snippet": item.content,
                }
                points.append(
                    QdrantPoint(
                        id=self._make_point_id(project_key, item),
                        vector=vector,
                        payload=payload,
                    )
                )
            self._store.upsert_points(points, batch_size=self._batch_size)

            resolved_entries = [
                CodeChunkEmbedding(
                    relative_path=entry.relative_path,
                    absolute_path=entry.absolute_path,
                    start_line=entry.start_line,
                    end_line=entry.end_line,
                    language=entry.language,
                    symbol=entry.symbol,
                    content=entry.content,
                    vector=vector,
                )
                for entry, vector in zip(pending_entries, normalized_vectors)
            ]

        chunk_count = len(pending_entries)
        file_count = len(file_paths)

        build_time = time.perf_counter() - start
        return CodebaseIndex(
            project_root=root,
            project_key=project_key,
            project_name=root.name,
            collection_name=collection_name,
            entries=tuple(resolved_entries),
            file_count=file_count,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc),
            build_time_seconds=build_time,
        )

    @staticmethod
    def _make_point_id(project_key: str, item: CodeChunkEmbedding) -> str:
        raw = f"{project_key}:{item.relative_path}:{item.start_line}:{item.end_line}"
        digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))

    @staticmethod
    def _compile_directory_matcher(target_directories: Optional[Sequence[str]]) -> Optional[Callable[[str], bool]]:
        if not target_directories:
            return None

        patterns: List[str] = []
        for raw in target_directories:
            if not raw:
                continue
            pattern = raw.strip().lstrip("./")
            if not pattern:
                continue
            if pattern.endswith("/"):
                pattern = pattern.rstrip("/") + "/**"
            if "**" not in pattern and not any(char in pattern for char in "*?["):
                pattern = pattern.rstrip("/") + "/**"
            patterns.append(pattern)

        if not patterns:
            return None

        def matcher(path: str) -> bool:
            return any(fnmatch(path, pat) for pat in patterns)

        return matcher

    def __del__(self) -> None:  # pragma: no cover - 清理路径
        self.close()
