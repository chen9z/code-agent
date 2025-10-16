from __future__ import annotations

"""Shared semantic code indexing and search utilities for tools and flows."""

import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from litellm import embedding as litellm_embedding

from configs.manager import get_config
from integrations.splitter import chunk_code_file, iter_repository_files
from integrations.tree_sitter.parser import TagKind, TreeSitterProjectParser
from integrations.vector_store import LocalQdrantStore, QdrantConfig, QdrantPoint


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


DEFAULT_EMBEDDING_BASE = "http://127.0.0.1:8000/v1"


def _resolve_endpoint() -> str:
    """Resolve the embedding API base URL."""

    return (os.getenv("EMBEDDING_API_BASE") or DEFAULT_EMBEDDING_BASE).rstrip("/")


def _project_identifier(root: Path) -> str:
    digest = hashlib.sha1(str(root).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class CodeChunkEmbedding:
    relative_path: str
    absolute_path: str
    start_line: int
    end_line: int
    language: Optional[str]
    symbol: Optional[str]
    content: str
    vector: Tuple[float, ...]


@dataclass(frozen=True)
class CodebaseIndex:
    project_root: Path
    project_key: str
    project_name: str
    entries: Tuple[CodeChunkEmbedding, ...]
    file_count: int
    chunk_count: int
    indexed_at: datetime
    build_time_seconds: float


@dataclass(frozen=True)
class PendingEmbeddingItem:
    relative_path: str
    absolute_path: str
    start_line: int
    end_line: int
    language: Optional[str]
    symbol: Optional[str]
    snippet: str


class EmbeddingClient:
    """Thin wrapper over a LiteLLM-compatible embedding endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        batch_size: int = 16,
        timeout: float = 30.0,
    ) -> None:
        normalized = endpoint.rstrip("/")
        suffix = "/embeddings"
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
        self.api_base = normalized
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("CODEBASE_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        )
        provider = (
            os.getenv("CODEBASE_EMBEDDING_PROVIDER")
            or os.getenv("EMBEDDING_PROVIDER")
            or "openai"
        )
        self.provider = provider
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        span = self.batch_size
        for start in range(0, len(texts), span):
            chunk = list(texts[start : start + span])
            results.extend(self._embed(chunk))
        return results

    def _embed(self, inputs: Sequence[str]) -> List[List[float]]:
        try:
            response = litellm_embedding(
                model=self.model,
                input=list(inputs),
                api_base=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                custom_llm_provider=self.provider,
            )
        except Exception as exc:  # pragma: no cover - 依赖网络
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            raise RuntimeError("Embedding response missing expected 'data' entries")

        vectors: List[List[float]] = []
        for entry in data:
            embedding = getattr(entry, "embedding", None)
            if embedding is None and isinstance(entry, dict):
                embedding = entry.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding response missing 'embedding' vector")
            try:
                vector = [float(v) for v in embedding]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Embedding vector contained non-numeric values") from exc
            vectors.append(vector)
        return vectors


@dataclass(frozen=True)
class SemanticSearchHit:
    score: float
    chunk: CodeChunkEmbedding


class SemanticCodeIndexer:
    """集中式语义索引构建与检索实现。"""

    def __init__(
        self,
        *,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
        max_snippet_chars: int = 800,
        vector_store: Optional[LocalQdrantStore] = None,
        qdrant_config: Optional[QdrantConfig] = None,
    ) -> None:
        cfg = get_config()
        rag_cfg = getattr(cfg, "rag", None)
        chunk_size = 200
        if rag_cfg is not None:
            chunk_size = max(32, int(getattr(rag_cfg, "chunk_size", 200)))

        endpoint = _resolve_endpoint()
        model = os.getenv("CODEBASE_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "jinaai/jina-embeddings-v4"
        batch = int(batch_size or os.getenv("CODEBASE_EMBEDDING_BATCH", "16"))
        api_timeout = float(os.getenv("CODEBASE_EMBEDDING_TIMEOUT", "30"))

        store_path = os.getenv("CODEBASE_QDRANT_PATH", "storage")
        store_root = Path(store_path).expanduser()
        store_root.mkdir(parents=True, exist_ok=True)
        config = qdrant_config or QdrantConfig(path=str(store_root))
        self._store = vector_store or LocalQdrantStore(config)
        self._embedder = embedding_client or EmbeddingClient(
            endpoint=endpoint,
            model=model,
            batch_size=batch,
            timeout=api_timeout,
        )
        self._chunk_size = chunk_size
        self._batch_size = max(1, batch)
        self._indices: Dict[str, CodebaseIndex] = {}
        self._lock = threading.Lock()
        self._max_snippet_chars = max(120, int(max_snippet_chars))

    def ensure_index(self, project_root: Path | str, *, refresh: bool = False) -> CodebaseIndex:
        root = Path(project_root).expanduser().resolve()
        key = str(root)
        with self._lock:
            self._store.use_collection(root.name)
            cached = self._indices.get(key)
            collection_missing = not self._store.collection_exists(root.name)
            if not refresh and cached is not None and not collection_missing:
                return cached
            index = self._build_index(root)
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
        self._store.use_collection(index.project_name)
        embeddings = self._embedder.embed_batch([query])
        if not embeddings:
            return [], index
        query_vector = tuple(float(v) for v in embeddings[0])

        fetch_limit = max(limit, 1)
        if target_directories:
            fetch_limit = min(max(limit * 4, limit + 4), 50)

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
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            entry = hit.chunk
            preview = entry.content.replace("\n", " ")
            preview = _shorten(preview.strip(), 160) if preview else ""
            lines.append(
                f"{idx}. {entry.relative_path}:{entry.start_line}-{entry.end_line} (score {hit.score:.3f})"
            )
            if preview:
                lines.append(f"   {preview}")
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
        summary = "\n".join(lines)
        return {"results": results, "summary": summary}

    @staticmethod
    def index_metadata(index: CodebaseIndex) -> Dict[str, Any]:
        return {
            "project_key": index.project_key,
            "project_root": str(index.project_root),
            "chunk_count": index.chunk_count,
            "file_count": index.file_count,
            "indexed_at": index.indexed_at.isoformat().replace("+00:00", "Z"),
            "build_time_seconds": round(index.build_time_seconds, 3),
        }

    def _build_index(self, root: Path) -> CodebaseIndex:
        start = time.perf_counter()
        project_key = _project_identifier(root)
        self._store.use_collection(root.name)
        entries: List[CodeChunkEmbedding] = []
        file_paths: set[str] = set()

        parser = TreeSitterProjectParser()
        try:
            symbols = parser.parse_project(root)
        finally:
            parser.close()

        covered_paths: set[str] = set()
        items: List[PendingEmbeddingItem] = []
        for symbol in symbols:
            if symbol.kind is not TagKind.DEF:
                continue
            snippet = symbol.metadata.get("code_snippet")
            if not isinstance(snippet, str):
                continue
            snippet_text = snippet.rstrip("\n")
            if not snippet_text.strip():
                continue
            item = PendingEmbeddingItem(
                relative_path=symbol.relative_path,
                absolute_path=symbol.absolute_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                language=symbol.language,
                symbol=symbol.metadata.get("identifier") or symbol.name,
                snippet=snippet_text,
            )
            covered_paths.add(symbol.absolute_path)
            file_paths.add(symbol.absolute_path)
            entries.append(self._entry_from_item(item, None))
            items.append(item)

        for file_path in iter_repository_files(root):
            absolute = str(file_path)
            if absolute in covered_paths:
                continue
            relative_path = file_path.relative_to(root).as_posix()
            try:
                chunks = chunk_code_file(file_path, chunk_size=self._chunk_size)
            except Exception:
                continue

            for chunk in chunks:
                snippet_text = chunk.content.rstrip("\n")
                if not snippet_text.strip():
                    continue
                item = PendingEmbeddingItem(
                    relative_path=relative_path,
                    absolute_path=absolute,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language,
                    symbol=chunk.symbol,
                    snippet=snippet_text,
                )
                entries.append(self._entry_from_item(item, None))
                items.append(item)
                file_paths.add(item.absolute_path)

        existing_records = self._store.list_point_ids(project_key)
        if existing_records:
            remove_ids = [str(record.id) for record in existing_records.values()]
            self._store.delete_points(remove_ids)

        if items:
            texts = [item.snippet for item in items]
            embeddings = self._embedder.embed_batch(texts)
            if len(embeddings) != len(items):
                raise RuntimeError("Embedding service returned mismatched vector count")
            if embeddings:
                self._store.use_collection(root.name, vector_size=len(embeddings[0]))

            points: List[QdrantPoint] = []
            for item, vector in zip(items, embeddings):
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
                    "snippet": _shorten(item.snippet, self._max_snippet_chars),
                }
                points.append(
                    QdrantPoint(
                        id=self._make_point_id(project_key, item),
                        vector=vector,
                        payload=payload,
                    )
                )
            self._store.upsert_points(points, batch_size=self._batch_size)

        chunk_count = len(items)
        file_count = len(file_paths)

        build_time = time.perf_counter() - start
        return CodebaseIndex(
            project_root=root,
            project_key=project_key,
            project_name=root.name,
            entries=tuple(entries),
            file_count=file_count,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc),
            build_time_seconds=build_time,
        )

    def _entry_from_item(self, item: PendingEmbeddingItem, vector: Optional[Sequence[float]]) -> CodeChunkEmbedding:
        normalized: Tuple[float, ...] = tuple(float(v) for v in vector) if vector is not None else ()
        return CodeChunkEmbedding(
            relative_path=item.relative_path,
            absolute_path=item.absolute_path,
            start_line=item.start_line,
            end_line=item.end_line,
            language=item.language,
            symbol=item.symbol,
            content=_shorten(item.snippet, self._max_snippet_chars),
            vector=normalized,
        )

    @staticmethod
    def _make_point_id(project_key: str, item: PendingEmbeddingItem) -> str:
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
