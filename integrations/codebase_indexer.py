from __future__ import annotations

"""Shared semantic code indexing and search utilities for tools and flows."""

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from diskcache import Cache

from configs.manager import get_config
from integrations.splitter import chunk_code_file, iter_repository_files
from integrations.tree_sitter.parser import TagKind, TreeSitterProjectParser


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _resolve_endpoint() -> str:
    explicit = os.getenv("CODEBASE_EMBEDDING_ENDPOINT")
    if explicit:
        return explicit.rstrip("/")

    base = (
        os.getenv("CODEBASE_EMBEDDING_API_BASE")
        or os.getenv("EMBEDDING_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or "http://127.0.0.1:8000/v1"
    )
    base = base.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


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
    cache_hits: int
    cache_misses: int


@dataclass(frozen=True)
class PendingEmbeddingItem:
    cache_key: str
    relative_path: str
    absolute_path: str
    start_line: int
    end_line: int
    language: Optional[str]
    symbol: Optional[str]
    snippet: str


class EmbeddingClient:
    """Thin wrapper over an OpenAI-compatible embedding HTTP endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        batch_size: int = 16,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("CODEBASE_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        )
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
        payload = json.dumps({"model": self.model, "input": list(inputs)}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(self.endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - 网络错误
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding request failed with status {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - 网络错误
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - 非法响应
            raise RuntimeError("Embedding response was not valid JSON") from exc

        data = parsed.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            raise RuntimeError("Embedding response missing expected 'data' entries")

        vectors: List[List[float]] = []
        for entry in data:
            embedding = entry.get("embedding") if isinstance(entry, dict) else None
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
        cache: Optional[Cache] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
        max_snippet_chars: int = 800,
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

        cache_dir = Path(os.getenv("CODEBASE_SEARCH_CACHE_DIR", "storage/codebase_search_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = cache or Cache(cache_dir)
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
            if not refresh and key in self._indices:
                return self._indices[key]
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
        candidates = self._filter_entries(index.entries, target_directories)
        if not candidates:
            return [], index

        query_vector = tuple(float(v) for v in self._embedder.embed_batch([query])[0])
        scored: List[SemanticSearchHit] = []
        for entry in candidates:
            if not entry.vector:
                continue
            score = sum(q * v for q, v in zip(query_vector, entry.vector))
            scored.append(SemanticSearchHit(score=score, chunk=entry))
        scored.sort(key=lambda item: item.score, reverse=True)
        if limit >= 0:
            scored = scored[: max(1, limit)]
        return scored, index

    def close(self) -> None:
        try:
            self._cache.close()
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
            "cache_hits": index.cache_hits,
            "cache_misses": index.cache_misses,
        }

    def _build_index(self, root: Path) -> CodebaseIndex:
        start = time.perf_counter()
        project_key = _project_identifier(root)
        entries: List[CodeChunkEmbedding] = []
        pending: List[PendingEmbeddingItem] = []
        cache_hits = 0
        cache_misses = 0

        parser = TreeSitterProjectParser()
        try:
            symbols = parser.parse_project(root)
        finally:
            parser.close()

        covered_paths: set[str] = set()

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
                cache_key=self._make_cache_key(
                    project_key,
                    symbol.relative_path,
                    symbol.start_line,
                    symbol.end_line,
                    snippet_text,
                ),
                relative_path=symbol.relative_path,
                absolute_path=symbol.absolute_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                language=symbol.language,
                symbol=symbol.metadata.get("identifier") or symbol.name,
                snippet=snippet_text,
            )
            covered_paths.add(symbol.absolute_path)
            cached_vector = self._cache.get(item.cache_key)
            if cached_vector is not None:
                cache_hits += 1
                entries.append(self._entry_from_item(item, cached_vector))
            else:
                cache_misses += 1
                pending.append(item)
                if len(pending) >= self._batch_size:
                    entries.extend(self._flush_pending(pending))

        if pending:
            entries.extend(self._flush_pending(pending))

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
                    cache_key=self._make_cache_key(
                        project_key,
                        relative_path,
                        chunk.start_line,
                        chunk.end_line,
                        snippet_text,
                    ),
                    relative_path=relative_path,
                    absolute_path=absolute,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language,
                    symbol=chunk.symbol,
                    snippet=snippet_text,
                )
                cached_vector = self._cache.get(item.cache_key)
                if cached_vector is not None:
                    cache_hits += 1
                    entries.append(self._entry_from_item(item, cached_vector))
                else:
                    cache_misses += 1
                    pending.append(item)
                    if len(pending) >= self._batch_size:
                        entries.extend(self._flush_pending(pending))

        if pending:
            entries.extend(self._flush_pending(pending))

        build_time = time.perf_counter() - start
        return CodebaseIndex(
            project_root=root,
            project_key=project_key,
            project_name=root.name,
            entries=tuple(entries),
            file_count=len({entry.absolute_path for entry in entries}),
            chunk_count=len(entries),
            indexed_at=datetime.now(timezone.utc),
            build_time_seconds=build_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def _entry_from_item(self, item: PendingEmbeddingItem, vector: Sequence[float]) -> CodeChunkEmbedding:
        normalized = tuple(float(v) for v in vector)
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

    def _flush_pending(self, pending: List[PendingEmbeddingItem]) -> List[CodeChunkEmbedding]:
        items = list(pending)
        pending.clear()
        if not items:
            return []
        texts: List[str] = [item.snippet for item in items]
        embeddings = self._embedder.embed_batch(texts)
        if len(embeddings) != len(items):
            raise RuntimeError("Embedding service returned mismatched vector count")

        new_entries: List[CodeChunkEmbedding] = []
        for item, vector in zip(items, embeddings):
            normalized = tuple(float(v) for v in vector)
            self._cache.set(item.cache_key, normalized)
            new_entries.append(self._entry_from_item(item, normalized))
        return new_entries

    def _make_cache_key(
        self,
        project_key: str,
        relative_path: str,
        start_line: int,
        end_line: int,
        snippet: str,
    ) -> str:
        digest = hashlib.sha1(snippet.encode("utf-8", errors="ignore")).hexdigest()
        return f"{project_key}:{relative_path}:{start_line}:{end_line}:{digest}"

    @staticmethod
    def _filter_entries(
        entries: Sequence[CodeChunkEmbedding], target_directories: Optional[Sequence[str]]
    ) -> List[CodeChunkEmbedding]:
        if not target_directories:
            return list(entries)

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
            return list(entries)

        filtered: List[CodeChunkEmbedding] = []
        for entry in entries:
            rel_path = entry.relative_path
            if any(fnmatch(rel_path, pat) for pat in patterns):
                filtered.append(entry)
        return filtered

    def __del__(self) -> None:  # pragma: no cover - 清理路径
        self.close()
