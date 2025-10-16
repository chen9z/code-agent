from __future__ import annotations

"""Utilities for persisting parsed symbols into a local Qdrant vector store."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from integrations.tree_sitter.parser import ParsedSymbol


EmbedderFn = Callable[[ParsedSymbol], Sequence[float]]


@dataclass
class QdrantConfig:
    """Configuration options for local Qdrant usage."""

    path: str = "./storage/qdrant"
    collection: str = "code_symbols"
    vector_size: Optional[int] = None
    distance: qmodels.Distance = qmodels.Distance.COSINE


@dataclass
class QdrantPoint:
    """Simple container for a Qdrant point."""

    id: str
    vector: Sequence[float]
    payload: Mapping[str, Any]


class LocalQdrantStore:
    """Thin wrapper around the Qdrant client for symbol indexing."""

    def __init__(self, config: Optional[QdrantConfig] = None) -> None:
        self.config = config or QdrantConfig()
        self.client = QdrantClient(path=self.config.path)
        self._vector_size: Optional[int] = self.config.vector_size
        if self._vector_size is not None:
            self._ensure_collection(self._vector_size)

    def use_collection(self, collection_name: str, vector_size: Optional[int] = None) -> None:
        """Switch active collection and optionally ensure it exists."""

        self.config.collection = collection_name
        if vector_size is not None:
            self._ensure_collection(vector_size)

    @property
    def vector_size(self) -> Optional[int]:
        return self._vector_size

    def ensure_collection(self, vector_size: int) -> None:
        self._ensure_collection(vector_size)

    def collection_exists(self, collection_name: str) -> bool:
        """Return whether the given collection exists without mutating state."""

        if self.config.collection == collection_name:
            return self._collection_exists()

        original_collection = self.config.collection
        try:
            self.config.collection = collection_name
            return self._collection_exists()
        finally:
            self.config.collection = original_collection

    def describe_collection(self) -> Optional[Any]:
        try:
            return self.client.get_collection(self.config.collection)
        except AttributeError:
            # Older qdrant-client versions
            try:
                return self.client.get_collection(self.config.collection)
            except Exception:
                return None
        except Exception:
            return None

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.config.collection))  # type: ignore[attr-defined]
        except AttributeError:
            try:
                self.client.get_collection(self.config.collection)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _ensure_collection(self, vector_size: Optional[int] = None) -> None:
        size = vector_size or self._vector_size or self.config.vector_size
        if size is None:
            raise ValueError("vector_size is required to initialize Qdrant collection")

        if self._vector_size is None:
            self._vector_size = size
        if self.config.vector_size is None:
            self.config.vector_size = size

        if self._collection_exists():
            return

        self.client.create_collection(
            collection_name=self.config.collection,
            vectors_config=qmodels.VectorParams(
                size=size,
                distance=self.config.distance,
            ),
        )

    def upsert_symbols(
        self,
        project_name: str,
        project_path: str,
        symbols: Sequence[ParsedSymbol],
        *,
        embedder: Optional[EmbedderFn] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        payload_overrides: Optional[Mapping[str, object]] = None,
        batch_size: int = 128,
    ) -> None:
        """Upsert ParsedSymbol entries into Qdrant."""

        if not symbols:
            return
        if embeddings is None and embedder is None:
            raise ValueError("Either embeddings or embedder must be provided")
        if embeddings is not None and len(embeddings) != len(symbols):
            raise ValueError("Embeddings length must match symbols length")

        def iter_vectors() -> Iterable[Sequence[float]]:
            if embeddings is not None:
                for vector in embeddings:
                    yield vector
            else:
                assert embedder is not None
                for symbol in symbols:
                    yield embedder(symbol)

        vector_iter = iter_vectors()

        points: List[QdrantPoint] = []
        for symbol in symbols:
            vector = next(vector_iter)
            payload = symbol.to_payload(project_name, project_path)
            if payload_overrides:
                payload.update(payload_overrides)
            points.append(
                QdrantPoint(
                    id=symbol.point_id(project_name),
                    vector=vector,
                    payload=payload,
                )
            )
        self.upsert_points(points, batch_size=batch_size)

    def upsert_points(self, points: Sequence[QdrantPoint], *, batch_size: int = 128) -> None:
        if not points:
            return
        vector_size = len(points[0].vector)
        if vector_size == 0:
            raise ValueError("Vectors must be non-empty")
        self._ensure_collection(vector_size)

        batch: List[QdrantPoint] = []
        for point in points:
            if len(point.vector) != vector_size:
                raise ValueError(
                    f"Embedding size {len(point.vector)} != expected vector size {vector_size}"
                )
            batch.append(point)
            if len(batch) >= batch_size:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)

    def list_point_ids(self, project_key: str) -> Dict[str, qmodels.Record]:
        filter_ = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="project_key",
                    match=qmodels.MatchValue(value=project_key),
                )
            ]
        )
        records: Dict[str, qmodels.Record] = {}
        offset: Optional[Dict[str, Any]] = None
        while True:
            try:
                points, offset = self.client.scroll(
                    collection_name=self.config.collection,
                    scroll_filter=filter_,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except ValueError:
                return {}
            for record in points:
                payload = record.payload or {}
                cache_key = payload.get("cache_key") if isinstance(payload, dict) else None
                key = cache_key if isinstance(cache_key, str) else str(record.id)
                records[key] = record
            if offset is None:
                break
        return records

    def delete_points(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        try:
            self.client.delete(
                collection_name=self.config.collection,
                points_selector=qmodels.PointIdsList(points=list(ids)),
                wait=True,
            )
        except ValueError:
            return

    def delete_project(self, project_name: str, *, project_key: Optional[str] = None) -> None:
        """Delete all points belonging to a project."""

        must: List[qmodels.FieldCondition] = []
        if project_name:
            must.append(
                qmodels.FieldCondition(
                    key="project_name",
                    match=qmodels.MatchValue(value=project_name),
                )
            )
        if project_key:
            must.append(
                qmodels.FieldCondition(
                    key="project_key",
                    match=qmodels.MatchValue(value=project_key),
                )
            )
        if not must:
            raise ValueError("At least one identifier is required to delete a project")

        try:
            self.client.delete(
                collection_name=self.config.collection,
                points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=must)),
                wait=True,
            )
        except ValueError:
            return

    def count(self, project_key: str) -> int:
        filter_ = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="project_key",
                    match=qmodels.MatchValue(value=project_key),
                )
            ]
        )
        try:
            result = self.client.count(
                collection_name=self.config.collection,
                exact=True,
                filter=filter_,
            )
        except ValueError:
            return 0
        return int(getattr(result, "count", 0))

    def search(
        self,
        *,
        vector: Sequence[float],
        project_key: str,
        limit: int,
        score_threshold: Optional[float] = None,
    ) -> List[qmodels.ScoredPoint]:
        if limit <= 0:
            return []
        filter_ = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="project_key",
                    match=qmodels.MatchValue(value=project_key),
                )
            ]
        )
        try:
            return self.client.search(
                collection_name=self.config.collection,
                query_vector=list(vector),
                query_filter=filter_,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
        except ValueError:
            return []

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _flush(self, points: Sequence[QdrantPoint]) -> None:
        payload = [
            qmodels.PointStruct(
                id=point.id,
                vector=list(point.vector),
                payload=dict(point.payload),
            )
            for point in points
        ]
        self.client.upsert(
            collection_name=self.config.collection,
            points=payload,
            wait=True,
        )
