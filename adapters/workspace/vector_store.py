from __future__ import annotations

"""Utilities for persisting parsed symbols into a local Qdrant vector store."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


@dataclass
class QdrantPoint:
    """Simple container for a Qdrant point."""

    id: str
    vector: Sequence[float]
    payload: Mapping[str, Any]


class LocalQdrantStore:
    """Thin wrapper around the Qdrant client for symbol indexing."""

    def __init__(self, path: str = "./storage/qdrant") -> None:
        self.client = QdrantClient(path=path)
        self.collection_name: Optional[str] = None
        self._vector_size: Optional[int] = None

    def use_collection(self, collection_name: str, vector_size: Optional[int] = None) -> None:
        """Switch active collection and optionally ensure it exists."""
        self.collection_name = collection_name
        if vector_size is not None:
            self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.collection_name:
            raise ValueError("Collection name not set. Call use_collection() first.")
            
        self._vector_size = vector_size
        
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert_points(self, points: Sequence[QdrantPoint], *, batch_size: int = 128, wait: bool = True) -> None:
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
                self._flush(batch, wait=wait)
                batch = []
        if batch:
            self._flush(batch, wait=wait)

    def list_point_ids(self, project_key: str) -> Dict[str, qmodels.Record]:
        if not self.collection_name:
            return {}
            
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
                    collection_name=self.collection_name,
                    scroll_filter=filter_,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
            except ValueError:
                return {}
            for record in points:
                key = str(record.id)
                records[key] = record
            if offset is None:
                break
        return records

    def delete_points(self, ids: Sequence[str]) -> None:
        if not ids or not self.collection_name:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=list(ids)),
                wait=True,
            )
        except ValueError:
            return

    def delete_project(self, project_name: str, *, project_key: Optional[str] = None) -> None:
        """Delete all points belonging to a project."""
        if not self.collection_name:
            return

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
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=must)),
                wait=True,
            )
        except ValueError:
            return

    def count(self, project_key: str) -> int:
        if not self.collection_name:
            return 0
            
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
                collection_name=self.collection_name,
                exact=True,
                count_filter=filter_,
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
        payload_filter: Optional[qmodels.Filter] = None,
    ) -> List[qmodels.ScoredPoint]:
        if limit <= 0 or not self.collection_name:
            return []
        must = [
            qmodels.FieldCondition(
                key="project_key",
                match=qmodels.MatchValue(value=project_key),
            )
        ]
        if payload_filter is not None:
            must.extend(payload_filter.must or [])
            filter_ = qmodels.Filter(
                must=must,
                should=payload_filter.should,
                must_not=payload_filter.must_not,
            )
        else:
            filter_ = qmodels.Filter(must=must)
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                query_filter=filter_,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
        except ValueError:
            return []
        points = getattr(response, "points", None)
        if not points:
            return []
        return list(points)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
            
    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def _flush(self, points: Sequence[QdrantPoint], wait: bool = True) -> None:
        if not self.collection_name:
            return
            
        payload = [
            qmodels.PointStruct(
                id=point.id,
                vector=list(point.vector),
                payload=dict(point.payload),
            )
            for point in points
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=payload,
            wait=wait,
        )
