from __future__ import annotations

"""Utilities for persisting parsed symbols into a local Qdrant vector store."""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from integrations.tree_sitter import ParsedSymbol


EmbedderFn = Callable[[ParsedSymbol], Sequence[float]]


@dataclass
class QdrantConfig:
    """Configuration options for local Qdrant usage."""

    path: str = "./storage/qdrant"
    collection: str = "code_symbols"
    vector_size: int = 1536
    distance: qmodels.Distance = qmodels.Distance.COSINE


class LocalQdrantStore:
    """Thin wrapper around the Qdrant client for symbol indexing."""

    def __init__(self, config: Optional[QdrantConfig] = None) -> None:
        self.config = config or QdrantConfig()
        self.client = QdrantClient(path=self.config.path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        exists = False
        try:
            exists = self.client.collection_exists(self.config.collection)  # type: ignore[attr-defined]
        except AttributeError:
            # Older qdrant-client versions
            try:
                self.client.get_collection(self.config.collection)
                exists = True
            except Exception:
                exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.config.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.config.vector_size,
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

        batch_points: List[qmodels.PointStruct] = []
        for idx, symbol in enumerate(symbols):
            vector = next(vector_iter)
            if len(vector) != self.config.vector_size:
                raise ValueError(
                    f"Embedding size {len(vector)} != configured vector_size {self.config.vector_size}"
                )
            payload = symbol.to_payload(project_name, project_path)
            if payload_overrides:
                payload.update(payload_overrides)
            point = qmodels.PointStruct(
                id=symbol.point_id(project_name),
                vector=vector,
                payload=payload,
            )
            batch_points.append(point)
            if len(batch_points) >= batch_size:
                self._flush(batch_points)
                batch_points = []

        if batch_points:
            self._flush(batch_points)

    def _flush(self, points: List[qmodels.PointStruct]) -> None:
        self.client.upsert(
            collection_name=self.config.collection,
            points=points,
            wait=True,
        )

    def delete_project(self, project_name: str) -> None:
        """Delete all points belonging to a project."""

        self.client.delete(
            collection_name=self.config.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="project_name",
                            match=qmodels.MatchValue(value=project_name),
                        )
                    ]
                )
            ),
            wait=True,
        )
