from __future__ import annotations

"""Embedding client helpers shared across retrieval pipelines."""

from dataclasses import dataclass
import os
from typing import List, Protocol, Sequence

import httpx


class EmbeddingClientProtocol(Protocol):
    """Minimal interface required by retrieval components."""

    def embed_batch(
            self,
            texts: Sequence[str],
            *,
            task: str = "code",
    ) -> List[List[float]]:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class DefaultEmbeddingClient(EmbeddingClientProtocol):
    endpoint: str
    model: str
    api_key: str | None = None
    batch_size: int = 16
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint
        self.endpoint = f"{self.endpoint}/embeddings"
        self.batch_size = self.batch_size
        if self.api_key is None:
            self.api_key = (
                os.getenv("OPENAI_API_KEY")
            )

    def embed_batch(
            self,
            texts: Sequence[str],
            *,
            task: str = "code",
    ) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start: start + self.batch_size])
            results.extend(self._embed(batch, task=task))
        return results

    def _embed(self, inputs: Sequence[str], *, task: str) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "input": list(inputs),
            "task": task,
        }
        try:
            resp = httpx.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - 网络依赖
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        body = resp.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(inputs):
            raise RuntimeError("Embedding response missing expected 'data' entries")

        vectors: List[List[float]] = []
        for entry in data:
            embedding = None
            if isinstance(entry, dict):
                embedding = entry.get("embedding") or entry.get("vector")
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding response missing 'embedding' vector")
            try:
                vector = [float(v) for v in embedding]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Embedding vector contained non-numeric values") from exc
            vectors.append(vector)
        return vectors


def create_embedding_client(
        *,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 16,
        timeout: float = 120.0,
):
    resolved_endpoint = endpoint or os.getenv("EMBEDDING_API_BASE")
    resolved_model = model or os.getenv("EMBEDDING_MODEL")
    return DefaultEmbeddingClient(
        endpoint=resolved_endpoint,
        model=resolved_model,
        api_key=api_key,
        batch_size=batch_size,
        timeout=timeout,
    )
