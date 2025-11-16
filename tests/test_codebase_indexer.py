from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from integrations.codebase_indexer import CodebaseIndex, SemanticCodeIndexer


class DummyEmbedder:
    def embed_batch(self, texts, *, labels=None):
        return [[0.1, 0.1] for _ in texts]


class DummyStore:
    def __init__(self) -> None:
        self.active_collection: str | None = None
        self.collections_with_vectors: set[str] = set()
        self.use_calls: list[tuple[str, int | None]] = []
        self.search_calls: list[str | None] = []

    def use_collection(self, name: str, vector_size: int | None = None) -> None:
        self.active_collection = name
        self.use_calls.append((name, vector_size))
        if vector_size is not None:
            self.collections_with_vectors.add(name)

    def collection_exists(self, name: str) -> bool:
        return name in self.collections_with_vectors

    def search(self, *, vector, project_key, limit, score_threshold=None):  # noqa: D401
        self.search_calls.append(self.active_collection)
        return []

    # Methods below satisfy the interface expected by SemanticCodeIndexer during cleanup.
    def close(self) -> None:  # pragma: no cover - noop cleanup
        pass


def _fake_index(root: Path, project_key: str, collection_name: str) -> CodebaseIndex:
    return CodebaseIndex(
        project_root=root,
        project_key=project_key,
        project_name=root.name,
        collection_name=collection_name,
        entries=tuple(),
        file_count=0,
        chunk_count=0,
        indexed_at=datetime.now(timezone.utc),
        build_time_seconds=0.0,
    )


def test_indexer_uses_unique_collection_names(tmp_path, monkeypatch):
    repo_a = tmp_path / "repo"
    repo_b = tmp_path / "other" / "repo"
    repo_a.mkdir()
    repo_b.parent.mkdir(parents=True, exist_ok=True)
    repo_b.mkdir()

    store = DummyStore()
    embedder = DummyEmbedder()
    indexer = SemanticCodeIndexer(embedding_client=embedder, vector_store=store)

    built: list[tuple[Path, str]] = []

    def fake_build(self, root, *, project_key, collection_name, show_progress=False):
        built.append((root, collection_name))
        store.collections_with_vectors.add(collection_name)
        return _fake_index(root, project_key, collection_name)

    monkeypatch.setattr(SemanticCodeIndexer, "_build_index", fake_build)

    index_a = indexer.ensure_index(repo_a)
    index_b = indexer.ensure_index(repo_b)

    assert index_a.collection_name != index_b.collection_name
    assert all(name.startswith("repo_") for _, name in built)

    indexer.search(repo_a, "needle")
    assert store.search_calls[-1] == index_a.collection_name
