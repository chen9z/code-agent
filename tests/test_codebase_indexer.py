from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import retrieval.codebase_indexer as codebase_indexer

from retrieval.codebase_indexer import CodebaseIndex, SemanticCodeIndexer
from config.config import reload_config


class DummyEmbedder:
    def embed_batch(self, texts, *, task="code"):
        return [[0.1, 0.1] for _ in texts]


class DummyStore:
    def __init__(self) -> None:
        self.active_collection: str | None = None
        self.collections_with_vectors: set[str] = set()
        self.use_calls: list[tuple[str, int | None]] = []
        self.search_calls: list[str | None] = []
        self.last_filter = None
        self.upserts: list = []
        self.deleted: list[str] = []
        self.records_by_key: dict[str, dict[str, SimpleNamespace]] = {}
        self.prefix_indexes: set[str] = set()

    def use_collection(self, name: str, vector_size: int | None = None) -> None:
        self.active_collection = name
        self.use_calls.append((name, vector_size))
        if vector_size is not None:
            self.collections_with_vectors.add(name)
        self.prefix_indexes.add("relative_path")
        self.prefix_indexes.add("relative_path")

    def collection_exists(self, name: str) -> bool:
        return name in self.collections_with_vectors

    def search(self, *, vector, project_key, limit, score_threshold=None, payload_filter=None):  # noqa: D401
        self.search_calls.append(self.active_collection)
        self.last_filter = payload_filter
        return []

    def list_point_ids(self, project_key):
        return self.records_by_key.get(project_key, {})

    def delete_points(self, remove_ids):
        self.deleted.extend(remove_ids)

    def upsert_points(self, points, batch_size: int) -> None:
        self.upserts.append((points, batch_size))
        # 模拟已写入存储，方便 hydrate 逻辑读取
        for point in points:
            payload = getattr(point, "payload", {})
            record = SimpleNamespace(payload=payload)
            project_key = payload.get("project_key")
            if project_key:
                bucket = self.records_by_key.setdefault(project_key, {})
                bucket[str(point.id)] = record

    # Methods below satisfy the interface expected by SemanticCodeIndexer during cleanup.
    def close(self) -> None:  # pragma: no cover - noop cleanup
        pass


def _fake_index(root: Path, project_key: str, collection_name: str) -> CodebaseIndex:
    return CodebaseIndex(
        project_root=root,
        project_key=project_key,
        project_name=root.name,
        collection_name=collection_name,
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


def test_indexer_reports_progress(monkeypatch, tmp_path, caplog):
    foo = tmp_path / "foo.py"
    foo.write_text("def foo():\n    return 1\n", encoding="utf-8")
    bar = tmp_path / "bar.py"
    bar.write_text("def bar():\n    return foo()\n", encoding="utf-8")

    def fake_iter_repository_files(root):
        return [foo, bar]

    def fake_chunk_code_file(path, *, chunk_size):
        text = Path(path).read_text(encoding="utf-8")
        return [
            SimpleNamespace(
                content=text,
                start_line=1,
                end_line=len(text.splitlines()),
                language="python",
            )
        ]

    monkeypatch.setattr(codebase_indexer, "iter_repository_files", fake_iter_repository_files)
    monkeypatch.setattr(codebase_indexer, "chunk_code_file", fake_chunk_code_file)

    store = DummyStore()
    embedder = DummyEmbedder()
    indexer = SemanticCodeIndexer(
        embedding_client=embedder,
        vector_store=store,
        batch_size=2,
    )

    with caplog.at_level(logging.INFO, logger="retrieval.codebase_indexer"):
        indexer.ensure_index(tmp_path, refresh=True, show_progress=True)
    messages = [record.getMessage() for record in caplog.records if record.name == "retrieval.codebase_indexer"]
    assert any("分块文件" in msg for msg in messages)
    assert any("完成索引" in msg for msg in messages)


def test_indexer_skips_rebuild_when_collection_exists(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    project_key = codebase_indexer._project_identifier(repo)
    collection_name = codebase_indexer._collection_name(repo.name, project_key)

    store = DummyStore()
    store.collections_with_vectors.add(collection_name)
    payload = {
        "project_key": project_key,
        "project_name": repo.name,
        "project_root": str(repo),
        "relative_path": "foo.py",
        "absolute_path": f"{repo}/foo.py",
        "start_line": 1,
        "end_line": 2,
        "language": "python",
        "snippet": "def foo():\n    return 1\n",
    }
    store.records_by_key[project_key] = {"foo": SimpleNamespace(payload=payload)}

    embedder = DummyEmbedder()
    indexer = SemanticCodeIndexer(embedding_client=embedder, vector_store=store)

    def boom(*_args, **_kwargs):
        raise AssertionError("should not rebuild")

    monkeypatch.setattr(SemanticCodeIndexer, "_build_index", boom)

    index = indexer.ensure_index(repo, refresh=False)
    assert index.collection_name == collection_name


def test_build_payload_filter_passes_through_prefix():
    filter_ = SemanticCodeIndexer._build_payload_filter("proj", ["src/agent", "docs/**", ""])
    assert filter_ is not None
    tokens = [cond.match.text for cond in filter_.should]
    assert "src/agent" in tokens
    assert "docs/**" in tokens


def test_indexer_covers_top_level_with_large_chunk(monkeypatch):
    """chunk_size 2048 时，顶层 __main__ 逻辑也会被索引。"""

    repo_root = Path(__file__).resolve().parent.parent
    target_file = repo_root / "code_agent.py"
    assert target_file.exists()

    with monkeypatch.context():
        monkeypatch.setenv("CHUNK_SIZE", "2048")
        reload_config()

        def fake_iter_repository_files(_root):
            return [target_file]

        monkeypatch.setattr(codebase_indexer, "iter_repository_files", fake_iter_repository_files)

        store = DummyStore()
        embedder = DummyEmbedder()
        indexer = SemanticCodeIndexer(
            embedding_client=embedder,
            vector_store=store,
            batch_size=8,
        )

        indexer.ensure_index(repo_root, refresh=True, show_progress=False)

        assert indexer.chunk_size == 2048

        payloads = [p.payload for batch, _ in store.upserts for p in batch]
        assert any("__main__" in (pl.get("snippet") or "") for pl in payloads)

    reload_config()
