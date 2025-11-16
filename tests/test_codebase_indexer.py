from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import retrieval.codebase_indexer as codebase_indexer

from retrieval.codebase_indexer import CodebaseIndex, SemanticCodeIndexer
from adapters.workspace.tree_sitter.parser import TagKind


class DummyEmbedder:
    def embed_batch(self, texts, *, task="code"):
        return [[0.1, 0.1] for _ in texts]


class DummyStore:
    def __init__(self) -> None:
        self.active_collection: str | None = None
        self.collections_with_vectors: set[str] = set()
        self.use_calls: list[tuple[str, int | None]] = []
        self.search_calls: list[str | None] = []
        self.upserts: list = []
        self.deleted: list[str] = []

    def use_collection(self, name: str, vector_size: int | None = None) -> None:
        self.active_collection = name
        self.use_calls.append((name, vector_size))
        if vector_size is not None:
            self.collections_with_vectors.add(name)

    def collection_exists(self, name: str) -> bool:
        return name in self.collections_with_vectors

    def search(self, *, vector, project_key, limit, score_threshold=None, payload_filter=None):  # noqa: D401
        self.search_calls.append(self.active_collection)
        return []

    def list_point_ids(self, project_key):
        return {}

    def delete_points(self, remove_ids):
        self.deleted.extend(remove_ids)

    def upsert_points(self, points, batch_size: int) -> None:
        self.upserts.append((points, batch_size))

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


def test_indexer_reports_progress(monkeypatch, tmp_path, caplog):
    foo = tmp_path / "foo.py"
    foo.write_text("def foo():\n    return 1\n", encoding="utf-8")
    bar = tmp_path / "bar.py"
    bar.write_text("def bar():\n    return foo()\n", encoding="utf-8")

    symbol = SimpleNamespace(
        kind=TagKind.DEF,
        metadata={"code_snippet": "def foo():\n    return 1\n", "identifier": "foo"},
        relative_path="foo.py",
        absolute_path=str(foo),
        start_line=1,
        end_line=2,
        language="python",
        name="foo",
    )

    class FakeParser:
        def parse_project(self, root):
            return [symbol]

        def close(self):
            pass

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
                symbol=None,
            )
        ]

    monkeypatch.setattr(codebase_indexer, "TreeSitterProjectParser", lambda: FakeParser())
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
    assert any("解析符号" in msg for msg in messages)
    assert any("分块文件" in msg for msg in messages)
    assert any("完成索引" in msg for msg in messages)
