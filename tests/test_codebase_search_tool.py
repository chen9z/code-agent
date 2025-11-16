from __future__ import annotations

from pathlib import Path
from typing import Sequence
from uuid import uuid4

from tools.codebase_search import CodebaseSearchTool


class DummyEmbeddingClient:
    """Deterministic embedding stub for tests."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: Sequence[str], *, task: str = "code") -> list[list[float]]:
        payload = list(texts)
        self.calls.append(payload)
        vectors: list[list[float]] = []
        for text in payload:
            lower = text.lower()
            foo = lower.count("foo") or 0.0001
            bar = lower.count("bar") + 0.001
            norm = (foo ** 2 + bar ** 2) ** 0.5
            if norm == 0.0:
                norm = 1.0
            vectors.append([float(foo / norm), float(bar / norm)])
        return vectors


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_codebase_search_ranks_matches(tmp_path, monkeypatch):
    write_file(tmp_path / "foo.py", "def foo():\n    return 1\n")
    write_file(tmp_path / "bar.py", "def bar():\n    return 2\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CODEBASE_QDRANT_COLLECTION", f"test_chunks_{uuid4().hex}")

    client = DummyEmbeddingClient()
    tool = CodebaseSearchTool(embedding_client=client, batch_size=4)

    result = tool.execute(query="Find foo helper")

    assert result["status"] == "success"
    data = result["data"]
    assert data["count"] >= 2
    assert len(data["results"]) >= 2
    assert data["results"][0]["path"].endswith("foo.py")
    assert data["results"][0]["score"] >= data["results"][1]["score"]
    assert "foo" in data["results"][0]["snippet"].lower()
    assert len(client.calls) == 2  # one for indexing, one for query


def test_codebase_search_uses_cached_embeddings(tmp_path, monkeypatch):
    write_file(tmp_path / "foo.py", "def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CODEBASE_QDRANT_COLLECTION", f"test_chunks_{uuid4().hex}")

    client = DummyEmbeddingClient()
    tool = CodebaseSearchTool(embedding_client=client, batch_size=4)

    _ = tool.execute(query="Find foo helper")
    first_calls = len(client.calls)

    _ = tool.execute(query="Find foo helper")

    assert len(client.calls) == first_calls + 1  # only the query embeds again


def test_codebase_search_refresh_index(tmp_path, monkeypatch):
    target = tmp_path / "lib.py"
    write_file(target, "def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CODEBASE_QDRANT_COLLECTION", f"test_chunks_{uuid4().hex}")

    client = DummyEmbeddingClient()
    tool = CodebaseSearchTool(embedding_client=client, batch_size=4)

    initial = tool.execute(query="Where is foo?")
    assert initial["data"]["results"][0]["path"].endswith("lib.py")

    # Update file with new dominant token and refresh index
    write_file(target, "def bar():\n    foo()\n")

    stale = tool.execute(query="Where is foo?")
    assert stale["data"]["results"][0]["path"].endswith("lib.py")
    refreshed = tool.execute(query="Where is bar?", refresh_index=True)
    assert refreshed["data"]["results"][0]["path"].endswith("lib.py")
    assert "bar" in refreshed["data"]["results"][0]["snippet"].lower()


def test_codebase_search_target_directories_filters_results(tmp_path, monkeypatch):
    write_file(tmp_path / "foo" / "alpha.py", "def foo_helper():\n    return 1\n")
    write_file(tmp_path / "bar" / "beta.py", "def bar_helper():\n    return 2\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CODEBASE_QDRANT_COLLECTION", f"test_chunks_{uuid4().hex}")

    client = DummyEmbeddingClient()
    tool = CodebaseSearchTool(embedding_client=client, batch_size=4)

    result = tool.execute(query="helper", target_directories=["foo"])

    assert result["status"] == "success"
    data = result["data"]
    assert data["count"] >= 1
    paths = {entry["path"] for entry in data["results"]}
    assert any(path.endswith("foo/alpha.py") for path in paths)
    assert not any(path.endswith("bar/beta.py") for path in paths)
    assert data["target_directories"] == ["foo"]


def test_codebase_search_rejects_empty_query(tmp_path, monkeypatch):
    write_file(tmp_path / "a.py", "def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEBASE_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CODEBASE_QDRANT_COLLECTION", f"test_chunks_{uuid4().hex}")

    client = DummyEmbeddingClient()
    tool = CodebaseSearchTool(embedding_client=client)

    result = tool.execute(query="   ")
    assert result["status"] == "error"
    assert "query" in result["data"]
