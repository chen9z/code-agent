from __future__ import annotations

from adapters.workspace.vector_store import LocalQdrantStore, QdrantConfig, QdrantPoint


def test_local_qdrant_store_roundtrip(tmp_path):
    config = QdrantConfig(path=str(tmp_path / "qdrant"), vector_size=3)
    store = LocalQdrantStore(config)

    payload = {
        "project_name": "demo_project",
        "project_path": str(tmp_path / "repo"),
        "language": "python",
        "relative_path": "example.py",
        "absolute_path": str(tmp_path / "repo" / "example.py"),
        "start_line": 1,
        "end_line": 5,
        "symbol_name": "area",
        "symbol_kind": "def",
        "metadata": {"code_snippet": "def area(r):\n    return r * r\n", "references": []},
    }
    point = QdrantPoint(
        id="b3b9f8c0-1f2b-11ee-be56-0242ac120002",  # valid UUID string
        vector=[0.1, 0.2, 0.3],
        payload=payload,
    )
    store.upsert_points([point], batch_size=10)

    count = store.client.count(collection_name=config.collection, exact=True)
    assert count.count == 1

    points, _ = store.client.scroll(
        collection_name=config.collection,
        with_payload=True,
        with_vectors=False,
    )
    assert points
    payload = points[0].payload
    assert payload["project_name"] == "demo_project"
    assert payload["metadata"]["code_snippet"].startswith("def area")

    store.delete_project("demo_project")
    count_after = store.client.count(collection_name=config.collection, exact=True)
    assert count_after.count == 0
