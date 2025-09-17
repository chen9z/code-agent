from __future__ import annotations

from integrations.tree_sitter import ParsedSymbol, TagKind
from integrations.vector_store import LocalQdrantStore, QdrantConfig


def test_local_qdrant_store_roundtrip(tmp_path):
    symbol = ParsedSymbol(
        relative_path="example.py",
        absolute_path=str(tmp_path / "repo" / "example.py"),
        language="python",
        start_line=1,
        end_line=5,
        name="area",
        kind=TagKind.DEF,
        metadata={
            "code_snippet": "def area(r):\n    return r * r\n",
            "references": [],
        },
    )

    config = QdrantConfig(path=str(tmp_path / "qdrant"), vector_size=3)
    store = LocalQdrantStore(config)

    store.upsert_symbols(
        "demo_project",
        project_path=str(tmp_path / "repo"),
        symbols=[symbol],
        embeddings=[[0.1, 0.2, 0.3]],
    )

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
