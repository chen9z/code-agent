from pathlib import Path

from adapters.workspace.semantic_splitter import SemanticSplitter


def test_chunk_includes_leading_comments():
    text = """# license\n# details\n\nclass Foo:\n    pass\n\n# stray comment\n\ndef bar():\n    return 1\n"""
    splitter = SemanticSplitter("python", chunk_size=200)
    docs = splitter.split("dummy.py", text)
    contents = [doc.content for doc in docs]
    assert any("class Foo" in chunk and "license" in chunk for chunk in contents)
    assert any("def bar" in chunk and "stray comment" in chunk for chunk in contents)

def test_java_comments_follow_class():
    text = """/**\n * License block\n */\npublic class Foo {\n    void ok() {}\n}\n\n// trailing comment\nclass Bar {\n    void bar() {}\n}\n"""
    splitter = SemanticSplitter("java", chunk_size=200)
    docs = splitter.split("Foo.java", text)
    contents = [doc.content for doc in docs]
    assert any("public class Foo" in chunk and "License block" in chunk for chunk in contents)
    assert any("class Bar" in chunk and "trailing comment" in chunk for chunk in contents)


def test_java_real_world_file_chunking():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "java" / "RetrievalAugmentationAdvisor.java"
    text = fixture.read_text(encoding="utf-8")
    splitter = SemanticSplitter("java", chunk_size=2000)
    docs = splitter.split(str(fixture), text)

    assert len(docs) == 7

    expected_markers = [
        ("Copyright", "package org.springframework.ai.rag.advisor"),
        ("Advisor that implements", "class RetrievalAugmentationAdvisor"),
        ("DOCUMENT_CONTEXT", "documentRetriever"),
        ("Transform original user query", "documentPostProcessors"),
        ("Processes a single query", "getDocumentsForQuery"),
        ("Builder", "queryTransformers"),
        ("documentPostProcessors", "build"),
    ]

    for doc, (marker_a, marker_b) in zip(docs, expected_markers):
        content = doc.content
        assert marker_a in content
        assert marker_b in content

    assert [doc.start_line for doc in docs] == [1, 43, 60, 106, 154, 199, 256]

    def chunk_contains(*snippets: str) -> bool:
        return any(all(snippet in doc.content for snippet in snippets) for doc in docs)

    assert chunk_contains("Advisor that implements", "class RetrievalAugmentationAdvisor")
    assert chunk_contains("Processes a single query", "getDocumentsForQuery")
