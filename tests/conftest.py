"""Pytest configuration for code-agent tests."""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

from retrieval.codebase_indexer import EmbeddingClient

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env so opik/other integrations inherit expected credentials in tests.
load_dotenv(dotenv_path=project_root / ".env", override=False)
if not os.getenv("OPIK_API_KEY"):
    os.environ["OPIK_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def patch_embedding_client(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_BASE", "http://localhost:0")
    monkeypatch.setenv("EMBEDDING_MODEL", "dummy-embedding")
    def fake_embed(self, texts, *, task="code"):
        results = []
        for text in texts:
            lower = str(text).lower()
            foo = lower.count("foo") or 0.0001
            bar = lower.count("bar") + 0.001
            norm = (foo ** 2 + bar ** 2) ** 0.5
            if norm == 0.0:
                norm = 1.0
            results.append([foo / norm, bar / norm])
        return results

    monkeypatch.setattr(EmbeddingClient, "embed_batch", fake_embed, raising=False)
