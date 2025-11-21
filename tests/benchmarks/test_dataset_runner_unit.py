from __future__ import annotations

import json
from pathlib import Path

from adapters.llm.llm import BaseLLMClient
from benchmarks.dataset.models import QuerySpec
from benchmarks.dataset.runner import DatasetRunner, build_dataset_from_raw, load_query_specs


class FakeLLMClient(BaseLLMClient):
    def __init__(self, chunk_args: dict[str, object]):
        self.chunk_args = chunk_args
        self.calls = 0

    def get_response(self, *_, **__):  # pragma: no cover - not used
        raise NotImplementedError

    def create_with_tools(self, **kwargs):
        if self.calls == 0:
            self.calls += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": "logging",
                            "tool_calls": [
                                {
                                    "id": "call-0",
                                    "function": {
                                        "name": "dataset_log_write_chunk",
                                        "arguments": json.dumps(self.chunk_args),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}}]}


def test_runner_logs_chunk(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "demo.py").write_text("print('demo')\n", encoding="utf-8")

    spec = QuerySpec(
        query_id="q-demo",
        query="print demo",
        repo_url="repo-url",
        branch="main",
        commit_id="abc123",
        path=str(repo),
    )
    llm = FakeLLMClient(
        {
            "path": "src/demo.py",
            "start_line": 1,
            "end_line": 1,
            "confidence": 0.95,
        }
    )
    artifacts_root = tmp_path / "storage" / "dataset"
    runner = DatasetRunner(llm_client=llm, run_name="run1", artifacts_root=artifacts_root)

    results = runner.run_queries([(spec, repo)])
    assert results[0].success

    raw_file = artifacts_root / "run1" / "raw_samples" / "q-demo.jsonl"
    assert raw_file.exists()
    rows = [json.loads(line) for line in raw_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    record = rows[0]
    assert record["repo_url"] == "repo-url"
    assert record["branch"] == "main"
    assert record["commit_id"] == "abc123"
    assert "repo" not in record
    chunk = record["chunk"]
    assert chunk["path"] == "src/demo.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 1


def test_load_query_specs_top_level_schema(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "query_id": "q-top",
                "query": "demo",
                "repo_url": "repo",
                "branch": "dev",
                "commit_id": "cafebabe",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    specs = load_query_specs(queries)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.query_id == "q-top"
    assert spec.repo_url == "repo"
    assert spec.branch == "dev"
    assert spec.commit_id == "cafebabe"
    assert spec.path is None


def test_load_query_specs_defaults_master(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "query_id": "q-default",
                "query": "demo",
                "repo_url": "repo",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    specs = load_query_specs(queries)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.branch == "main"
    assert spec.commit_id == ""


def test_build_dataset_from_raw_defaults_confidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "storage" / "dataset" / "run"
    raw_dir = run_dir / "raw_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "q-missing.jsonl"
    raw_file.write_text(
        json.dumps(
            {
                "chunk": {
                    "path": "src/demo.py",
                    "start_line": 1,
                    "end_line": 2,
                    "content": "print('demo')\npass",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = QuerySpec(
        query_id="q-missing",
        query="demo",
        repo_url="repo",
        branch="main",
        commit_id="head",
        path=None,
    )

    summary = build_dataset_from_raw(specs=[spec], run_dir=run_dir, run_name="run")
    assert summary.samples == 1
    assert summary.chunks == 1
    assert not summary.missing_queries
    assert "q-missing" not in summary.errors
    assert summary.dataset_path is not None
    payloads = [
        json.loads(line)
        for line in summary.dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert payloads[0]["golden_chunks"][0]["confidence"] == 0.0

def test_build_dataset_from_raw_dedup_prefers_higher_confidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "storage" / "dataset" / "run-high"
    raw_dir = run_dir / "raw_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "q-high.jsonl"
    raw_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk": {
                            "path": "src/demo.py",
                            "start_line": 10,
                            "end_line": 22,
                            "confidence": 0.6,
                            "content": "first",
                        }
                    }
                ),
                json.dumps(
                    {
                        "chunk": {
                            "path": "src/demo.py",
                            "start_line": 11,
                            "end_line": 20,
                            "confidence": 0.9,
                            "content": "second",
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = QuerySpec(
        query_id="q-high",
        query="demo",
        repo_url="repo",
        branch="main",
        commit_id="head",
        path=None,
    )

    summary = build_dataset_from_raw(specs=[spec], run_dir=run_dir, run_name="run-high")
    payloads = [
        json.loads(line)
        for line in summary.dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    chunks = payloads[0]["golden_chunks"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["confidence"] == 0.9
    assert chunk["start_line"] == 11
    assert chunk["end_line"] == 20


def test_build_dataset_from_raw_dedup_prefers_shorter_when_conf_equal(tmp_path: Path) -> None:
    run_dir = tmp_path / "storage" / "dataset" / "run-short"
    raw_dir = run_dir / "raw_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "q-short.jsonl"
    raw_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk": {
                            "path": "src/demo.py",
                            "start_line": 5,
                            "end_line": 25,
                            "confidence": 0.8,
                            "content": "long",
                        }
                    }
                ),
                json.dumps(
                    {
                        "chunk": {
                            "path": "src/demo.py",
                            "start_line": 10,
                            "end_line": 15,
                            "confidence": 0.8,
                            "content": "short",
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = QuerySpec(
        query_id="q-short",
        query="demo",
        repo_url="repo",
        branch="main",
        commit_id="head",
        path=None,
    )

    summary = build_dataset_from_raw(specs=[spec], run_dir=run_dir, run_name="run-short")
    payloads = [
        json.loads(line)
        for line in summary.dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    chunks = payloads[0]["golden_chunks"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["start_line"] == 10
    assert chunk["end_line"] == 15
    assert chunk["confidence"] == 0.8
