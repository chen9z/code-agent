from __future__ import annotations

import json
import subprocess
from pathlib import Path

from adapters.llm.llm import BaseLLMClient
from benchmarks.dataset.models import QuerySpec
from benchmarks.dataset.runner import DatasetRunner, prepare_queries, build_dataset_from_raw
from benchmarks.dataset.snapshot_manager import SnapshotManager


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


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "add", "src/main.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
        .decode("utf-8")
        .strip()
    )

    repo_url = str(repo)
    specs = [
        QuerySpec(
            query_id="q1",
            query="print ok",
            repo_url=repo_url,
            branch="main",
            commit=commit,
            path=None,
        )
    ]

    manager = SnapshotManager(base_dir=tmp_path / "storage" / "dataset")
    prepared = prepare_queries(specs, manager=manager)

    artifacts_root = tmp_path / "storage" / "dataset"
    runner = DatasetRunner(llm_client=FakeLLMClient(
        {
            "path": "src/main.py",
            "start_line": 1,
            "end_line": 1,
            "confidence": 0.9,
        }
    ), run_name="run", artifacts_root=artifacts_root)
    results = runner.run_queries(prepared)
    assert results[0].success

    raw_dir = artifacts_root / "run" / "raw_samples"
    assert raw_dir.exists()
    raw_file = raw_dir / "q1.jsonl"
    rows = [json.loads(line) for line in raw_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"
    run_dir = artifacts_root / "run"
    summary = build_dataset_from_raw(specs=specs, run_dir=run_dir, run_name="run")
    assert summary.samples == 1
    assert summary.chunks == 1
    assert not summary.missing_queries
    assert summary.dataset_path is not None
    payloads = [
        json.loads(line)
        for line in summary.dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    record = payloads[0]
    assert record["query_id"] == "q1"
    assert record["repo_url"] == repo_url
    assert record["branch"] == "main"
    assert record["commit"] == commit
    assert "schema_version" not in record
    assert "repo" not in record
    assert len(record["golden_chunks"]) == 1
