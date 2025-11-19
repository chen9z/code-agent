from __future__ import annotations

import json
from pathlib import Path

from adapters.llm.llm import BaseLLMClient
from benchmarks.dataset.models import QuerySpec
from benchmarks.dataset.runner import DatasetRunner, prepare_queries
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

    specs = [
        QuerySpec(
            query_id="q1",
            query="print ok",
            repo_url="repo-url",
            branch="main",
            commit="demo",
            path=str(repo),
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
    assert not raw_dir.exists()
    dataset_path = artifacts_root / "run" / "datasets"
    assert not dataset_path.exists()
