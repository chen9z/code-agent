from __future__ import annotations

import json
from pathlib import Path

from adapters.llm.llm import BaseLLMClient
from benchmarks.dataset.models import QuerySpec
from benchmarks.dataset.runner import DatasetRunner


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
        commit="abc123",
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
    chunk = rows[0]["chunk"]
    assert chunk["path"] == "src/demo.py"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 1
