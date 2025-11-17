from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.llm.llm import BaseLLMClient
from runtime.dataset_agent import DatasetSynthesisAgent
from tools.dataset_log import DatasetQueryContext


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


def test_agent_triggers_dataset_log(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "demo.py").write_text("print('demo')\n", encoding="utf-8")

    ctx = DatasetQueryContext(
        query_id="q-demo",
        query="print demo",
        repo_url="repo-url",
        branch="main",
        commit="abc123",
        snapshot_path=repo,
    )
    llm = FakeLLMClient(
        {
            "path": "src/demo.py",
            "start_line": 1,
            "end_line": 1,
            "confidence": 0.95,
        }
    )
    run_dir = tmp_path / "storage" / "dataset"
    agent = DatasetSynthesisAgent(
        query_context=ctx,
        snapshot_root=repo,
        llm_client=llm,
        workspace=repo,
    run_name="run1",
    artifacts_root=run_dir,
    )

    agent.run_turn("print demo")

    raw_file = run_dir / "run1" / "raw_samples" / "q-demo.jsonl"
    assert raw_file.exists()
    payload = raw_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(payload) == 1
    row = json.loads(payload[0])
    assert row["chunk"]["path"] == "src/demo.py"
