from __future__ import annotations

import json
from pathlib import Path

from adapters.llm.llm import BaseLLMClient
from benchmarks.dataset.cli import prepare_queries
from benchmarks.dataset.dataset_builder import DatasetBuilder, DatasetSample
from benchmarks.dataset.extractor import RawSampleExtractor
from benchmarks.dataset.models import QuerySpec, RepoSpec
from benchmarks.dataset.runner import DatasetRunner
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
                                        "name": "dataset_log.write_chunk",
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
            repo=RepoSpec(url="repo-url", branch="main", commit="demo", path=str(repo)),
        )
    ]

    manager = SnapshotManager(base_dir=tmp_path / "artifacts")
    prepared = prepare_queries(specs, manager=manager)

    artifacts_root = tmp_path / "artifacts"
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
    extractor = RawSampleExtractor(raw_dir=raw_dir)
    extraction = extractor.extract("q1")
    assert extraction.chunks

    builder = DatasetBuilder(output_dir=artifacts_root / "run" / "datasets", run_name="run")
    builder.append(
        DatasetSample(
            query_id="q1",
            query="print ok",
            repo_url="repo-url",
            commit="demo",
            golden_chunks=extraction.chunks,
        )
    )
    dataset_path = artifacts_root / "run" / "datasets" / "dataset_run.jsonl"
    assert dataset_path.exists()
    content = dataset_path.read_text(encoding="utf-8").strip()
    assert "print('ok')" in content
