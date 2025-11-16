#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from benchmarks.dataset.dataset_builder import DatasetBuilder, DatasetSample
from benchmarks.dataset.extractor import RawSampleExtractor
from benchmarks.dataset.models import PreparedQuery, QuerySpec, RepoSpec
from benchmarks.dataset.runner import DatasetRunner
from benchmarks.dataset.snapshot_manager import SnapshotManager


def load_query_specs(path: Path) -> List[QuerySpec]:
    entries: List[QuerySpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            repo = payload.get("repo") or {}
            repo_path = repo.get("path") or repo.get("local_path") or payload.get("repo_path")
            spec = QuerySpec(
                query_id=str(payload["query_id"]),
                query=payload["query"],
                repo=RepoSpec(
                    url=repo.get("url") or repo_path or "",
                    branch=repo.get("branch", "main"),
                    commit=repo.get("commit", "working"),
                    path=repo_path,
                ),
            )
            entries.append(spec)
    return entries


def prepare_queries(specs: Iterable[QuerySpec], *, manager: SnapshotManager) -> List[PreparedQuery]:
    prepared: List[PreparedQuery] = []
    for spec in specs:
        repo = spec.repo
        metadata = manager.materialize(
            repo_url=repo.url or repo.path or "repo",
            branch=repo.branch,
            commit=repo.commit,
            source_path=repo.path,
        )
        prepared.append(
            PreparedQuery(
                spec=spec,
                snapshot_path=Path(metadata.snapshot_path),
            )
        )
    return prepared


def synthesize(args: argparse.Namespace) -> None:
    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d")
    run_dir = artifacts_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(Path(args.queries).expanduser().resolve())
    manager = SnapshotManager(base_dir=artifacts_root)
    prepared_queries = prepare_queries(specs, manager=manager)

    runner = DatasetRunner(run_name=run_name, artifacts_root=artifacts_root)
    run_results = runner.run_queries(prepared_queries)

    raw_dir = artifacts_root / run_name / "raw_samples"
    extractor = RawSampleExtractor(raw_dir=raw_dir)
    dataset_dir = artifacts_root / run_name / "datasets"
    builder = DatasetBuilder(output_dir=dataset_dir, run_name=run_name)

    anomalies_path = run_dir / "anomalies.jsonl"
    anomalies: List[dict] = []

    for prepared in prepared_queries:
        spec = prepared.spec
        extraction = extractor.extract(spec.query_id)
        if not extraction.chunks:
            anomalies.append({"query_id": spec.query_id, "error": "no_chunks"})
            continue
        sample = DatasetSample(
            query_id=spec.query_id,
            query=spec.query,
            repo_url=spec.repo.url,
            commit=spec.repo.commit,
            golden_chunks=extraction.chunks,
        )
        builder.append(sample)

    for result in run_results:
        if not result.success:
            anomalies.append({"query_id": result.query_id, "error": result.message})

    if anomalies:
        with anomalies_path.open("a", encoding="utf-8") as handle:
            for row in anomalies:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset synthesis orchestrator")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL file")
    parser.add_argument("--run-name", dest="run_name", help="Override run name (default YYYYMMDD)")
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifacts root directory")

    args = parser.parse_args(list(argv) if argv is not None else None)
    synthesize(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
