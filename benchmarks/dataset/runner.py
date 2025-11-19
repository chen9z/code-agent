from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from agent.prompts import DATASET_SYSTEM_PROMPT
from agent.session import CodeAgentSession
from config.config import get_config
from config.prompt import compose_system_prompt
from benchmarks.dataset.dataset_builder import DatasetBuilder, DatasetSample
from benchmarks.dataset.extractor import RawSampleExtractor
from benchmarks.dataset.snapshot_manager import SnapshotManager
from tools.dataset_log import DatasetLogTool, DatasetQueryContext
from tools.registry import ToolRegistry, create_default_registry

from benchmarks.dataset.models import DatasetRunResult, PreparedQuery, QuerySpec, RepoSpec


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

    recorded_samples = 0

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
        recorded_samples += 1

    for result in run_results:
        if not result.success:
            anomalies.append({"query_id": result.query_id, "error": result.message})

    if anomalies:
        with anomalies_path.open("a", encoding="utf-8") as handle:
            for row in anomalies:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    _print_summary(
        run_name=run_name,
        run_dir=run_dir,
        raw_dir=raw_dir,
        dataset_path=builder.dataset_path,
        anomalies_path=anomalies_path,
        total_queries=len(prepared_queries),
        agent_success=sum(1 for result in run_results if result.success),
        recorded_samples=recorded_samples,
        anomalies=anomalies,
    )


def _print_summary(
    *,
    run_name: str,
    run_dir: Path,
    raw_dir: Path,
    dataset_path: Path,
    anomalies_path: Path,
    total_queries: int,
    agent_success: int,
    recorded_samples: int,
    anomalies: List[dict],
) -> None:
    print("[数据集] 运行名称:", run_name)
    print("[数据集] 运行目录:", run_dir)
    print("[数据集] 原始样本目录:", raw_dir)
    dataset_status = "已生成" if dataset_path.exists() else "尚未生成"
    print(f"[数据集] 输出文件: {dataset_path} ({dataset_status})")
    print(
        f"[数据集] 查询总数 {total_queries}，Agent 执行成功 {agent_success} 个，生成样本 {recorded_samples} 个"
    )
    if anomalies:
        print(f"[数据集] 记录了 {len(anomalies)} 个异常，详情见 {anomalies_path}")
        for entry in anomalies:
            print(f"  - {entry.get('query_id')}: {entry.get('error')}")
    else:
        print("[数据集] 未记录异常")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset synthesis orchestrator")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL file")
    parser.add_argument("--run-name", dest="run_name", help="Override run name (default YYYYMMDD)")
    parser.add_argument("--artifacts-root", default="storage/dataset", help="Artifacts root directory")

    args = parser.parse_args(list(argv) if argv is not None else None)
    synthesize(args)
    return 0


class DatasetRunner:
    """Sequential runner that configures CodeAgentSession for dataset synthesis."""

    def __init__(
        self,
        *,
        llm_client=None,
        run_name: Optional[str] = None,
        artifacts_root: str | Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%d")
        self.artifacts_root = Path(artifacts_root or "storage/dataset").expanduser().resolve()
        cfg = get_config()
        self.tool_timeout_seconds = float(cfg.cli_tool_timeout_seconds)
        self.max_iterations = 6

    def run_queries(self, queries: Iterable[PreparedQuery]) -> List[DatasetRunResult]:
        results: List[DatasetRunResult] = []
        for prepared in queries:
            spec = prepared.spec
            context = DatasetQueryContext(
                query_id=spec.query_id,
                query=spec.query,
                repo_url=spec.repo.url,
                branch=spec.repo.branch,
                commit=spec.repo.commit,
                snapshot_path=prepared.snapshot_path,
            )
            workspace = prepared.snapshot_path
            session = self._build_session(context=context, workspace=workspace)
            try:
                session.run_turn(spec.query)
                results.append(DatasetRunResult(query_id=spec.query_id, success=True, message=None))
            except Exception as exc:  # pragma: no cover - 捕获 agent 异常
                results.append(DatasetRunResult(query_id=spec.query_id, success=False, message=str(exc)))
        return results

    # ------------------------------------------------------------------ internals
    def _build_session(
        self,
        *,
        context: DatasetQueryContext,
        workspace: Path,
    ) -> CodeAgentSession:
        registry = self._build_registry(context=context, workspace=workspace)
        environment = self._build_environment(context=context, workspace=workspace)
        system_prompt = compose_system_prompt(DATASET_SYSTEM_PROMPT, environment=environment)
        return CodeAgentSession(
            registry=registry,
            llm_client=self.llm_client,
            max_iterations=self.max_iterations,
            system_prompt=system_prompt,
            environment=environment,
            workspace=workspace,
            temperature=0.0,
            tool_timeout_seconds=self.tool_timeout_seconds,
            verbose=True,
        )

    def _build_registry(
        self,
        *,
        context: DatasetQueryContext,
        workspace: Path,
    ) -> ToolRegistry:
        registry = create_default_registry(
            include={"bash", "read", "grep", "glob", "todo_write","dataset_log_write_chunk"},
            project_root=workspace,
        )
        dataset_tool = DatasetLogTool(
            context=context,
            artifacts_root=self.artifacts_root,
            run_name=self.run_name,
        )
        registry.register(dataset_tool, name=dataset_tool.name)
        return registry

    def _build_environment(
        self,
        *,
        context: DatasetQueryContext,
        workspace: Path,
    ) -> Mapping[str, object]:
        return {
            "snapshot_root": str(workspace),
            "cwd": str(workspace),
            "query_id": context.query_id,
            "artifacts_root": str(self.artifacts_root),
        }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
