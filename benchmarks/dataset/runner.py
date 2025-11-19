from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from agent.prompts import DATASET_SYSTEM_PROMPT
from agent.session import CodeAgentSession
from config.config import get_config
from config.prompt import compose_system_prompt
from benchmarks.dataset.snapshot_manager import SnapshotManager
from tools.dataset_log import DatasetLogTool, DatasetQueryContext
from tools.registry import ToolRegistry, create_default_registry

from benchmarks.dataset.models import DatasetRunResult, QuerySpec


@dataclass
class DatasetAggregateSummary:
    dataset_path: Optional[Path]
    samples: int
    chunks: int
    missing_queries: List[str] = field(default_factory=list)
    errors: Dict[str, List[str]] = field(default_factory=dict)


def load_query_specs(path: Path) -> List[QuerySpec]:
    entries: List[QuerySpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            repo_url = payload.get("repo_url")
            if not repo_url:
                raise ValueError(f"repo_url missing for query_id {payload.get('query_id')}")
            branch = payload.get("branch") or "main"
            commit = payload.get("commit") or "working"
            spec = QuerySpec(
                query_id=payload["query_id"],
                query=payload["query"],
                repo_url=repo_url,
                branch=branch,
                commit=commit,
                path=None,
            )
            entries.append(spec)
    return entries


def prepare_queries(specs: Iterable[QuerySpec], *, manager: SnapshotManager) -> List[tuple[QuerySpec, Path]]:
    prepared: List[tuple[QuerySpec, Path]] = []
    for spec in specs:
        metadata = manager.materialize(
            repo_url=spec.repo_url or spec.path or "repo",
            branch=spec.branch,
            commit=spec.commit,
            source_path=spec.path,
        )
        prepared.append((spec, Path(metadata.snapshot_path)))
    return prepared


def build_dataset_from_raw(
    *,
    specs: Iterable[QuerySpec],
    run_dir: Path,
    run_name: str,
) -> DatasetAggregateSummary:
    spec_entries = list(specs)
    raw_dir = run_dir / "raw_samples"
    dataset_dir = run_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        return DatasetAggregateSummary(
            dataset_path=None,
            samples=0,
            chunks=0,
            missing_queries=[spec.query_id for spec in spec_entries],
            errors={"__runner__": ["raw_samples directory missing"]},
        )

    dataset_path = dataset_dir / f"dataset_{run_name}.jsonl"
    if dataset_path.exists():
        dataset_path.unlink()

    missing: List[str] = []
    errors: Dict[str, List[str]] = {}
    total_samples = 0
    total_chunks = 0

    for spec in spec_entries:
        chunks, chunk_errors = _extract_chunks(raw_dir=raw_dir, query_id=spec.query_id)
        if chunk_errors:
            errors[spec.query_id] = chunk_errors
        if not chunks:
            missing.append(spec.query_id)
            continue

        record = {
            "query_id": spec.query_id,
            "query": spec.query,
            "repo_url": spec.repo_url,
            "branch": spec.branch,
            "commit": spec.commit,
            "golden_chunks": chunks,
        }
        with dataset_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
        total_samples += 1
        total_chunks += len(chunks)

    if total_samples == 0 and dataset_path.exists():
        dataset_path.unlink()
        dataset_path = None

    return DatasetAggregateSummary(
        dataset_path=dataset_path,
        samples=total_samples,
        chunks=total_chunks,
        missing_queries=missing,
        errors=errors,
    )


def _extract_chunks(*, raw_dir: Path, query_id: str) -> Tuple[List[Dict[str, object]], List[str]]:
    raw_file = raw_dir / f"{query_id}.jsonl"
    if not raw_file.exists():
        return [], ["raw sample missing"]

    seen: set[Tuple[str, int, int]] = set()
    chunks: List[Dict[str, object]] = []
    errors: List[str] = []

    with raw_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                errors.append("json decode error")
                continue

            chunk = payload.get("chunk") or {}
            path = chunk.get("path", "")
            start_line = int(chunk.get("start_line", 0))
            end_line = int(chunk.get("end_line", 0))
            if not path or start_line <= 0 or end_line < start_line:
                continue

            key = (path, start_line, end_line)
            if key in seen:
                continue
            seen.add(key)

            content = chunk.get("content", "")
            chunks.append(
                {
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "confidence": float(chunk.get("confidence", 0)),
                    "content": content,
                    "content_hash": sha1(content.encode("utf-8")).hexdigest(),
                }
            )

    return chunks, errors


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

    anomalies_path = run_dir / "anomalies.jsonl"
    anomalies: List[dict] = []

    for result in run_results:
        if not result.success:
            anomalies.append({"query_id": result.query_id, "error": result.message})

    if anomalies:
        with anomalies_path.open("a", encoding="utf-8") as handle:
            for row in anomalies:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    dataset_summary = build_dataset_from_raw(
        specs=specs,
        run_dir=run_dir,
        run_name=run_name,
    )

    _print_summary(
        run_name=run_name,
        run_dir=run_dir,
        anomalies_path=anomalies_path,
        total_queries=len(prepared_queries),
        agent_success=sum(1 for result in run_results if result.success),
        anomalies=anomalies,
        dataset_summary=dataset_summary,
    )


def _print_summary(
    *,
    run_name: str,
    run_dir: Path,
    anomalies_path: Path,
    total_queries: int,
    agent_success: int,
    anomalies: List[dict],
    dataset_summary: DatasetAggregateSummary,
) -> None:
    print("[数据集] 运行名称:", run_name)
    print("[数据集] 运行目录:", run_dir)
    print(f"[数据集] raw_samples 输出目录: {run_dir / 'raw_samples'}")
    print(f"[数据集] 查询总数 {total_queries}，Agent 执行成功 {agent_success} 个")
    failed = total_queries - agent_success
    if failed:
        print(f"[数据集] 失败 {failed} 个，详情见 {anomalies_path}")
    else:
        print("[数据集] 未记录异常")
    for entry in anomalies:
        print(f"  - {entry.get('query_id')}: {entry.get('error')}")

    if dataset_summary.dataset_path:
        print(
            f"[数据集] 聚合输出: {dataset_summary.dataset_path} "
            f"(samples={dataset_summary.samples}, chunks={dataset_summary.chunks})"
        )
    else:
        print("[数据集] 未生成聚合数据文件")

    if dataset_summary.missing_queries:
        preview = ", ".join(dataset_summary.missing_queries[:5])
        suffix = "..." if len(dataset_summary.missing_queries) > 5 else ""
        print(f"  - 无 chunk 的 query: {preview}{suffix}")

    if dataset_summary.errors:
        for query_id, messages in dataset_summary.errors.items():
            joined = "; ".join(messages)
            print(f"  - {query_id}: {joined}")


@contextmanager
def _workspace_context(target: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(original)


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
        self.run_dir = self.artifacts_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg = get_config()
        self.tool_timeout_seconds = float(cfg.cli_tool_timeout_seconds)
        self.max_iterations = 50

    def run_queries(self, queries: Iterable[tuple[QuerySpec, Path]]) -> List[DatasetRunResult]:
        results: List[DatasetRunResult] = []
        for spec, snapshot_path in queries:
            context = DatasetQueryContext(
                query_id=spec.query_id,
                query=spec.query,
                repo_url=spec.repo_url,
                branch=spec.branch,
                commit=spec.commit,
                snapshot_path=snapshot_path,
            )
            workspace = snapshot_path
            session = self._build_session(context=context, workspace=workspace)
            try:
                with _workspace_context(workspace):
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
        environment = self._build_environment(context=context)
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
    ) -> Mapping[str, object]:
        snapshot_root = str(context.snapshot_path)
        return {
            "snapshot_root": snapshot_root,
            "cwd": snapshot_root,
            "query_id": context.query_id,
            "artifacts_root": str(self.artifacts_root),
        }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
