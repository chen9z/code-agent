from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from agent.prompts import DATASET_SYSTEM_PROMPT
from agent.session import CodeAgentSession
from config.config import get_config
from config.prompt import compose_system_prompt
from tools.dataset_log import DatasetLogTool, DatasetQueryContext
from tools.registry import ToolRegistry, create_default_registry

from .models import DatasetRunResult, PreparedQuery


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
        )

    def _build_registry(
        self,
        *,
        context: DatasetQueryContext,
        workspace: Path,
    ) -> ToolRegistry:
        registry = create_default_registry(
            include={"bash", "read", "grep", "glob", "todo_write"},
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
