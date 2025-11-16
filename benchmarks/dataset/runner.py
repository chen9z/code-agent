from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from runtime.dataset_agent import DatasetSynthesisAgent
from tools.dataset_log import DatasetQueryContext

from .models import DatasetRunResult, PreparedQuery


class DatasetRunner:
    """Minimal sequential runner for DatasetSynthesisAgent."""

    def __init__(
        self,
        *,
        llm_client=None,
        run_name: Optional[str] = None,
        artifacts_root: str | Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%d")
        self.artifacts_root = Path(artifacts_root or "artifacts").expanduser().resolve()

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
            agent = DatasetSynthesisAgent(
                query_context=context,
                snapshot_root=prepared.snapshot_path,
                llm_client=self.llm_client,
                workspace=prepared.snapshot_path,
                run_name=self.run_name,
                artifacts_root=self.artifacts_root,
            )
            try:
                agent.run_turn(spec.query)
                results.append(DatasetRunResult(query_id=spec.query_id, success=True, message=None))
            except Exception as exc:  # pragma: no cover - 捕获 agent 异常
                results.append(DatasetRunResult(query_id=spec.query_id, success=False, message=str(exc)))
        return results
