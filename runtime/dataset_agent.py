"""DatasetSynthesisAgent runtime orchestrator built on CodeAgentSession."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

from agent.prompts import DATASET_SYSTEM_PROMPT
from agent.session import CodeAgentSession
from adapters.llm.llm import BaseLLMClient, get_default_llm_client
from config.config import get_config
from config.prompt import compose_system_prompt
from tools.dataset_log import DatasetLogTool, DatasetQueryContext
from tools.registry import ToolRegistry, create_default_registry
from ui.emission import OutputCallback


class DatasetSynthesisAgent:
    """Thin adapter that configures CodeAgentSession for dataset synthesis."""

    def __init__(
        self,
        *,
        query_context: DatasetQueryContext,
        snapshot_root: str | Path,
        registry: Optional[ToolRegistry] = None,
        llm_client: Optional[BaseLLMClient] = None,
        max_iterations: int = 6,
        workspace: str | Path | None = None,
        run_name: Optional[str] = None,
        artifacts_root: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        self.snapshot_root = Path(snapshot_root).expanduser().resolve()
        self.query_context = query_context
        self.artifacts_root = Path(artifacts_root or "storage/dataset").expanduser().resolve()
        self.max_iterations = max(1, max_iterations)
        self.llm_client = llm_client or get_default_llm_client()
        cfg = get_config()

        self.registry = registry or self._build_registry(run_name=run_name)
        system_prompt = compose_system_prompt(DATASET_SYSTEM_PROMPT, environment=self._env())

        self.session = CodeAgentSession(
            registry=self.registry,
            llm_client=self.llm_client,
            max_iterations=self.max_iterations,
            system_prompt=system_prompt,
            environment=self._env(),
            workspace=self.workspace or self.snapshot_root,
            temperature=0.0,
            tool_timeout_seconds=float(cfg.cli_tool_timeout_seconds),
        )

    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[OutputCallback] = None,
    ) -> Dict[str, object]:
        return self.session.run_turn(user_input, output_callback=output_callback)

    def set_tool_timeout_seconds(self, seconds: Optional[float]) -> None:
        self.session.set_tool_timeout_seconds(seconds)

    # ------------------------------------------------------------------ internals
    def _build_registry(self, run_name: Optional[str]) -> ToolRegistry:
        registry = create_default_registry(
            include={"read", "grep", "glob", "codebase_search"},
            project_root=self.workspace or self.snapshot_root,
        )
        dataset_tool = DatasetLogTool(
            context=self.query_context,
            artifacts_root=self.artifacts_root,
            run_name=run_name,
        )
        registry.register(dataset_tool, name=dataset_tool.name)
        return registry

    def _env(self) -> Mapping[str, object]:
        env: Dict[str, object] = {
            "snapshot_root": str(self.snapshot_root),
            "query_id": self.query_context.query_id,
            "artifacts_root": str(self.artifacts_root),
        }
        if self.workspace:
            env["cwd"] = str(self.workspace)
        return env
