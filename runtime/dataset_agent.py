"""DatasetSynthesisAgent runtime orchestrator.

该模块遵循 docs/dataset_synthesis_plan.md 的最小实现：
- 仅暴露受限工具集（read/grep/glob/codebase_search + dataset_log）
- 温度固定 0，尽量确定性；不写长段落回复
- 由 orchestrator 注入快照/查询上下文
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agent.prompts import DATASET_SYSTEM_PROMPT
from adapters.llm.llm import BaseLLMClient, get_default_llm_client
from config.config import get_config
from config.prompt import compose_system_prompt
from runtime.tool_runner import ToolExecutionRunner, ToolResult
from tools.dataset_log import DatasetLogTool, DatasetQueryContext
from tools.registry import ToolRegistry, create_default_registry
from ui.emission import OutputCallback, OutputMessage, create_emit_event


def _emit(callback: Optional[OutputCallback], message: OutputMessage) -> None:
    if callable(callback):
        callback(message)


def _prepare_messages(
    history: Optional[Iterable[Mapping[str, Any]]],
    system_prompt: str,
    user_input: str,
) -> List[Dict[str, Any]]:
    cloned = [dict(item) for item in (history or [])]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(msg for msg in cloned if msg.get("role") != "system")
    if user_input:
        messages.append({"role": "user", "content": user_input})
    return messages


class DatasetSynthesisAgent:
    """局部对话会话，驱动 DatasetSynthesisAgent 提交 golden chunk。"""

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
        cfg = get_config()
        self.model = cfg.llm_model
        # Dataset agent 强制温度 0
        self.temperature = 0.0
        self.max_iterations = max(1, max_iterations)
        self.llm_client = llm_client or get_default_llm_client()
        self.registry = registry or self._build_registry(run_name=run_name)
        self.system_prompt = compose_system_prompt(DATASET_SYSTEM_PROMPT, environment=self._env())
        self.executor = ToolExecutionRunner(self.registry, default_timeout_seconds=float(cfg.cli_tool_timeout_seconds))
        self.messages: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ public
    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[OutputCallback] = None,
    ) -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty")
        messages = _prepare_messages(self.messages, self.system_prompt, user_input)
        _emit(output_callback, create_emit_event("user", user_input))

        tool_results: List[ToolResult] = []
        iterations = 0
        final_content: Optional[str] = None

        while True:
            response = self._call_llm(messages)
            assistant_content = response.get("content")
            if assistant_content:
                _emit(output_callback, create_emit_event("assistant", assistant_content))
            calls = response.get("tool_calls") or []
            if not calls:
                final_content = assistant_content
                break
            outputs = self.executor.run(
                calls,
                messages=messages,
                output_callback=output_callback,
            )
            tool_results.extend(outputs)
            iterations += 1
            if iterations >= self.max_iterations:
                break

        result = {"content": final_content, "tool_results": list(tool_results), "messages": list(messages)}
        self.messages = [m for m in messages if self._is_valid_message(m)]
        return result

    def set_tool_timeout_seconds(self, seconds: Optional[float]) -> None:
        if seconds is None or seconds <= 0:
            return
        self.executor.set_default_timeout(float(seconds))

    # ------------------------------------------------------------------ internals
    def _build_registry(self, run_name: Optional[str]) -> ToolRegistry:
        # 限定工具集
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

    def _env(self) -> Mapping[str, Any]:
        env: Dict[str, Any] = {
            "snapshot_root": str(self.snapshot_root),
            "query_id": self.query_context.query_id,
            "artifacts_root": str(self.artifacts_root),
        }
        if self.workspace:
            env["cwd"] = str(self.workspace)
        return env

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        tools = self.registry.to_openai_tools()
        response = self.llm_client.create_with_tools(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.temperature,
        )
        plan = self._parse_tool_response(response)
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": plan.get("content")}
        tool_calls = plan.get("tool_calls") or []
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call.get("name"),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for call in tool_calls
                if call.get("name")
            ]
        messages.append(assistant_message)
        return plan

    @staticmethod
    def _parse_tool_response(response: Any) -> Dict[str, Any]:
        message = DatasetSynthesisAgent._extract_message(response)
        if not message:
            return {"tool_calls": [], "content": None}

        tool_calls = DatasetSynthesisAgent._extract_tool_calls(message)
        content_text = DatasetSynthesisAgent._message_content_to_str(
            DatasetSynthesisAgent._get_attr(message, "content")
        )

        if tool_calls:
            normalized: List[Dict[str, Any]] = []
            for idx, call in enumerate(tool_calls):
                fn = DatasetSynthesisAgent._get_attr(call, "function", {})
                name = DatasetSynthesisAgent._get_attr(fn, "name")
                arguments_raw = DatasetSynthesisAgent._get_attr(fn, "arguments", "{}")
                try:
                    arguments = json.loads(arguments_raw) if arguments_raw else {}
                except json.JSONDecodeError:
                    arguments = {}
                normalized.append(
                    {
                        "id": DatasetSynthesisAgent._get_attr(call, "id", f"call-{idx}"),
                        "name": str(name) if name else "",
                        "arguments": arguments,
                    }
                )
            return {"tool_calls": [c for c in normalized if c["name"]], "content": content_text}

        return {"tool_calls": [], "content": content_text}

    @staticmethod
    def _extract_message(response: Any) -> Any:
        if response is None:
            return None
        if hasattr(response, "choices"):
            choices = getattr(response, "choices", [])
            if choices:
                return getattr(choices[0], "message", None)
        if isinstance(response, dict):
            choices = response.get("choices")
            if choices:
                message = choices[0].get("message")
                if message is not None:
                    return message
        return None

    @staticmethod
    def _extract_tool_calls(message: Any) -> List[Any]:
        calls = DatasetSynthesisAgent._get_attr(message, "tool_calls")
        if calls is None:
            return []
        return calls if isinstance(calls, list) else []

    @staticmethod
    def _message_content_to_str(content: Any) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    @staticmethod
    def _is_valid_message(raw: Any) -> bool:
        return isinstance(raw, dict) and isinstance(raw.get("role"), str) and "content" in raw
