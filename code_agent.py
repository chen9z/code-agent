"""Code Agent flow and CLI orchestrator built on the Flow/Node runtime."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from __init__ import Flow, Node
from cli.code_agent_cli import (
    CodeAgentCLIFlow,
    run_cli_main as _run_cli_main,
    run_code_agent_cli as _run_code_agent_cli,
    run_code_agent_once as _run_code_agent_once,
)
from cli.rich_output import create_rich_output, preview_payload as _preview_payload, stringify_payload as _stringify_payload
from clients.llm import get_default_llm_client
from configs.manager import get_config
from core.prompt import (
    SECURITY_SYSTEM_PROMPT,
    _BASE_SYSTEM_PROMPT,
    _SUMMARY_INSTRUCTIONS,
    compose_system_prompt,
)
from nodes.tool_execution import ToolExecutionBatchNode, ToolOutput
from tools.registry import ToolRegistry, create_default_registry

if TYPE_CHECKING:
    from rich.console import Console


def build_code_agent_system_prompt(
    *,
    base_prompt: str = _BASE_SYSTEM_PROMPT,
    extra_sections: Optional[Sequence[str]] = None,
    environment: Optional[Mapping[str, Any]] = None,
    include_security_prompt: bool = True,
) -> str:
    """Compose the system prompt using the shared helper."""

    sections: List[str] = []
    if include_security_prompt:
        sections.append(SECURITY_SYSTEM_PROMPT)
    if extra_sections:
        sections.extend(extra_sections)
    return compose_system_prompt(base_prompt, extra_sections=sections, environment=environment)


def _emit(shared: Dict[str, Any], message: str) -> None:
    callback = shared.get("output_callback")
    if callable(callback):
        callback(message)


class ConversationContextNode(Node):
    """Initialises conversation history and appends the latest user input."""

    def __init__(self, system_prompt: str = _BASE_SYSTEM_PROMPT) -> None:
        super().__init__()
        self.system_prompt = system_prompt

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        history = list(shared.get("history") or self.params.get("history") or [])
        if not history:
            history = [{"role": "system", "content": self.system_prompt}]
        elif history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": self.system_prompt})

        user_input = self.params.get("user_input") or shared.get("user_input")
        if user_input:
            history.append({"role": "user", "content": str(user_input)})
        # Drop any additional 'system' messages beyond the first to reduce prompt injection surface.
        if history:
            filtered = [history[0]] + [m for m in history[1:] if m.get("role") != "system"]
        else:
            filtered = history
        shared["history"] = filtered
        return {"has_user_input": bool(user_input)}

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> str:
        user_input = shared.get("user_input")
        if user_input:
            _emit(shared, f"[user] {user_input}")
        return "plan"


class ToolPlanningNode(Node):
    """Uses the LLM to produce a tool execution plan."""

    def __init__(self, registry: ToolRegistry, model: str, llm_client=None) -> None:
        super().__init__()
        self.registry = registry
        self.model = model
        self.llm = llm_client or get_default_llm_client()

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        descriptors = self.registry.describe()
        history = list(shared.get("history") or [])
        return {"history": history, "descriptors": descriptors}

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        history = prep_res["history"]
        descriptors = [self._strip_explanation_from_descriptor(d) for d in prep_res["descriptors"]]
        serialized_tools = json.dumps(descriptors, ensure_ascii=False, indent=2)
        planning_prompt = "Available tools:\n" + serialized_tools
        messages = history + [{"role": "system", "content": planning_prompt}]
        tools = [self._strip_explanation_from_tool(tool) for tool in self.registry.to_openai_tools()]

        response = self.llm.create_with_tools(
            model=self.model,
            messages=messages,
            tools=tools,
            parallel_tool_calls=True,
        )
        plan = self._parse_tool_response(response)
        plan["raw_text"] = self._extract_raw_response(response)
        return plan

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        shared["tool_plan"] = {
            "tool_calls": exec_res.get("tool_calls", []),
            "final_response": exec_res.get("final_response"),
            "thoughts": exec_res.get("thoughts"),
            "raw_text": exec_res.get("raw_text"),
        }
        history = shared.setdefault("history", [])
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": exec_res.get("thoughts")
            or f"Tool plan: {json.dumps(exec_res.get('tool_calls', []), ensure_ascii=False)}",
        }

        tool_calls = exec_res.get("tool_calls") or []
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call.get("key", ""),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for call in tool_calls
                if call.get("key")
            ]
        history.append(assistant_message)
        thoughts = assistant_message.get("content")
        if thoughts:
            _emit(shared, f"[assistant:planner] {thoughts}")
        for call in tool_calls:
            name = call.get("key")
            if not name:
                continue
            args_preview = _preview_payload(call.get("arguments") or {}, 180)
            message_body = name
            if args_preview and args_preview not in {"{}", "null"}:
                message_body += f" | args: {args_preview}"
            _emit(shared, f"[plan] {message_body}")
        return "execute" if tool_calls else "summarize"

    def _parse_tool_response(self, response: Any) -> Dict[str, Any]:
        message = self._extract_message(response)
        if not message:
            return {"tool_calls": [], "final_response": None, "thoughts": None}

        tool_calls = self._extract_tool_calls(message)
        content_text = self._message_content_to_str(self._get_attr(message, "content"))

        if tool_calls:
            normalized_calls: List[Dict[str, Any]] = []
            for idx, call in enumerate(tool_calls):
                function = self._get_attr(call, "function", {})
                name = self._get_attr(function, "name")
                arguments_str = self._get_attr(function, "arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    arguments = {}
                normalized_calls.append(
                    {
                        "id": self._get_attr(call, "id", f"call-{idx}"),
                        "key": str(name) if name else "",
                        "arguments": arguments,
                    }
                )
            return {
                "tool_calls": [c for c in normalized_calls if c["key"]],
                "final_response": None,
                "thoughts": content_text,
            }

        return {
            "tool_calls": [],
            "final_response": content_text,
            "thoughts": content_text,
        }

    # JSON-plan fallback removed

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
        tool_calls = ToolPlanningNode._get_attr(message, "tool_calls")
        if tool_calls is None:
            return []
        if isinstance(tool_calls, list):
            return tool_calls
        return []

    @staticmethod
    def _message_content_to_str(content: Any) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content)

    @staticmethod
    def _extract_raw_response(response: Any) -> str:
        try:
            if hasattr(response, "model_dump_json"):
                return response.model_dump_json()
            if hasattr(response, "model_dump"):
                return json.dumps(response.model_dump(), ensure_ascii=False)
            if isinstance(response, dict):
                return json.dumps(response, ensure_ascii=False)
            return str(response)
        except Exception:  # pragma: no cover - diagnostics only
            return str(response)

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    @classmethod
    def _strip_explanation_from_descriptor(cls, descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(descriptor)

    @classmethod
    def _strip_explanation_from_tool(cls, tool: Mapping[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(tool)


class SummaryNode(Node):
    """Creates the final natural language response using the LLM."""

    def __init__(self, model: str, llm_client=None, system_prompt: str = _BASE_SYSTEM_PROMPT) -> None:
        super().__init__()
        self.model = model
        self.llm = llm_client or get_default_llm_client()
        self.system_prompt = system_prompt

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "history": list(shared.get("history") or []),
            "tool_results": list(shared.get("tool_results") or []),
            "tool_plan": shared.get("tool_plan") or {},
        }

    def exec(self, prep_res: Dict[str, Any]) -> str:
        history = self._trim_history(prep_res["history"])
        tool_results = prep_res["tool_results"]
        plan = prep_res["tool_plan"]

        results_summary_lines: List[str] = []
        for result in tool_results:
            if isinstance(result, ToolOutput):
                key = result.key
                if result.error:
                    results_summary_lines.append(
                        f"Tool {key} failed with error: {result.error}"
                    )
                else:
                    payload = result.result.data if result.result else None
                    results_summary_lines.append(
                        f"Tool {key} succeeded with result: {_preview_payload(payload, 400)}"
                    )
                continue

            key = result.get("key") if isinstance(result, dict) else None
            error_payload = result.get("error") if isinstance(result, dict) else None
            if error_payload:
                results_summary_lines.append(
                    f"Tool {key} failed with error: {error_payload}"
                )
            else:
                payload = result.get("result") if isinstance(result, dict) else result
                results_summary_lines.append(
                    f"Tool {key} succeeded with result: {_preview_payload(payload, 400)}"
                )
        if not results_summary_lines:
            results_summary_lines.append("No tool results were generated.")

        plan_final = plan.get("final_response")
        summary_prompt = (
            f"{_SUMMARY_INSTRUCTIONS}\n\n"
            f"Latest user request:\n{self._latest_user_content(history)}\n\n"
            f"Tool outcomes:\n- " + "\n- ".join(results_summary_lines)
        )
        if plan_final:
            summary_prompt += f"\n\nPlanner suggestion:\n{plan_final}"

        messages = self._build_messages(history, summary_prompt)
        chunks = list(self.llm.get_response(model=self.model, messages=messages, stream=False))
        return "".join(chunks).strip()

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: str) -> str:
        shared["final_response"] = exec_res
        shared.setdefault("history", []).append({"role": "assistant", "content": exec_res})
        if exec_res:
            _emit(shared, f"[assistant] {exec_res}")
        return "complete"

    @staticmethod
    def _latest_user_content(history: List[Dict[str, Any]]) -> str:
        for message in reversed(history):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return "(no direct user input captured)"

    @staticmethod
    def _trim_history(history: List[Dict[str, Any]], *, max_chars: int = 6000, max_messages: int = 12) -> List[Dict[str, Any]]:
        trimmed: List[Dict[str, Any]] = []
        remaining_chars = max_chars
        for message in reversed(history):
            content = message.get("content", "")
            content_str = _stringify_payload(content)
            remaining_chars -= len(content_str)
            trimmed.append(message)
            if len(trimmed) >= max_messages or remaining_chars <= 0:
                break
        trimmed.reverse()
        return SummaryNode._ensure_history_coherence(history, trimmed)

    @staticmethod
    def _ensure_history_coherence(
        full_history: List[Dict[str, Any]], trimmed: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        coherent = list(trimmed)

        # Always ensure the latest user message is present.
        latest_user = None
        for message in reversed(full_history):
            if message.get("role") == "user":
                latest_user = message
                break
        if latest_user and latest_user not in coherent:
            coherent.insert(0, latest_user)

        # Ensure each tool message is preceded by the assistant tool-call declaration.
        idx = 0
        while idx < len(coherent):
            message = coherent[idx]
            if message.get("role") == "tool":
                if not SummaryNode._has_preceding_tool_parent(coherent, idx, message.get("tool_call_id")):
                    parent = SummaryNode._find_tool_parent(full_history, message.get("tool_call_id"))
                    if parent and parent not in coherent:
                        coherent.insert(idx, parent)
                        idx += 1
                        continue
            idx += 1

        return coherent

    @staticmethod
    def _has_preceding_tool_parent(messages: List[Dict[str, Any]], idx: int, tool_call_id: Any) -> bool:
        if idx <= 0:
            return False
        parent = messages[idx - 1]
        if parent.get("role") != "assistant":
            return False
        for call in parent.get("tool_calls") or []:
            if call.get("id") == tool_call_id:
                return True
        return False

    @staticmethod
    def _find_tool_parent(full_history: List[Dict[str, Any]], tool_call_id: Any) -> Optional[Dict[str, Any]]:
        if not tool_call_id:
            return None
        for message in reversed(full_history):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if call.get("id") == tool_call_id:
                    return message
        return None

    def _build_messages(self, history: List[Dict[str, Any]], summary_prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for message in history:
            if message.get("role") == "system":
                continue
            messages.append(message)
        messages.append({"role": "user", "content": summary_prompt})
        return messages


class ToolAgentFlow(Flow):
    """End-to-end agent flow orchestrating tool planning, execution, and summarisation."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        llm_client=None,
        model: Optional[str] = None,
        *,
        max_iterations: int = 25,
        system_prompt: Optional[str] = None,
    ) -> None:
        if model is None:
            cfg = get_config()
            model = cfg.llm.model
        self.registry = registry or create_default_registry()
        self.llm = llm_client or get_default_llm_client()
        self.model = model
        self.max_iterations = max_iterations if max_iterations >= 1 else 1

        super().__init__()
        resolved_prompt = system_prompt or _BASE_SYSTEM_PROMPT
        self.context_node = ConversationContextNode(system_prompt=resolved_prompt)
        self.planning_node = ToolPlanningNode(self.registry, model=self.model, llm_client=self.llm)
        self.execution_node = ToolExecutionBatchNode(self.registry)
        self.summary_node = SummaryNode(model=self.model, llm_client=self.llm, system_prompt=resolved_prompt)

        self.execution_node.max_iterations = self.max_iterations

        self.start(self.context_node)
        self.context_node.next(self.planning_node, "plan")
        self.planning_node.next(self.execution_node, "execute")
        self.planning_node.next(self.summary_node, "summarize")
        self.execution_node.next(self.planning_node, "plan")
        self.execution_node.next(self.summary_node, "summarize")

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        history = self.params.get("history") or shared.get("history")
        if history:
            shared["history"] = list(history)
        user_input = self.params.get("user_input") or shared.get("user_input")
        if not user_input:
            raise ValueError("user_input is required for the tool agent flow")
        shared["user_input"] = user_input
        output_callback = self.params.get("output_callback") or shared.get("output_callback")
        if output_callback is not None:
            shared["output_callback"] = output_callback
        shared["tool_iterations_used"] = 0
        return {}

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> Dict[str, Any]:
        return {
            "final_response": shared.get("final_response"),
            "tool_results": shared.get("tool_results", []),
            "tool_plan": shared.get("tool_plan"),
            "history": shared.get("history", []),
        }


def create_tool_agent_flow(
    registry: Optional[ToolRegistry] = None,
    llm_client=None,
    model: Optional[str] = None,
    *,
    max_iterations: int = 25,
    system_prompt: Optional[str] = None,
) -> ToolAgentFlow:
    return ToolAgentFlow(
        registry=registry,
        llm_client=llm_client,
        model=model,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
    )


FlowFactory = Callable[[], ToolAgentFlow]


def create_code_agent_flow(
    *,
    registry: Optional[ToolRegistry] = None,
    llm_client: Any = None,
    max_parallel_workers: int = 4,
    model: Optional[str] = None,
    max_iterations: int = 25,
    system_prompt: Optional[str] = None,
    environment: Optional[Mapping[str, Any]] = None,
) -> ToolAgentFlow:
    """Return a `ToolAgentFlow` instance with all default tools registered."""

    resolved_prompt = system_prompt
    if resolved_prompt is None:
        resolved_prompt = build_code_agent_system_prompt(
            base_prompt=_BASE_SYSTEM_PROMPT,
            extra_sections=None,
            environment=environment,
        )

    flow = create_tool_agent_flow(
        registry=registry,
        llm_client=llm_client,
        model=model,
        max_iterations=max_iterations,
        system_prompt=resolved_prompt,
    )
    if max_parallel_workers < 1:
        raise ValueError("max_parallel_workers must be >= 1")
    if hasattr(flow, "execution_node") and hasattr(flow.execution_node, "max_parallel_workers"):
        flow.execution_node.max_parallel_workers = max_parallel_workers
    return flow


class CodeAgentSession:
    """In-memory conversation session for the Code Agent CLI."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        llm_client: Any = None,
        max_parallel_workers: int = 4,
        max_iterations: int = 25,
        flow_factory: Optional[FlowFactory] = None,
        system_prompt: Optional[str] = None,
        environment: Optional[Mapping[str, Any]] = None,
        workspace: Optional[str | Path] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        self.registry = registry or create_default_registry(project_root=self.workspace)
        self.llm_client = llm_client
        self.max_parallel_workers = max_parallel_workers
        self.max_iterations = max_iterations if max_iterations >= 1 else 1
        self.system_prompt = system_prompt
        if environment is not None:
            self.environment = environment
        elif self.workspace is not None:
            self.environment = {"cwd": str(self.workspace)}
        else:
            self.environment = None
        if flow_factory is not None:
            self._flow_factory = flow_factory
        else:
            default_model = get_config().llm.model
            self._flow_factory = lambda: create_code_agent_flow(
                registry=self.registry,
                llm_client=self.llm_client,
                max_parallel_workers=self.max_parallel_workers,
                max_iterations=self.max_iterations,
                model=default_model,
                system_prompt=self.system_prompt,
                environment=self.environment,
            )
        self.history: List[Dict[str, Any]] = []

    def run_turn(
        self,
        user_input: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty")
        flow = self._flow_factory()
        params: Dict[str, Any] = {"user_input": user_input.strip()}
        if self.history:
            params["history"] = list(self.history)
        if output_callback is not None:
            params["output_callback"] = output_callback
        flow.set_params(params)
        shared: Dict[str, Any] = {}
        if output_callback is not None:
            shared["output_callback"] = output_callback
        result = flow._run(shared)
        updated = result.get("history")
        if isinstance(updated, list):
            self.history = [self._normalize_message(msg) for msg in updated if self._is_valid_message(msg)]
        else:
            self.history.extend(
                [
                    {"role": "user", "content": user_input.strip()},
                    {"role": "assistant", "content": result.get("final_response", "")},
                ]
            )
        return result

    @staticmethod
    def _is_valid_message(raw: Any) -> bool:
        return isinstance(raw, dict) and isinstance(raw.get("role"), str) and "content" in raw

    @staticmethod
    def _normalize_message(raw: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(raw)
        normalized["role"] = str(normalized.get("role"))
        content = normalized.get("content", "")
        if isinstance(content, str):
            normalized["content"] = content
        elif content is None:
            normalized["content"] = ""
        else:
            normalized["content"] = content
        return normalized


def run_code_agent_cli(
    *,
    session: Optional["CodeAgentSession"] = None,
    session_factory: Optional[Callable[[], "CodeAgentSession"]] = None,
    input_iter: Optional[Iterable[str]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    console: Optional["Console"] = None,
) -> int:
    factory = session_factory or (lambda: CodeAgentSession())
    return _run_code_agent_cli(
        session=session,
        session_factory=factory,
        input_iter=input_iter,
        output_callback=output_callback,
        console=console,
    )


def run_code_agent_once(
    prompt: str,
    *,
    session: Optional["CodeAgentSession"] = None,
    session_factory: Optional[Callable[[], "CodeAgentSession"]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    console: Optional["Console"] = None,
) -> Dict[str, Any]:
    factory = session_factory or (lambda: CodeAgentSession())
    return _run_code_agent_once(
        prompt,
        session=session,
        session_factory=factory,
        output_callback=output_callback,
        console=console,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for running the code agent directly."""

    return _run_cli_main(
        argv,
        session_factory=lambda: CodeAgentSession(max_iterations=100),
    )


if __name__ == "__main__":
    raise SystemExit(main())
