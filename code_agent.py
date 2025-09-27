"""Code Agent flow and CLI orchestrator built on the Flow/Node runtime."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from __init__ import Flow, Node
from clients.llm import get_default_llm_client
from configs.manager import get_config
from nodes.tool_execution import ToolExecutionBatchNode
from tools.registry import ToolRegistry, create_default_registry

_BASE_SYSTEM_PROMPT = (
    "You are Code Agent, an autonomous software assistant. Maintain the conversation "
    "history, decide which registered tools to call, and when finished produce a concise "
    "natural language answer."
)

# JSON-plan fallback removed; we rely on native tool-calling.

_SUMMARY_INSTRUCTIONS = (
    "Provide the final answer to the user using the available context. Reference tool "
    "results when they exist and be explicit about any limitations."
)

# Guidance used only when the client supports native tool calls.
_PLANNER_TOOLCALL_INSTRUCTIONS = (
    "Plan tool usage using the tool-calling interface. Call tools when they help, "
    "and if no tool is needed, reply normally. Do not output JSON schemas. "
    "When calls depend on prior results, issue them one at a time; when they are "
    "independent, you may include multiple tool calls in the same assistant message."
)


def _stringify_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _preview_payload(value: Any, limit: int) -> str:
    text = _stringify_payload(value)
    if limit <= 3 or len(text) <= limit:
        return text[:limit]
    return f"{text[: limit - 3]}..."


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
        descriptors = prep_res["descriptors"]
        serialized_tools = json.dumps(descriptors, ensure_ascii=False, indent=2)
        planning_prompt = _PLANNER_TOOLCALL_INSTRUCTIONS + "\nAvailable tools:\n" + serialized_tools
        messages = history + [{"role": "system", "content": planning_prompt}]
        tools = self.registry.to_openai_tools()

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
        history.append(
            {
                "role": "assistant",
                "content": exec_res.get("thoughts")
                or f"Tool plan: {json.dumps(exec_res.get('tool_calls', []), ensure_ascii=False)}",
            }
        )
        return "execute" if exec_res.get("tool_calls") else "summarize"

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


class SummaryNode(Node):
    """Creates the final natural language response using the LLM."""

    def __init__(self, model: str, llm_client=None) -> None:
        super().__init__()
        self.model = model
        self.llm = llm_client or get_default_llm_client()

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
            status = result.get("status")
            key = result.get("key")
            if status == "success":
                results_summary_lines.append(
                    f"Tool {key} succeeded with output: {_preview_payload(result.get('output'), 400)}"
                )
            else:
                results_summary_lines.append(
                    f"Tool {key} failed with error: {result.get('error')}"
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
        return trimmed

    def _build_messages(self, history: List[Dict[str, Any]], summary_prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": _BASE_SYSTEM_PROMPT}]
        for message in history:
            if message.get("role") == "system":
                continue
            messages.append(message)
        messages.append({"role": "user", "content": summary_prompt})
        return messages


class ToolAgentFlow(Flow):
    """End-to-end agent flow orchestrating tool planning, execution, and summarisation."""

    def __init__(self, registry: Optional[ToolRegistry] = None, llm_client=None, model: Optional[str] = None) -> None:
        if model is None:
            cfg = get_config()
            model = cfg.llm.model
        self.registry = registry or create_default_registry()
        self.llm = llm_client or get_default_llm_client()
        self.model = model

        super().__init__()
        self.context_node = ConversationContextNode()
        self.planning_node = ToolPlanningNode(self.registry, model=self.model, llm_client=self.llm)
        self.execution_node = ToolExecutionBatchNode(self.registry)
        self.summary_node = SummaryNode(model=self.model, llm_client=self.llm)

        self.start(self.context_node)
        self.context_node.next(self.planning_node, "plan")
        self.planning_node.next(self.execution_node, "execute")
        self.planning_node.next(self.summary_node, "summarize")
        self.execution_node.next(self.summary_node, "summarize")

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        history = self.params.get("history") or shared.get("history")
        if history:
            shared["history"] = list(history)
        user_input = self.params.get("user_input") or shared.get("user_input")
        if not user_input:
            raise ValueError("user_input is required for the tool agent flow")
        shared["user_input"] = user_input
        return {}

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> Dict[str, Any]:
        return {
            "final_response": shared.get("final_response"),
            "tool_results": shared.get("tool_results", []),
            "tool_plan": shared.get("tool_plan"),
            "history": shared.get("history", []),
        }


def create_tool_agent_flow(registry: Optional[ToolRegistry] = None, llm_client=None, model: Optional[str] = None) -> ToolAgentFlow:
    return ToolAgentFlow(registry=registry, llm_client=llm_client, model=model)


FlowFactory = Callable[[], ToolAgentFlow]


def create_code_agent_flow(
    *,
    registry: Optional[ToolRegistry] = None,
    llm_client: Any = None,
    max_parallel_workers: int = 4,
    model: Optional[str] = None,
) -> ToolAgentFlow:
    """Return a `ToolAgentFlow` instance with all default tools registered."""

    flow = create_tool_agent_flow(registry=registry, llm_client=llm_client, model=model)
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
        flow_factory: Optional[FlowFactory] = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.llm_client = llm_client
        self.max_parallel_workers = max_parallel_workers
        if flow_factory is not None:
            self._flow_factory = flow_factory
        else:
            default_model = get_config().llm.model
            self._flow_factory = lambda: create_code_agent_flow(
                registry=self.registry,
                llm_client=self.llm_client,
                max_parallel_workers=self.max_parallel_workers,
                model=default_model,
            )
        self.history: List[Dict[str, Any]] = []

    def run_turn(self, user_input: str) -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input cannot be empty")
        flow = self._flow_factory()
        params: Dict[str, Any] = {"user_input": user_input.strip()}
        if self.history:
            params["history"] = list(self.history)
        flow.set_params(params)
        result = flow._run({})
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


class _RunLoopNode(Node):
    def __init__(self, session: CodeAgentSession) -> None:
        super().__init__()
        self.session = session

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input_iter": self.params.get("input_iter"),
            "output_callback": self.params.get("output_callback") or print,
        }

    def exec(self, prep_res: Dict[str, Any]) -> int:
        iterator = prep_res["input_iter"] if prep_res["input_iter"] is not None else _stdin_iterator()
        output = prep_res["output_callback"]
        output("Entering Code Agent. Type 'exit' to quit.")
        for raw in iterator:
            message = raw.strip()
            if not message:
                continue
            if message.lower() in {"exit", "quit"}:
                break
            result = self.session.run_turn(message)
            _emit_result(result, output)
        return 0

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: int) -> int:
        shared["exit_code"] = exec_res
        return "complete"


class CodeAgentCLIFlow(Flow):
    def __init__(self, session: Optional[CodeAgentSession] = None) -> None:
        super().__init__()
        self.session = session or CodeAgentSession()
        self.loop_node = _RunLoopNode(self.session)
        self.start(self.loop_node)

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> int:
        return shared.get("exit_code", 0)


def run_code_agent_cli(
    *,
    session: Optional[CodeAgentSession] = None,
    input_iter: Optional[Iterable[str]] = None,
    output_callback: Callable[[str], None] = print,
) -> int:
    flow = CodeAgentCLIFlow(session=session)
    flow.set_params({"input_iter": input_iter, "output_callback": output_callback})
    return flow._run({})


def _emit_result(result: Mapping[str, Any], output_callback: Callable[[str], None]) -> None:
    plan = result.get("tool_plan") or {}
    thoughts = plan.get("thoughts")
    if thoughts:
        output_callback(f"[planner] {thoughts}")
    for call in plan.get("tool_calls") or []:
        key = call.get("key")
        output_callback(f"[plan] {key}")
    for tool_result in result.get("tool_results") or []:
        key = tool_result.get("key")
        status = tool_result.get("status")
        if status == "success":
            preview = _preview_payload(tool_result.get("output"), 160)
            output_callback(f"[tool] {key}: success {preview}")
        else:
            output_callback(f"[tool] {key}: error {tool_result.get('error')}")
    final = result.get("final_response")
    if final:
        output_callback(f"[assistant] {final}")


def _stdin_iterator() -> Iterable[str]:
    while True:
        try:
            yield input("You: ")
        except EOFError:
            break
