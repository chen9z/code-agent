from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from clients.llm import get_default_llm_client
from configs.manager import get_config
from __init__ import Flow, Node
from nodes.tool_execution import ToolExecutionBatchNode
from tools.registry import ToolRegistry, create_default_registry

_BASE_SYSTEM_PROMPT = (
    "You are Code Agent, an autonomous software assistant. Maintain the conversation "
    "history, decide which registered tools to call, and when finished produce a concise "
    "natural language answer."
)

_PLANNER_INSTRUCTIONS = (
    "When you respond you must output valid JSON with the schema:\n"
    "{\n"
    "  'thoughts': str,\n"
    "  'tool_calls': [\n"
    "    { 'id': str, 'key': str, 'arguments': dict, 'mode': 'parallel'|'sequential' }\n"
    "  ],\n"
    "  'final_response': str | null\n"
    "}\n"
    "Return an empty array for tool_calls if no tools are needed."
)

_SUMMARY_INSTRUCTIONS = (
    "Provide the final answer to the user using the available context. Reference tool "
    "results when they exist and be explicit about any limitations."
)


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
        shared["history"] = history
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
        planning_prompt = (
            _PLANNER_INSTRUCTIONS
            + "\nAvailable tools:\n"
            + serialized_tools
        )
        messages = history + [{"role": "system", "content": planning_prompt}]
        tools = self.registry.to_openai_tools()

        if hasattr(self.llm, "create_with_tools"):
            response = self.llm.create_with_tools(
                model=self.model,
                messages=messages,
                tools=tools,
                parallel_tool_calls=True,
            )
            plan = self._parse_tool_response(response)
            plan["raw_text"] = self._extract_raw_response(response)
        else:  # pragma: no cover - fallback path when client lacks tool support
            chunks = list(self.llm.get_response(model=self.model, messages=messages, stream=False))
            response_text = "".join(chunks).strip()
            plan = self._parse_plan_text(response_text)
            plan["raw_text"] = response_text
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
            mode = "parallel" if len(tool_calls) > 1 else "sequential"
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
                        "mode": mode,
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

    def _parse_plan_text(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"tool_calls": [], "final_response": text, "thoughts": None}

        if not isinstance(data, dict):
            return {"tool_calls": [], "final_response": str(data), "thoughts": None}

        raw_calls = data.get("tool_calls") or []
        normalized_calls: List[Dict[str, Any]] = []
        if isinstance(raw_calls, list):
            for idx, call in enumerate(raw_calls):
                if not isinstance(call, dict):
                    continue
                key = call.get("key") or call.get("tool")
                if not key:
                    continue
                arguments = call.get("arguments") or call.get("args") or {}
                if not isinstance(arguments, dict):
                    continue
                mode = call.get("mode") or ("parallel" if call.get("parallel") else "sequential")
                normalized_calls.append(
                    {
                        "id": call.get("id") or f"call-{idx}",
                        "key": str(key),
                        "arguments": arguments,
                        "mode": str(mode),
                    }
                )
        final_response = data.get("final_response")
        if isinstance(final_response, dict):
            final_response = json.dumps(final_response, ensure_ascii=False)
        elif final_response is not None:
            final_response = str(final_response)
        thoughts = data.get("thoughts")
        if isinstance(thoughts, (dict, list)):
            thoughts = json.dumps(thoughts, ensure_ascii=False)
        elif thoughts is not None:
            thoughts = str(thoughts)
        return {
            "tool_calls": normalized_calls,
            "final_response": final_response,
            "thoughts": thoughts,
        }

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
        history = prep_res["history"]
        tool_results = prep_res["tool_results"]
        plan = prep_res["tool_plan"]

        results_summary_lines: List[str] = []
        for result in tool_results:
            status = result.get("status")
            key = result.get("key")
            if status == "success":
                results_summary_lines.append(
                    f"Tool {key} succeeded with output: {json.dumps(result.get('output'), ensure_ascii=False)[:400]}"
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

        messages = [{"role": "system", "content": _BASE_SYSTEM_PROMPT}, {"role": "user", "content": summary_prompt}]
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


class ToolAgentFlow(Flow):
    """End-to-end agent flow orchestrating tool planning, execution, and summarisation."""

    def __init__(self, registry: Optional[ToolRegistry] = None, llm_client=None) -> None:
        cfg = get_config()
        model = cfg.llm.model
        self.registry = registry or create_default_registry()
        self.llm = llm_client or get_default_llm_client()

        super().__init__()
        self.context_node = ConversationContextNode()
        self.planning_node = ToolPlanningNode(self.registry, model=model, llm_client=self.llm)
        self.execution_node = ToolExecutionBatchNode(self.registry)
        self.summary_node = SummaryNode(model=model, llm_client=self.llm)

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


def create_tool_agent_flow(registry: Optional[ToolRegistry] = None, llm_client=None) -> ToolAgentFlow:
    return ToolAgentFlow(registry=registry, llm_client=llm_client)

