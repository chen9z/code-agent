"""Simplified Code Agent CLI built on the Flow/Node runtime primitives."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from __init__ import Flow, Node
from tool_agent import ToolAgentFlow, create_tool_agent_flow
from tools.registry import ToolRegistry, create_default_registry

FlowFactory = Callable[[], ToolAgentFlow]


def create_code_agent_flow(
    *,
    registry: Optional[ToolRegistry] = None,
    llm_client: Any = None,
    max_parallel_workers: int = 4,
) -> ToolAgentFlow:
    """Return a `ToolAgentFlow` instance with all default tools registered."""

    flow = create_tool_agent_flow(registry=registry, llm_client=llm_client)
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
        self._flow_factory: FlowFactory = flow_factory or (
            lambda: create_code_agent_flow(
                registry=self.registry,
                llm_client=self.llm_client,
                max_parallel_workers=self.max_parallel_workers,
            )
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
        return {"role": str(raw.get("role")), "content": str(raw.get("content", ""))}


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
        mode = (call.get("mode") or "sequential").lower()
        output_callback(f"[plan] {mode}:{key}")
    for tool_result in result.get("tool_results") or []:
        key = tool_result.get("key")
        status = tool_result.get("status")
        if status == "success":
            preview = json.dumps(tool_result.get("output"), ensure_ascii=False)[:160]
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

