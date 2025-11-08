"""End-to-end tests for the tool agent flow."""

import json
from pathlib import Path

from code_agent import create_tool_agent_flow
from nodes.tool_execution import ToolOutput
from tools.registry import create_default_registry


class StubLLMClient:
    def __init__(self, plan_message, summary_response):
        self.plan_message = plan_message
        self.summary_response = summary_response
        self.calls: list[dict[str, object]] = []

    def create_with_tools(
        self,
        *,
        model,
        messages,
        tools,
        tool_choice=None,
        parallel_tool_calls=True,
        temperature=None,
    ):
        self.calls.append(
            {
                "kind": "plan",
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
            }
        )
        return {"choices": [{"message": self.plan_message}]}

    def get_response(self, model, messages, *, temperature=None, stream=False):
        self.calls.append(
            {"kind": "summary", "model": model, "messages": messages, "temperature": temperature}
        )
        yield self.summary_response


def test_tool_agent_flow_executes_tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.txt"
    target.write_text("Hello world!", encoding="utf-8")

    plan_message = {
        "role": "assistant",
        "content": "Listing files via bash and reading contents.",
        "tool_calls": [
            {
                "id": "call_bash",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": f"ls {workspace}"}),
                },
            },
            {
                "id": "call_glob",
                "function": {
                    "name": "glob",
                    "arguments": json.dumps({"path": str(workspace), "pattern": "*.txt"}),
                },
            },
            {
                "id": "call_read",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"file_path": str(target)}),
                },
            },
        ],
    }

    stub = StubLLMClient(
        plan_message=plan_message,
        summary_response="The directory contains notes.txt with greeting content.",
    )

    registry = create_default_registry(include=["bash", "glob", "read"])
    flow = create_tool_agent_flow(registry=registry, llm_client=stub, max_iterations=1)
    result = flow.run({"user_input": f"Please review {workspace}"})

    assert result["final_response"] == "The directory contains notes.txt with greeting content."
    assert len(result["tool_results"]) == 3
    assert all(isinstance(entry, ToolOutput) for entry in result["tool_results"])
    assert all(entry.status == "success" for entry in result["tool_results"])
    assert result["tool_plan"]["tool_calls"]
    assert len(stub.calls) >= 2
    planning_messages = stub.calls[0]["messages"]
    system_msgs = [msg for msg in planning_messages if msg.get("role") == "system"]
    assert len(system_msgs) == 1
    assert "Available tools" not in system_msgs[0].get("content", "")


def test_tool_agent_flow_direct_response(tmp_path):
    plan_message = {
        "role": "assistant",
        "content": "Answer directly.",
        "tool_calls": [],
    }

    stub = StubLLMClient(
        plan_message=plan_message,
        summary_response="Direct answer without tool usage.",
    )

    flow = create_tool_agent_flow(
        registry=create_default_registry(include=["read"]),
        llm_client=stub,
        max_iterations=1,
    )
    result = flow.run({"user_input": "Just say hello"})

    assert result["tool_results"] == []
    assert result["final_response"] == "Direct answer without tool usage."
    assert len(stub.calls) >= 2
