from __future__ import annotations

from typing import Any, Dict, List

from tools.base import BaseTool


_ALLOWED_STATUSES = {"pending", "in_progress", "completed"}


class TodoWriteTool(BaseTool):
    """Structured todo list tool for tracking coding tasks during a session."""

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def description(self) -> str:
        return (
            "Use this tool to create and manage a structured task list for your current coding session. "
            "This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\n\n"
            "## When to Use This Tool\nUse this tool proactively in these scenarios:\n\n"
            "1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n"
            "2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n"
            "3. User explicitly requests todo list - When the user directly asks you to use the todo list\n"
            "4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n"
            "5. After receiving new instructions - Immediately capture user requirements as todos\n"
            "6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time\n"
            "7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation\n\n"
            "## Task States\n- pending\n- in_progress\n- completed"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["todos"],
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The updated todo list",
                    "items": {
                        "type": "object",
                        "required": ["content", "status", "activeForm"],
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": sorted(_ALLOWED_STATUSES),
                            },
                            "content": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "activeForm": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "additionalProperties": False,
                    },
                }
            },
            "additionalProperties": False,
        }

    def execute(self, *, todos: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            normalized = self._validate_and_normalize(todos)
            formatted = self._format_summary(normalized)
            return {
                "todos": normalized,
                "result": formatted,
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            return {"error": str(exc)}

    def _validate_and_normalize(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        if not isinstance(todos, list):
            raise TypeError("todos must be a list of todo objects")
        if not todos:
            raise ValueError("todos list cannot be empty")

        normalized: List[Dict[str, str]] = []
        in_progress_count = 0
        status_counts = {key: 0 for key in _ALLOWED_STATUSES}
        for index, raw in enumerate(todos):
            if not isinstance(raw, dict):
                raise TypeError(f"todo entry at index {index} must be an object")
            try:
                status = str(raw["status"])  # type: ignore[index]
                content = str(raw["content"])  # type: ignore[index]
                active_form = str(raw["activeForm"])  # type: ignore[index]
            except KeyError as missing:
                raise KeyError(f"todo entry at index {index} is missing required field '{missing.args[0]}'")

            status = status.strip()
            content = content.strip()
            active_form = active_form.strip()

            if status not in _ALLOWED_STATUSES:
                raise ValueError(
                    f"todo entry at index {index} has invalid status '{status}'. "
                    f"Allowed values: {sorted(_ALLOWED_STATUSES)}"
                )
            if not content:
                raise ValueError(f"todo entry at index {index} must have non-empty content")
            if not active_form:
                raise ValueError(f"todo entry at index {index} must have non-empty activeForm")

            if status == "in_progress":
                in_progress_count += 1
            status_counts[status] += 1

            normalized.append({
                "status": status,
                "content": content,
                "activeForm": active_form,
            })

        if in_progress_count == 1:
            return normalized

        completed_only = (
            in_progress_count == 0
            and status_counts["pending"] == 0
            and status_counts["completed"] == len(normalized)
        )

        if completed_only:
            return normalized

        raise ValueError(
            "Exactly one todo must have status 'in_progress'. Found "
            f"{in_progress_count} entries marked in_progress."
        )

    @staticmethod
    def _format_summary(todos: List[Dict[str, str]]) -> str:
        buckets: Dict[str, List[Dict[str, str]]] = {status: [] for status in _ALLOWED_STATUSES}
        for todo in todos:
            buckets[todo["status"]].append(todo)

        lines: List[str] = []
        status_order = ["in_progress", "pending", "completed"]
        for status in status_order:
            items = buckets[status]
            if not items:
                continue
            header = status.replace("_", " ").title()
            lines.append(f"{header} ({len(items)}):")
            for todo in items:
                lines.append(f"- {todo['content']} ({todo['activeForm']})")
        return "\n".join(lines)
