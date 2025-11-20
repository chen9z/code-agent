from __future__ import annotations

"""Shared type definitions for tool execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    """Standardized record for tool execution outcomes."""

    status: str = "unknown"
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    tool_call: Optional[ToolCall] = None

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = "unknown"
        if self.content is None:
            self.content = ""
        if self.data is None:
            self.data = {}

    @property
    def result_text(self) -> str:
        return self.content or ""

    def attach_call(self, tool_call: ToolCall) -> "ToolResult":
        self.tool_call = tool_call
        return self

    def as_dict(self) -> Dict[str, Any]:
        call = self.tool_call
        return {
            "inputs": {
                "name": call.name if call else None,
                "arguments": call.arguments if call else {},
                "call_id": call.call_id if call else None,
            },
            "result": {
                "status": self.status,
                "content": self.content,
                "data": self.data,
            },
        }

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "status": self.status,
            "content": self.content,
            "data": self.data,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class ToolValidationError(Exception):
    """Raised when tool argument validation fails."""

    def __init__(self, message: str, *, raw_arguments: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.raw_arguments = raw_arguments or {}
