from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tools.base import BaseTool


@dataclass(frozen=True)
class ToolSpec:
    """Immutable metadata describing a tool instance."""

    key: str
    name: str
    description: str
    parameters: Dict[str, Any]
    tool: BaseTool

    def as_openai_tool(self) -> Dict[str, Any]:
        """Return a function-call schema compatible with OpenAI chat tool definitions."""
        return {
            "type": "function",
            "function": {
                "name": self.key,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Central registry for tools exposed to agent flows."""

    def __init__(self) -> None:
        self._registry: Dict[str, ToolSpec] = {}

    def register(self, tool: BaseTool, key: Optional[str] = None) -> ToolSpec:
        """Register a tool instance under a normalized key."""
        candidate_key = self._normalize_key(key) if key else self._normalize_key(tool.name)
        if candidate_key in self._registry:
            raise ValueError(f"Tool '{candidate_key}' is already registered")
        spec = ToolSpec(
            key=candidate_key,
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            tool=tool,
        )
        self._registry[candidate_key] = spec
        return spec

    def get(self, key: str) -> ToolSpec:
        """Retrieve a tool specification by key."""
        canonical = self._normalize_key(key)
        try:
            return self._registry[canonical]
        except KeyError as exc:  # pragma: no cover - validated via tests
            raise KeyError(f"Tool '{key}' is not registered") from exc

    def list(self) -> List[ToolSpec]:
        """Return all registered tool specifications sorted by key."""
        return [self._registry[k] for k in sorted(self._registry)]

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions compatible with OpenAI function calling."""
        return [spec.as_openai_tool() for spec in self.list()]

    def describe(self) -> List[Mapping[str, Any]]:
        """Return lightweight tool descriptors for prompt construction."""
        descriptors: List[Mapping[str, Any]] = []
        for spec in self.list():
            descriptors.append(
                {
                    "key": spec.key,
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                }
            )
        return descriptors

    @staticmethod
    def _normalize_key(raw: str) -> str:
        if not raw:
            raise ValueError("Tool name/key must be a non-empty string")
        return raw.strip().lower().replace(" ", "_")


def create_default_registry(include: Optional[Iterable[str]] = None) -> ToolRegistry:
    """Construct a registry with the built-in tool implementations."""
    from tools.bash import BashTool
    from tools.edit import EditTool
    from tools.glob import GlobTool
    from tools.grep import GrepSearchTool
    from tools.multi_edit import MultiEditTool
    from tools.read import ReadTool
    from tools.todo_write import TodoWriteTool
    from tools.write import WriteTool

    tool_classes = {
        "bash": BashTool,
        "edit": EditTool,
        "glob": GlobTool,
        "grep": GrepSearchTool,
        "multi_edit": MultiEditTool,
        "read": ReadTool,
        "todo_write": TodoWriteTool,
        "write": WriteTool,
    }

    registry = ToolRegistry()
    selected = set(t.lower() for t in include) if include else None
    for key, tool_cls in tool_classes.items():
        if selected is not None and key not in selected:
            continue
        registry.register(tool_cls(), key=key)
    return registry
