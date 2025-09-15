from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """Abstract base class for all tools in the code-agent system."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters."""
        pass

    def __call__(self, **kwargs) -> Any:
        """Make the tool callable."""
        return self.execute(**kwargs)