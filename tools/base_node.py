from core.base import Node
from typing import Any, Dict, Optional
from abc import abstractmethod


class BaseToolNode(Node):
    """Base class for tool nodes that wrap existing BaseTool functionality."""
    
    def __init__(self, max_retries: int = 1, wait: int = 0):
        super().__init__(max_retries, wait)
        self.tool = None
        self._init_tool()
    
    @abstractmethod
    def _init_tool(self):
        """Initialize the underlying tool instance."""
        pass
    
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
    
    def exec(self, params: Dict[str, Any]) -> Any:
        """Execute the tool with the given parameters."""
        return self.tool.execute(**params)
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for execution."""
        # Extract tool-specific parameters from shared context
        tool_params = {}
        for param_name in self.parameters.get("properties", {}).keys():
            if param_name in shared:
                tool_params[param_name] = shared[param_name]
        
        # Include required parameters
        for param_name in self.parameters.get("required", []):
            if param_name not in tool_params and param_name in shared:
                tool_params[param_name] = shared[param_name]
        
        return tool_params
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> Any:
        """Post-process execution results."""
        # Store results in shared context for downstream nodes
        if isinstance(exec_res, dict):
            shared.update(exec_res)
        
        # Return action for flow control
        action = prep_res.get("action") if isinstance(prep_res, dict) else None
        return action or "default"
