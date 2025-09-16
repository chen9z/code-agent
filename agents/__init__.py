"""Agent registries and helpers."""

from importlib import import_module
from typing import Any, Callable, Dict


AgentFactory = Callable[[], Any]


_REGISTRY: Dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    _REGISTRY[name] = factory


def get_agent(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Agent '{name}' is not registered")
    return _REGISTRY[name]()


def load_default_agents() -> None:
    import_module("agents.code_rag")

