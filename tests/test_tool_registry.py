"""Tests for the centralized tool registry."""

import pytest

from tools.registry import ToolRegistry, ToolSpec, create_default_registry
from tools.read import ReadTool


def test_register_and_get_tool():
    registry = ToolRegistry()
    spec = registry.register(ReadTool(), key="read")

    assert isinstance(spec, ToolSpec)
    assert spec.key == "read"
    assert spec.name == "Read"
    assert "file_path" in spec.parameters["properties"]

    fetched = registry.get("read")
    assert fetched is spec


@pytest.mark.parametrize("query", ["Read", "read", "READ", " read "])
def test_key_normalization(query: str):
    registry = ToolRegistry()
    registry.register(ReadTool(), key="read")

    assert registry.get(query).key == "read"


def test_duplicate_registration_raises():
    registry = ToolRegistry()
    registry.register(ReadTool(), key="read")

    with pytest.raises(ValueError):
        registry.register(ReadTool(), key="read")


def test_default_registry_contains_expected_tools():
    registry = create_default_registry()
    keys = [spec.key for spec in registry.list()]

    assert "read" in keys
    assert "write" in keys
    assert "todo_write" in keys
    assert "codebase_search" in keys
    assert keys == sorted(keys)


def test_openai_descriptor_structure():
    registry = create_default_registry(include=["read"])
    openai_tools = registry.to_openai_tools()

    assert len(openai_tools) == 1
    tool_def = openai_tools[0]
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "read"
    assert "parameters" in tool_def["function"]


def test_describe_returns_metadata():
    registry = create_default_registry(include=["read"])
    descriptors = registry.describe()

    assert isinstance(descriptors, list)
    assert descriptors[0]["key"] == "read"
    assert descriptors[0]["name"] == "Read"
    assert "description" in descriptors[0]
    assert "parameters" in descriptors[0]
