"""Tests for the centralized tool registry."""

import pytest

from tools.registry import ToolRegistry, ToolSpec, create_default_registry
from tools.read import ReadTool


def test_register_and_get_tool():
    registry = ToolRegistry()
    spec = registry.register(ReadTool(), name="read")

    assert isinstance(spec, ToolSpec)
    assert spec.name == "read"
    assert spec.tool.name == "Read"
    assert "file_path" in spec.parameters["properties"]

    fetched = registry.get("read")
    assert fetched is spec


@pytest.mark.parametrize("query", ["Read", "read", "READ", " read "])
def test_name_normalization(query: str):
    registry = ToolRegistry()
    registry.register(ReadTool(), name="read")

    assert registry.get(query).name == "read"


def test_duplicate_registration_raises():
    registry = ToolRegistry()
    registry.register(ReadTool(), name="read")

    with pytest.raises(ValueError):
        registry.register(ReadTool(), name="read")


def test_default_registry_contains_expected_tools():
    registry = create_default_registry()
    names = [spec.name for spec in registry.list()]

    assert "bash" in names
    assert "codebase_search" in names
    assert "read" in names
    assert "write" in names
    assert "todo_write" in names
    assert names == sorted(names)


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
    assert descriptors[0]["name"] == "read"
    assert descriptors[0]["display_name"] == "Read"
    assert "description" in descriptors[0]
    assert "parameters" in descriptors[0]


def test_codebase_search_respects_project_root(tmp_path):
    registry = create_default_registry(
        include=["codebase_search"],
        project_root=tmp_path,
    )
    spec = registry.get("codebase_search")

    assert spec.tool.name == "codebase_search"
    assert spec.tool._default_root == tmp_path.resolve()
