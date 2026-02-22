"""Tests for skill base classes and @tool decorator."""

import pytest
from arc.core.types import Capability, ToolResult
from arc.skills.base import tool, ToolDef, FunctionSkill


def test_tool_decorator():
    """@tool decorator creates ToolDef with correct attributes."""

    @tool(name="greet", description="Greet someone")
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert isinstance(greet, ToolDef)
    assert greet.name == "greet"
    assert greet.description == "Greet someone"
    assert "name" in greet.parameters["properties"]
    assert greet.parameters["required"] == ["name"]


def test_tool_decorator_default_name():
    """@tool uses function name if name not provided."""

    @tool(description="Test")
    async def my_function() -> str:
        return "test"

    assert my_function.name == "my_function"


def test_tool_decorator_with_capabilities():
    """@tool captures required capabilities."""

    @tool(
        name="delete",
        description="Delete file",
        capabilities=[Capability.FILE_DELETE],
    )
    async def delete_file(path: str) -> str:
        return "deleted"

    assert Capability.FILE_DELETE in delete_file.capabilities


def test_tool_decorator_parameter_types():
    """@tool extracts correct JSON types from type hints."""

    @tool(name="test")
    async def test_func(
        text: str,
        count: int,
        rate: float,
        enabled: bool,
    ) -> str:
        return "ok"

    props = test_func.parameters["properties"]
    assert props["text"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["rate"]["type"] == "number"
    assert props["enabled"]["type"] == "boolean"


def test_tool_decorator_optional_params():
    """@tool handles optional parameters correctly."""

    @tool(name="test")
    async def test_func(required: str, optional: str = "default") -> str:
        return "ok"

    assert test_func.parameters["required"] == ["required"]


@pytest.mark.asyncio
async def test_function_skill():
    """FunctionSkill wraps @tool functions into a skill."""

    @tool(name="add", description="Add two numbers")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    @tool(name="multiply", description="Multiply two numbers")
    async def multiply(a: int, b: int) -> str:
        return str(a * b)

    skill = FunctionSkill(
        name="math",
        description="Math operations",
        tools=[add, multiply],
    )

    manifest = skill.manifest()
    assert manifest.name == "math"
    assert len(manifest.tools) == 2

    tool_names = [t.name for t in manifest.tools]
    assert "add" in tool_names
    assert "multiply" in tool_names


@pytest.mark.asyncio
async def test_function_skill_execute():
    """FunctionSkill executes tools correctly."""

    @tool(name="double")
    async def double(n: int) -> str:
        return str(n * 2)

    skill = FunctionSkill(
        name="math",
        description="Math",
        tools=[double],
    )

    result = await skill.execute_tool("double", {"n": 5})
    assert result.success is True
    assert result.output == "10"


@pytest.mark.asyncio
async def test_function_skill_unknown_tool():
    """FunctionSkill returns error for unknown tools."""

    @tool(name="test")
    async def test_func() -> str:
        return "ok"

    skill = FunctionSkill(
        name="test_skill",
        description="Test",
        tools=[test_func],
    )

    result = await skill.execute_tool("nonexistent", {})
    assert result.success is False
    assert "Unknown tool" in result.error


@pytest.mark.asyncio
async def test_function_skill_execution_error():
    """FunctionSkill captures exceptions in ToolResult."""

    @tool(name="fail")
    async def fail() -> str:
        raise ValueError("intentional error")

    skill = FunctionSkill(
        name="test_skill",
        description="Test",
        tools=[fail],
    )

    result = await skill.execute_tool("fail", {})
    assert result.success is False
    assert "intentional error" in result.error