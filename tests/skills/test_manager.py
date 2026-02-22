"""Tests for the SkillManager."""

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.skills.base import tool, FunctionSkill
from arc.skills.manager import SkillManager


@pytest.fixture
def kernel():
    return Kernel(config=ArcConfig())


@pytest.fixture
def manager(kernel):
    return SkillManager(kernel)


@pytest.fixture
def sample_skill():
    @tool(name="greet")
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    @tool(name="farewell")
    async def farewell(name: str) -> str:
        return f"Goodbye, {name}!"

    return FunctionSkill(
        name="greeter",
        description="Greeting skill",
        tools=[greet, farewell],
    )


@pytest.mark.asyncio
async def test_register(manager, sample_skill):
    """Skills can be registered."""
    await manager.register(sample_skill)

    assert "greeter" in manager.skill_names
    assert "greet" in manager.tool_names
    assert "farewell" in manager.tool_names


@pytest.mark.asyncio
async def test_execute_tool(manager, sample_skill):
    """SkillManager routes tool execution to correct skill."""
    await manager.register(sample_skill)

    result = await manager.execute_tool("greet", {"name": "Arc"})
    assert result.success is True
    assert "Hello, Arc" in result.output


@pytest.mark.asyncio
async def test_unknown_tool(manager, sample_skill):
    """Unknown tool returns error."""
    await manager.register(sample_skill)

    result = await manager.execute_tool("nonexistent", {})
    assert result.success is False
    assert "Unknown tool" in result.error


@pytest.mark.asyncio
async def test_get_all_tool_specs(manager, sample_skill):
    """get_all_tool_specs returns specs from all skills."""
    await manager.register(sample_skill)

    specs = manager.get_all_tool_specs()
    assert len(specs) == 2

    names = [s.name for s in specs]
    assert "greet" in names
    assert "farewell" in names


@pytest.mark.asyncio
async def test_get_tool_skill(manager, sample_skill):
    """get_tool_skill returns correct skill name."""
    await manager.register(sample_skill)

    assert manager.get_tool_skill("greet") == "greeter"
    assert manager.get_tool_skill("farewell") == "greeter"
    assert manager.get_tool_skill("nonexistent") is None


@pytest.mark.asyncio
async def test_multiple_skills(manager):
    """Multiple skills can be registered."""

    @tool(name="add")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    @tool(name="subtract")
    async def subtract(a: int, b: int) -> str:
        return str(a - b)

    math_skill = FunctionSkill("math", "Math ops", [add, subtract])

    @tool(name="upper")
    async def upper(text: str) -> str:
        return text.upper()

    text_skill = FunctionSkill("text", "Text ops", [upper])

    await manager.register(math_skill)
    await manager.register(text_skill)

    assert len(manager.skill_names) == 2
    assert len(manager.tool_names) == 3

    # Execute from different skills
    r1 = await manager.execute_tool("add", {"a": 1, "b": 2})
    assert r1.output == "3"

    r2 = await manager.execute_tool("upper", {"text": "hello"})
    assert r2.output == "HELLO"