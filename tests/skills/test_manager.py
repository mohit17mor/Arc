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


# ── Tool argument validation tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_validation_rejects_missing_required_param(manager, sample_skill):
    """Missing required parameter returns clear error without executing."""
    await manager.register(sample_skill)

    result = await manager.execute_tool("greet", {})
    assert result.success is False
    assert "missing required" in result.error.lower()
    assert "name" in result.error


@pytest.mark.asyncio
async def test_validation_error_shows_expected_params(manager, sample_skill):
    """Validation error message includes expected parameters summary."""
    await manager.register(sample_skill)

    result = await manager.execute_tool("greet", {})
    assert result.success is False
    assert "Expected parameters:" in result.error
    assert "name" in result.error
    assert "string" in result.error
    assert "required" in result.error


@pytest.mark.asyncio
async def test_validation_rejects_wrong_type(manager):
    """Wrong parameter type returns clear error without executing."""

    @tool(name="add")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    math_skill = FunctionSkill("math", "Math operations", [add])
    await manager.register(math_skill)

    # Pass string instead of integer
    result = await manager.execute_tool("add", {"a": "not_a_number", "b": 2})
    assert result.success is False
    assert "wrong type" in result.error.lower()
    assert "a" in result.error
    assert "integer" in result.error


@pytest.mark.asyncio
async def test_validation_passes_correct_args(manager, sample_skill):
    """Valid arguments pass validation and tool executes normally."""
    await manager.register(sample_skill)

    result = await manager.execute_tool("greet", {"name": "Arc"})
    assert result.success is True
    assert "Hello, Arc" in result.output


@pytest.mark.asyncio
async def test_validation_allows_extra_args(manager, sample_skill):
    """Extra args pass validation but may fail at tool level — that's expected.

    Validation doesn't reject extra args because LLMs sometimes add them
    and some tools accept **kwargs.  The tool itself decides.
    """
    await manager.register(sample_skill)

    result = await manager.execute_tool(
        "greet", {"name": "Arc", "extra_param": "ignored"}
    )
    # Validation passed (no "was not executed" in error), but tool may
    # reject the extra kwarg at the Python level — that's fine.
    if not result.success:
        assert "was not executed" not in result.error


@pytest.mark.asyncio
async def test_validation_multiple_missing_params(manager):
    """Multiple missing required params are all listed in the error."""

    @tool(name="create_file")
    async def create_file(path: str, content: str) -> str:
        return "ok"

    fs_skill = FunctionSkill("fs", "File operations", [create_file])
    await manager.register(fs_skill)

    result = await manager.execute_tool("create_file", {})
    assert result.success is False
    assert "path" in result.error
    assert "content" in result.error


@pytest.mark.asyncio
async def test_validation_optional_params_not_required(manager):
    """Optional parameters do not trigger missing-required errors."""

    @tool(name="search")
    async def search(query: str, max_results: int = 10) -> str:
        return f"searched: {query}"

    search_skill = FunctionSkill("searcher", "Search things", [search])
    await manager.register(search_skill)

    # Only provide required param, omit optional
    result = await manager.execute_tool("search", {"query": "AI news"})
    assert result.success is True
    assert "searched: AI news" in result.output


@pytest.mark.asyncio
async def test_validation_tool_name_in_error(manager, sample_skill):
    """Error message includes the tool name so the LLM knows which call failed."""
    await manager.register(sample_skill)

    result = await manager.execute_tool("greet", {})
    assert result.success is False
    assert "greet" in result.error