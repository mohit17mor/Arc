"""Tests for the agent loop."""

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.core.types import ToolResult
from arc.llm.mock import MockLLMProvider
from arc.skills.base import tool, FunctionSkill
from arc.skills.manager import SkillManager
from arc.security.engine import SecurityEngine
from arc.agent.loop import AgentLoop, AgentConfig


@pytest.fixture
def kernel():
    return Kernel(config=ArcConfig())


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
async def skill_manager(kernel):
    manager = SkillManager(kernel)

    @tool(name="greet")
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    @tool(name="add")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    skill = FunctionSkill("test", "Test skill", [greet, add])
    await manager.register(skill)
    return manager


@pytest.fixture
def security(kernel):
    config = ArcConfig().security
    config.auto_allow = ["file:read", "file:write", "shell:exec"]
    return SecurityEngine(config, kernel)


@pytest.fixture
def agent(kernel, mock_llm, skill_manager, security):
    return AgentLoop(
        kernel=kernel,
        llm=mock_llm,
        skill_manager=skill_manager,
        security=security,
        system_prompt="You are a helpful assistant.",
        config=AgentConfig(max_iterations=5),
    )


@pytest.mark.asyncio
async def test_simple_response(agent, mock_llm):
    """Agent returns simple text response."""
    mock_llm.set_response("Hello! How can I help you?")

    chunks = []
    async for chunk in agent.run("Hi"):
        chunks.append(chunk)

    response = "".join(chunks)
    assert "Hello" in response
    assert agent.memory.message_count == 2  # user + assistant


@pytest.mark.asyncio
async def test_tool_call(agent, mock_llm):
    """Agent executes tool calls."""
    # First LLM call returns tool call
    mock_llm.set_tool_call("greet", {"name": "World"})
    # Second LLM call returns final response
    mock_llm.set_response("I greeted World for you!")

    chunks = []
    async for chunk in agent.run("Please greet World"):
        chunks.append(chunk)

    response = "".join(chunks)
    assert "greeted" in response.lower() or "World" in response

    # Memory should have: user, assistant (tool call), tool result, assistant
    assert agent.memory.message_count == 4


@pytest.mark.asyncio
async def test_multiple_tool_calls(agent, mock_llm):
    """Agent handles multiple tool calls in sequence."""
    # First: tool call to add
    mock_llm.set_tool_call("add", {"a": 2, "b": 3})
    # Second: another tool call
    mock_llm.set_tool_call("add", {"a": 10, "b": 20})
    # Third: final response
    mock_llm.set_response("The results are 5 and 30.")

    chunks = []
    async for chunk in agent.run("Add 2+3 and 10+20"):
        chunks.append(chunk)

    response = "".join(chunks)
    assert "5" in response or "30" in response


@pytest.mark.asyncio
async def test_max_iterations(agent, mock_llm):
    """Agent stops at max iterations."""
    # Always return tool calls, never complete
    for _ in range(10):
        mock_llm.set_tool_call("greet", {"name": "Test"})

    chunks = []
    async for chunk in agent.run("Keep greeting forever"):
        chunks.append(chunk)

    response = "".join(chunks)
    assert "Max iterations" in response


@pytest.mark.asyncio
async def test_events_emitted(agent, mock_llm, kernel):
    """Agent emits appropriate events."""
    events = []

    async def capture_event(event):
        events.append(event.type)

    kernel.on("*", capture_event)
    mock_llm.set_response("Done!")

    async for _ in agent.run("Test"):
        pass

    assert "agent:start" in events
    assert "agent:thinking" in events
    assert "llm:response" in events
    assert "agent:complete" in events


@pytest.mark.asyncio
async def test_reset(agent, mock_llm):
    """Agent reset clears memory."""
    mock_llm.set_response("First response")

    async for _ in agent.run("First message"):
        pass

    assert agent.memory.message_count == 2

    agent.reset()

    assert agent.memory.message_count == 0


@pytest.mark.asyncio
async def test_conversation_continuity(agent, mock_llm):
    """Agent maintains conversation context."""
    mock_llm.set_responses(["My name is Arc.", "I already told you, I'm Arc!"])

    # First turn
    async for _ in agent.run("What's your name?"):
        pass

    # Second turn (should have context from first)
    async for _ in agent.run("What did you say your name was?"):
        pass

    # Check that LLM received conversation history
    last_call = mock_llm.all_calls[-1]
    messages = last_call["messages"]
    
    # Should have: system, user1, assistant1, user2
    assert len(messages) >= 4