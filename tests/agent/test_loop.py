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
    """Agent exhausts iterations and then synthesises an answer from gathered context."""
    # Fill all 5 iterations with tool calls so the loop never naturally completes.
    for _ in range(5):
        mock_llm.set_tool_call("greet", {"name": "Test"})

    # The 6th LLM call is the synthesis call (no tools available).
    # Queue a plain text response so the agent can render a real answer.
    mock_llm.set_response("Based on what I gathered, here is my best answer.")

    chunks = []
    async for chunk in agent.run("Keep greeting forever"):
        chunks.append(chunk)

    response = "".join(chunks)
    # The synthesis response should appear in the output.
    assert "best answer" in response
    # The agent must have reached COMPLETE state (not errored).
    assert agent.state.status.value == "complete"


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


# ── Action verification tests ───────────────────────────────────


class TestActionVerification:
    """Tests for the LLM action promise detection and nudging."""

    def test_detects_action_promise(self):
        assert AgentLoop._text_promises_action("Let me search for that")
        assert AgentLoop._text_promises_action("I'll look that up for you")
        assert AgentLoop._text_promises_action("I will check the files now")
        assert AgentLoop._text_promises_action("Searching for flights...")
        assert AgentLoop._text_promises_action("Let me browse that website")
        assert AgentLoop._text_promises_action("I'll use the web_search tool")

    def test_no_false_positives(self):
        assert not AgentLoop._text_promises_action("Here are the results:")
        assert not AgentLoop._text_promises_action("The answer is 42.")
        assert not AgentLoop._text_promises_action("Based on my knowledge, Python is great.")
        assert not AgentLoop._text_promises_action(
            "I don't have access to real-time data."
        )
        assert not AgentLoop._text_promises_action("")

    def test_only_checks_prefix(self):
        """Action phrases buried deep in a long response are ignored."""
        long_text = ("x " * 300) + "let me search for that"
        assert not AgentLoop._text_promises_action(long_text)

    def test_case_insensitive(self):
        assert AgentLoop._text_promises_action("Let Me Search for that")
        assert AgentLoop._text_promises_action("I'LL LOOK THAT UP")

    @pytest.mark.asyncio
    async def test_nudge_triggers_on_broken_promise(
        self, kernel, skill_manager, security
    ):
        """When LLM promises action but doesn't call a tool, it gets nudged."""
        mock_llm = MockLLMProvider()

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=skill_manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=5),
        )

        # First call: LLM says "let me search" but returns COMPLETE (no tool call)
        # Second call (after nudge): LLM calls the tool
        # Third call: LLM gives final answer
        mock_llm.set_response("Let me search for that right away.")
        mock_llm.set_tool_call("greet", {"name": "World"})
        mock_llm.set_response("Done! I greeted World.")

        chunks = []
        async for chunk in agent.run("Search for something"):
            chunks.append(chunk)

        response = "".join(chunks)
        assert "Done" in response

        # Verify the nudge was injected — at least 3 LLM calls
        assert mock_llm.call_count >= 3

        # Check that the nudge message was in the conversation
        msgs = agent.memory.get_messages(include_system=False)
        nudge_found = any(
            "didn't call any tool" in (m.content or "")
            for m in msgs
            if m.role == "user"
        )
        assert nudge_found

    @pytest.mark.asyncio
    async def test_no_nudge_when_no_promise(self, kernel, skill_manager, security):
        """Normal text responses without action promises are not nudged."""
        mock_llm = MockLLMProvider()

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=skill_manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=5),
        )

        mock_llm.set_response("The answer is 42.")

        async for _ in agent.run("What is the answer?"):
            pass

        # Only 1 LLM call — no nudge needed
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_nudge_only_once(self, kernel, skill_manager, security):
        """If the LLM keeps promising without acting, don't loop forever."""
        mock_llm = MockLLMProvider()

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=skill_manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=4),
        )

        # LLM keeps saying "let me do X" without ever calling a tool
        mock_llm.set_responses([
            "Let me search for that.",
            "I'll look that up now.",
            "Let me check the web.",
            "I'll search right away.",  # max_iterations reached → synthesise
        ])

        chunks = []
        async for chunk in agent.run("Search for flights"):
            chunks.append(chunk)

        # Should not hang — respects max_iterations
        assert mock_llm.call_count <= 5  # 4 + possible synthesis