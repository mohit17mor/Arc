"""Tests for the agent loop."""

import asyncio

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.core.types import Capability, LLMChunk, SkillManifest, StopReason, ToolResult, ToolSpec
from arc.llm.mock import MockLLMProvider
from arc.skills.base import tool, FunctionSkill, Skill
from arc.skills.manager import SkillManager
from arc.skills.router import SkillRouter, USE_SKILL_TOOL
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


class _OnDemandBrowserSkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="browser_control",
            version="1.0.0",
            description="Interactive browser automation",
            tools=(
                ToolSpec(
                    name="browser_go",
                    description="Navigate browser",
                    parameters={"type": "object", "properties": {}, "required": []},
                    required_capabilities=frozenset([Capability.BROWSER]),
                ),
            ),
            always_available=False,
        )

    async def execute_tool(self, tool_name, arguments):
        return ToolResult(success=True, output="navigated")


class _AlwaysFailSkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="flaky_browser",
            version="1.0.0",
            description="Always failing browser-like tool",
            tools=(
                ToolSpec(
                    name="browser_go",
                    description="Navigate browser",
                    parameters={"type": "object", "properties": {}, "required": []},
                    required_capabilities=frozenset(),
                ),
            ),
            always_available=True,
        )

    async def execute_tool(self, tool_name, arguments):
        return ToolResult(success=False, output="", error="navigation failed")


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
async def test_plan_update_event_uses_agent_source(kernel, security):
    """Plan updates should be tagged with the owning agent id."""
    mock_llm = MockLLMProvider()
    manager = SkillManager(kernel)
    events = []

    async def capture_event(event):
        if event.type == "agent:plan_update":
            events.append(event)

    kernel.on("*", capture_event)

    agent = AgentLoop(
        kernel=kernel,
        llm=mock_llm,
        skill_manager=manager,
        security=security,
        system_prompt="You are helpful.",
        config=AgentConfig(max_iterations=4),
        agent_id="main",
    )

    mock_llm.set_tool_call(
        "update_plan",
        {"plan": [
            {"step": "Inspect repo", "status": "in_progress"},
            {"step": "Report findings", "status": "pending"},
        ]},
    )
    mock_llm.set_response("Done.")

    async for _ in agent.run("Check this quickly"):
        pass

    assert events
    assert events[-1].source == "main"
    assert events[-1].data.get("agent_id") == "main"

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


class TestControlSafeguards:
    @pytest.mark.asyncio
    async def test_meta_turn_hides_use_skill(self, kernel, security):
        mock_llm = MockLLMProvider()
        manager = SkillManager(kernel)
        await manager.register(_OnDemandBrowserSkill())
        router = SkillRouter(manager)

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=3),
            router=router,
        )

        mock_llm.set_response("I should have planned before starting.")

        async for _ in agent.run("Why did you not plan before using the browser?"):
            pass

        names = {t.name for t in (mock_llm.last_tools or [])}
        assert USE_SKILL_TOOL not in names

    @pytest.mark.asyncio
    async def test_plan_gate_nudges_before_any_tool(self, kernel, security):
        mock_llm = MockLLMProvider()
        manager = SkillManager(kernel)

        @tool(name="greet")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        await manager.register(FunctionSkill("test", "Test", [greet]))

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=6),
        )

        mock_llm.set_tool_call("greet", {"name": "World"})
        mock_llm.set_tool_call(
            "update_plan",
            {"plan": [
                {"step": "Greet user", "status": "in_progress"},
                {"step": "Respond", "status": "pending"},
            ]},
        )
        mock_llm.set_tool_call("greet", {"name": "World"})
        mock_llm.set_response("Done after planning first.")

        chunks = []
        async for chunk in agent.run("Say hello to World."):
            chunks.append(chunk)

        response = "".join(chunks)
        assert "Done after planning first." in response
        msgs = agent.memory.get_messages(include_system=False)
        nudge_found = any(
            "create a short plan" in (m.content or "").lower()
            for m in msgs if m.role == "user"
        )
        assert nudge_found


    @pytest.mark.asyncio
    async def test_interrupted_plan_requires_replan_before_more_tool_work(self, kernel, security):
        mock_llm = MockLLMProvider()
        manager = SkillManager(kernel)

        @tool(name="greet")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        await manager.register(FunctionSkill("test", "Test", [greet]))

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=6),
        )

        await agent._planning.initialize(kernel, {})
        agent._planning_initialized = True
        await agent._planning.execute_tool(
            "update_plan",
            {"plan": [
                {"step": "Greet user", "status": "in_progress"},
                {"step": "Respond", "status": "pending"},
            ]},
        )
        await agent._planning.mark_interrupted(reason="user_interrupt")

        mock_llm.set_tool_call("greet", {"name": "World"})
        mock_llm.set_tool_call(
            "update_plan",
            {"plan": [
                {"step": "Greet user", "status": "in_progress"},
                {"step": "Respond", "status": "pending"},
            ]},
        )
        mock_llm.set_tool_call("greet", {"name": "World"})
        mock_llm.set_response("Done after explicit replan.")

        chunks = []
        async for chunk in agent.run("Actually say hello to World now."):
            chunks.append(chunk)

        response = "".join(chunks)
        assert "Done after explicit replan." in response
        assert mock_llm.all_calls
        first_system = mock_llm.all_calls[0]["messages"][0].content or ""
        assert "interrupted previous plan" in first_system.lower()
        msgs = agent.memory.get_messages(include_system=False)
        nudge_found = any(
            "create a short plan" in (m.content or "").lower()
            for m in msgs if m.role == "user"
        )
        assert nudge_found

    @pytest.mark.asyncio
    async def test_repeated_failures_force_explanation_mode(self, kernel, security):
        mock_llm = MockLLMProvider()
        manager = SkillManager(kernel)
        await manager.register(_AlwaysFailSkill())

        agent = AgentLoop(
            kernel=kernel,
            llm=mock_llm,
            skill_manager=manager,
            security=security,
            system_prompt="You are helpful.",
            config=AgentConfig(max_iterations=6),
        )

        mock_llm.set_tool_call(
            "update_plan",
            {"plan": [
                {"step": "Try browser", "status": "in_progress"},
                {"step": "Explain failure", "status": "pending"},
            ]},
        )
        mock_llm.set_tool_call("browser_go", {})
        mock_llm.set_tool_call("browser_go", {})
        mock_llm.set_response("The browser tool kept failing, so I stopped and am explaining.")

        chunks = []
        async for chunk in agent.run("Open the browser and check flights."):
            chunks.append(chunk)

        response = "".join(chunks)
        assert "explaining" in response.lower()
        msgs = agent.memory.get_messages(include_system=False)
        breaker_found = any(
            "recent tool attempts failed" in (m.content or "").lower()
            for m in msgs if m.role == "user"
        )
        assert breaker_found


@pytest.mark.asyncio
async def test_completed_plan_prunes_update_plan_history(kernel, security):
    mock_llm = MockLLMProvider()
    manager = SkillManager(kernel)

    @tool(name="greet")
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    await manager.register(FunctionSkill("test", "Test", [greet]))

    agent = AgentLoop(
        kernel=kernel,
        llm=mock_llm,
        skill_manager=manager,
        security=security,
        system_prompt="You are helpful.",
        config=AgentConfig(max_iterations=6),
    )

    mock_llm.set_tool_call(
        "update_plan",
        {"plan": [
            {"step": "Greet user", "status": "in_progress"},
            {"step": "Wrap up", "status": "pending"},
        ]},
    )
    mock_llm.set_tool_call("greet", {"name": "World"})
    mock_llm.set_tool_call(
        "update_plan",
        {"plan": [
            {"step": "Greet user", "status": "completed"},
            {"step": "Wrap up", "status": "completed"},
        ]},
    )
    mock_llm.set_response("Done.")

    async for _ in agent.run("Say hello to World and finish."):
        pass

    messages = agent.memory.get_messages(include_system=False)
    assert all(m.name != "update_plan" for m in messages if m.role == "tool")
    assert all(
        not m.tool_calls or all(tc.name != "update_plan" for tc in m.tool_calls)
        for m in messages
        if m.role == "assistant"
    )
    assert any(m.name == "greet" for m in messages if m.role == "tool")


@pytest.mark.asyncio
async def test_completed_plan_not_injected_on_next_turn(kernel, security):
    mock_llm = MockLLMProvider()
    manager = SkillManager(kernel)

    @tool(name="greet")
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    await manager.register(FunctionSkill("test", "Test", [greet]))

    agent = AgentLoop(
        kernel=kernel,
        llm=mock_llm,
        skill_manager=manager,
        security=security,
        system_prompt="You are helpful.",
        config=AgentConfig(max_iterations=6),
    )

    mock_llm.set_tool_call(
        "update_plan",
        {"plan": [
            {"step": "Greet user", "status": "in_progress"},
            {"step": "Wrap up", "status": "pending"},
        ]},
    )
    mock_llm.set_tool_call("greet", {"name": "World"})
    mock_llm.set_tool_call(
        "update_plan",
        {"plan": [
            {"step": "Greet user", "status": "completed"},
            {"step": "Wrap up", "status": "completed"},
        ]},
    )
    mock_llm.set_response("Done.")

    async for _ in agent.run("Say hello to World and finish."):
        pass

    mock_llm.set_response("Plain follow-up.")
    async for _ in agent.run("Just answer normally"):
        pass

    last_call = mock_llm.all_calls[-1]
    system_content = last_call["messages"][0].content or ""
    assert "## Your Current Plan" not in system_content


@pytest.mark.asyncio
async def test_main_agent_compaction_uses_provider_prompt_size(kernel, security):
    mock_llm = MockLLMProvider(context_window=1000)
    manager = SkillManager(kernel)
    agent = AgentLoop(
        kernel=kernel,
        llm=mock_llm,
        skill_manager=manager,
        security=security,
        system_prompt="You are helpful.",
        config=AgentConfig(max_iterations=4),
    )

    captured: list[tuple[int, int]] = []

    def fake_check_and_start_background(session, token_count, token_budget, llm):
        captured.append((token_count, token_budget))

    agent._compaction.check_and_start_background = fake_check_and_start_background

    mock_llm.set_chunks(
        [
            LLMChunk(text="Done."),
            LLMChunk(
                stop_reason=StopReason.COMPLETE,
                input_tokens=800,
                output_tokens=5,
            ),
        ]
    )

    async for _ in agent.run("Hi"):
        pass

    assert captured == [(800, 1000)]


class _SlowMockLLM(MockLLMProvider):
    def set_slow_response(self, parts: list[str], delay: float = 0.02) -> None:
        self._responses.append([(parts, delay)])

    async def generate(
        self,
        messages,
        tools=None,
        temperature: float = 0.7,
        max_tokens=None,
        stop_sequences=None,
    ):
        self.call_count += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        self.all_calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "call_number": self.call_count,
            }
        )

        queued = self._responses.pop(0)
        parts, delay = queued[0]
        for part in parts:
            await asyncio.sleep(delay)
            yield type("Chunk", (), {"text": part, "tool_calls": None, "stop_reason": None, "input_tokens": 0, "output_tokens": 0})()
        yield type("Chunk", (), {"text": None, "tool_calls": None, "stop_reason": StopReason.COMPLETE, "input_tokens": 1, "output_tokens": 1})()


@pytest.mark.asyncio
async def test_run_can_be_cancelled_mid_stream(kernel, skill_manager, security):
    from arc.core.run_control import RunControlAction, RunControlManager, RunStatus

    mock_llm = _SlowMockLLM()
    mock_llm.set_slow_response(["Hello", " ", "world", "!"])
    run_control = RunControlManager()
    agent = AgentLoop(
        kernel=kernel,
        llm=mock_llm,
        skill_manager=skill_manager,
        security=security,
        system_prompt="You are a helpful assistant.",
        config=AgentConfig(max_iterations=5),
        run_control=run_control,
    )

    chunks = []

    async def cancel_soon():
        while agent.current_run_id is None:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.025)
        assert run_control.request(agent.current_run_id, RunControlAction.CANCEL)

    cancel_task = asyncio.create_task(cancel_soon())
    async for chunk in agent.run("Hi"):
        chunks.append(chunk)
    await cancel_task

    snapshot = run_control.get_run(agent.last_run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.CANCELLED
    assert "Hello" in "".join(chunks)
    assert agent.state.status.value == "complete"
