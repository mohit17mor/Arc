"""Tests for the Workflow Engine."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from arc.workflow.types import (
    OnFail,
    StepStatus,
    Workflow,
    WorkflowStep,
)
from arc.workflow.loader import (
    load_workflow_file,
    load_workflows,
    match_workflow,
    parse_workflow_from_dict,
)
from arc.workflow.engine import WorkflowEngine


# ━━━ Types tests ━━━


def test_step_defaults():
    """WorkflowStep has sensible defaults."""
    step = WorkflowStep(instruction="do something")
    assert step.retry == 0
    assert step.on_fail == OnFail.STOP
    assert step.ask_if_unclear is True
    assert step.tool is None
    assert step.status == StepStatus.PENDING


def test_workflow_creation():
    """Workflow can be created with steps."""
    steps = [
        WorkflowStep(instruction="step one", index=0),
        WorkflowStep(instruction="step two", index=1),
    ]
    wf = Workflow(name="test", steps=steps, trigger_patterns=["test|demo"])
    assert wf.name == "test"
    assert len(wf.steps) == 2
    assert wf.trigger_patterns == ["test|demo"]


# ━━━ Loader tests ━━━


def test_parse_simple_steps():
    """Simple string steps are parsed correctly."""
    data = {
        "name": "simple",
        "steps": [
            "do the first thing",
            "then do the second thing",
            "finally summarize",
        ],
    }
    wf = parse_workflow_from_dict(data)
    assert wf.name == "simple"
    assert len(wf.steps) == 3
    assert wf.steps[0].instruction == "do the first thing"
    assert wf.steps[1].instruction == "then do the second thing"
    assert wf.steps[2].instruction == "finally summarize"
    assert wf.steps[0].index == 0
    assert wf.steps[2].index == 2


def test_parse_extended_steps():
    """Extended dict steps are parsed correctly."""
    data = {
        "name": "extended",
        "steps": [
            {"do": "get the data", "retry": 3, "on_fail": "continue"},
            {"do": "analyze it", "ask_if_unclear": False},
            {"do": "run a command", "shell": "kubectl get pods"},
        ],
    }
    wf = parse_workflow_from_dict(data)
    assert len(wf.steps) == 3

    assert wf.steps[0].retry == 3
    assert wf.steps[0].on_fail == OnFail.CONTINUE

    assert wf.steps[1].ask_if_unclear is False

    assert wf.steps[2].shell == "kubectl get pods"


def test_parse_mixed_steps():
    """Simple and extended steps can coexist."""
    data = {
        "name": "mixed",
        "steps": [
            "do the simple thing",
            {"do": "do the complex thing", "retry": 2},
            "wrap it up",
        ],
    }
    wf = parse_workflow_from_dict(data)
    assert len(wf.steps) == 3
    assert wf.steps[0].instruction == "do the simple thing"
    assert wf.steps[1].retry == 2
    assert wf.steps[2].instruction == "wrap it up"


def test_parse_triggers():
    """Trigger patterns are parsed from string and list."""
    # String form
    wf1 = parse_workflow_from_dict({
        "name": "t1",
        "trigger": "investigate|debug|rca",
        "steps": ["do something"],
    })
    assert wf1.trigger_patterns == ["investigate|debug|rca"]

    # List form
    wf2 = parse_workflow_from_dict({
        "name": "t2",
        "trigger": ["investigate", "debug", "rca"],
        "steps": ["do something"],
    })
    assert len(wf2.trigger_patterns) == 3


def test_parse_no_steps_raises():
    """Workflow with no steps raises ValueError."""
    with pytest.raises(ValueError, match="no steps"):
        parse_workflow_from_dict({"name": "empty", "steps": []})


def test_parse_step_no_instruction_raises():
    """Extended step without 'do' raises ValueError."""
    with pytest.raises(ValueError, match="no 'do' instruction"):
        parse_workflow_from_dict({
            "name": "bad",
            "steps": [{"retry": 2}],
        })


def test_load_workflow_file(tmp_path):
    """Load a workflow from a YAML file."""
    yaml_content = """
name: test-flow
description: A test workflow
trigger: "test|demo"
steps:
  - fetch the data
  - analyze it
  - report the findings
"""
    f = tmp_path / "test-flow.yaml"
    f.write_text(yaml_content, encoding="utf-8")

    wf = load_workflow_file(f)
    assert wf is not None
    assert wf.name == "test-flow"
    assert wf.description == "A test workflow"
    assert len(wf.steps) == 3


def test_load_workflows_from_directory(tmp_path):
    """Load multiple workflows from a directory."""
    (tmp_path / "a.yaml").write_text(
        "name: alpha\nsteps:\n  - step one\n", encoding="utf-8"
    )
    (tmp_path / "b.yml").write_text(
        "name: beta\nsteps:\n  - step one\n", encoding="utf-8"
    )
    (tmp_path / "not-yaml.txt").write_text("ignore me", encoding="utf-8")

    workflows = load_workflows(tmp_path)
    assert len(workflows) == 2
    names = {w.name for w in workflows}
    assert names == {"alpha", "beta"}


def test_load_workflows_empty_directory(tmp_path):
    """Empty directory returns empty list."""
    assert load_workflows(tmp_path) == []


def test_load_workflows_nonexistent_directory():
    """Nonexistent directory returns empty list."""
    assert load_workflows(Path("/nonexistent/path")) == []


def test_match_workflow_by_trigger():
    """match_workflow matches based on regex trigger patterns."""
    wf1 = Workflow(
        name="rca",
        steps=[WorkflowStep(instruction="x", index=0)],
        trigger_patterns=["investigate|rca|root cause"],
    )
    wf2 = Workflow(
        name="deploy",
        steps=[WorkflowStep(instruction="x", index=0)],
        trigger_patterns=["deploy|release"],
    )

    assert match_workflow("investigate ticket PROJ-123", [wf1, wf2]) == wf1
    assert match_workflow("do an rca for payments", [wf1, wf2]) == wf1
    assert match_workflow("deploy to production", [wf1, wf2]) == wf2
    assert match_workflow("hello world", [wf1, wf2]) is None


def test_match_workflow_case_insensitive():
    """Trigger matching is case insensitive."""
    wf = Workflow(
        name="test",
        steps=[WorkflowStep(instruction="x", index=0)],
        trigger_patterns=["investigate"],
    )
    assert match_workflow("Please INVESTIGATE this", [wf]) == wf


# ━━━ Engine tests ━━━


def _make_mock_kernel():
    """Create a mock kernel with an async emit method."""
    kernel = MagicMock()
    kernel.emit = AsyncMock()
    return kernel


def _make_mock_agent(responses: list[str]):
    """Create a mock agent that yields predefined responses."""
    agent = MagicMock()
    call_count = 0

    async def mock_run(user_input: str):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        for word in responses[idx].split():
            yield word + " "

    agent.run = mock_run
    return agent


async def test_engine_runs_all_steps():
    """Engine executes all steps in order."""
    agent = _make_mock_agent([
        "Found ticket PROJ-123 about payment timeout",
        "Found 5 ERROR entries in logs",
        "Root cause: database connection pool exhausted",
    ])
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(instruction="get the ticket", index=0),
            WorkflowStep(instruction="search logs", index=1),
            WorkflowStep(instruction="find root cause", index=2),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    output = ""
    async for chunk in engine.run(wf, user_message="investigate PROJ-123"):
        output += chunk

    # All 3 steps should have been started and completed (via events)
    event_types = [c.args[0].type for c in kernel.emit.call_args_list]
    assert event_types.count("workflow:step_start") == 3
    assert event_types.count("workflow:step_complete") == 3
    assert "workflow:complete" in event_types
    # Agent output should contain the actual response text
    assert "ticket" in output.lower() or "PROJ" in output


async def test_engine_stops_on_fail():
    """Engine pauses at failed step when on_fail=stop and asks user for help."""
    call_count = 0

    async def failing_run(user_input: str):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("MCP server unreachable")
        yield "OK"

    agent = MagicMock()
    agent.run = failing_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(instruction="step one", index=0),
            WorkflowStep(instruction="step two (will fail)", index=1, on_fail=OnFail.STOP),
            WorkflowStep(instruction="step three (should not run)", index=2),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    output = ""
    async for chunk in engine.run(wf):
        output += chunk

    # Step progress is via events now, check kernel.emit was called
    event_types = [c.args[0].type for c in kernel.emit.call_args_list]
    assert "workflow:step_start" in event_types
    assert "workflow:step_failed" in event_types
    assert "workflow:paused" in event_types

    # Step 3 should never have started
    step_starts = [c.args[0].data["step"] for c in kernel.emit.call_args_list
                   if c.args[0].type == "workflow:step_start"]
    assert 3 not in step_starts

    # Paused event should contain help info
    paused_event = next(c.args[0] for c in kernel.emit.call_args_list
                        if c.args[0].type == "workflow:paused")
    assert paused_event.data["error"] == "MCP server unreachable"
    assert paused_event.data["step"] == 2
    assert len(paused_event.data["remaining"]) == 1


async def test_engine_continues_on_fail():
    """Engine continues past failed step when on_fail=continue."""
    call_count = 0

    async def failing_run(user_input: str):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("optional step failed")
        yield "OK"

    agent = MagicMock()
    agent.run = failing_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(instruction="step one", index=0),
            WorkflowStep(instruction="optional step", index=1, on_fail=OnFail.CONTINUE),
            WorkflowStep(instruction="step three", index=2),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    output = ""
    async for chunk in engine.run(wf):
        output += chunk

    event_types = [c.args[0].type for c in kernel.emit.call_args_list]
    assert "workflow:step_complete" in event_types  # step 1 completed
    assert "workflow:step_failed" in event_types    # step 2 failed
    # Step 3 should still run (on_fail=continue)
    step_starts = [c.args[0].data["step"] for c in kernel.emit.call_args_list
                   if c.args[0].type == "workflow:step_start"]
    assert 3 in step_starts  # step 3 was started
    assert "workflow:complete" in event_types  # workflow finished


async def test_engine_retry():
    """Engine retries a step on failure."""
    call_count = 0

    async def flaky_run(user_input: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient error")
        yield "Success on retry"

    agent = MagicMock()
    agent.run = flaky_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(instruction="flaky step", index=0, retry=1),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    output = ""
    async for chunk in engine.run(wf):
        output += chunk

    assert call_count == 2  # first attempt failed, second succeeded
    event_types = [c.args[0].type for c in kernel.emit.call_args_list]
    assert "workflow:step_complete" in event_types  # eventually succeeded


async def test_engine_context_passed_between_steps():
    """Previous step results are available as context for next steps."""
    prompts_received = []

    async def capture_run(user_input: str):
        prompts_received.append(user_input)
        yield "Step result data"

    agent = MagicMock()
    agent.run = capture_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(instruction="get data", index=0),
            WorkflowStep(instruction="analyze data", index=1),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    async for _ in engine.run(wf, user_message="original request"):
        pass

    # Step 2's prompt should contain results from step 1
    assert len(prompts_received) == 2
    assert "original request" in prompts_received[0]
    assert "Step result data" in prompts_received[1]


async def test_engine_ask_if_unclear_in_prompt():
    """ask_if_unclear=True adds clarification instruction to prompt."""
    prompts_received = []

    async def capture_run(user_input: str):
        prompts_received.append(user_input)
        yield "OK"

    agent = MagicMock()
    agent.run = capture_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(instruction="do something", index=0, ask_if_unclear=True),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    async for _ in engine.run(wf):
        pass

    assert "ask" in prompts_received[0].lower() or "clarif" in prompts_received[0].lower()


async def test_engine_explicit_shell_in_prompt():
    """Explicit shell command is included in the step prompt."""
    prompts_received = []

    async def capture_run(user_input: str):
        prompts_received.append(user_input)
        yield "OK"

    agent = MagicMock()
    agent.run = capture_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[
            WorkflowStep(
                instruction="check pods",
                index=0,
                shell="kubectl get pods -n payments",
            ),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    async for _ in engine.run(wf):
        pass

    assert "kubectl get pods" in prompts_received[0]


# ━━━ Human-in-the-loop tests ━━━


from arc.workflow.engine import _response_is_question


class TestQuestionDetection:
    """Tests for implicit question detection in step output."""

    def test_direct_question(self):
        assert _response_is_question("Which date range should I search?")

    def test_question_with_context(self):
        text = (
            "I found the relevant code. However, there are multiple modules "
            "that could be affected. Could you specify which module to focus on?"
        )
        assert _response_is_question(text)

    def test_not_a_question(self):
        assert not _response_is_question("I found 5 results and summarized them.")

    def test_rhetorical_not_detected(self):
        # Ends with ? but no question indicator phrase
        assert not _response_is_question("Great result, right?")

    def test_empty_string(self):
        assert not _response_is_question("")

    def test_please_provide(self):
        assert _response_is_question(
            "I need more details. Please provide the ticket ID?"
        )

    def test_before_i_proceed(self):
        assert _response_is_question(
            "The search returned many results. Before I proceed, "
            "which category are you interested in?"
        )

    def test_question_only_in_tail(self):
        """Long text with question at end is detected."""
        long_prefix = "Here is some analysis. " * 50
        text = long_prefix + "Would you like me to continue with the next step?"
        assert _response_is_question(text)


class TestWaitForInputFlag:
    """Tests for the wait_for_input WorkflowStep field."""

    def test_default_is_false(self):
        step = WorkflowStep(instruction="do something")
        assert step.wait_for_input is False

    def test_set_to_true(self):
        step = WorkflowStep(instruction="ask user", wait_for_input=True)
        assert step.wait_for_input is True

    def test_parsed_from_yaml(self, tmp_path):
        yaml_content = """
name: test-input
steps:
  - do: Ask the user which environment to deploy to
    wait_for_input: true
  - do: Deploy to the selected environment
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        wf = load_workflow_file(yaml_file)
        assert wf.steps[0].wait_for_input is True
        assert wf.steps[1].wait_for_input is False


class TestWorkflowPauseResume:
    """Tests for the pause/resume mechanism in WorkflowEngine."""

    @pytest.mark.asyncio
    async def test_explicit_wait_for_input_pauses(self):
        """Step with wait_for_input=True causes engine to pause."""
        kernel = _make_mock_kernel()

        responses = [
            "Which environment? staging or production?",
            "Deploying to staging now. Done.",
        ]
        agent = _make_mock_agent(responses)

        wf = Workflow(
            name="deploy",
            steps=[
                WorkflowStep(
                    instruction="Ask which environment",
                    index=0,
                    wait_for_input=True,
                ),
                WorkflowStep(instruction="Deploy", index=1),
            ],
        )

        engine = WorkflowEngine(agent, kernel)
        collected: list[str] = []

        async def run_workflow():
            async for chunk in engine.run(wf):
                collected.append(chunk)

        # Start the workflow — it will pause after step 1
        task = asyncio.create_task(run_workflow())

        # Wait a bit for step 1 to complete and pause
        await asyncio.sleep(0.1)
        assert engine.is_waiting_for_input

        # Provide user input — workflow resumes
        engine.provide_input("staging")
        await asyncio.sleep(0.1)

        # Wait for workflow to finish
        await asyncio.wait_for(task, timeout=5.0)

        full_output = "".join(collected)
        assert "staging" in full_output.lower() or "Deploying" in full_output

        # Verify waiting_input event was emitted
        emitted_types = [
            c.args[0].type for c in kernel.emit.call_args_list
        ]
        assert "workflow:waiting_input" in emitted_types

    @pytest.mark.asyncio
    async def test_implicit_question_pauses(self):
        """Agent asking a question implicitly pauses the workflow."""
        kernel = _make_mock_kernel()

        responses = [
            "I found multiple options. Which date range should I search?",
            "Searching last 7 days. Found results.",
        ]
        agent = _make_mock_agent(responses)

        wf = Workflow(
            name="search",
            steps=[
                WorkflowStep(
                    instruction="Search for relevant data",
                    index=0,
                    ask_if_unclear=True,
                ),
                WorkflowStep(instruction="Summarize results", index=1),
            ],
        )

        engine = WorkflowEngine(agent, kernel)
        collected: list[str] = []

        async def run_workflow():
            async for chunk in engine.run(wf):
                collected.append(chunk)

        task = asyncio.create_task(run_workflow())
        await asyncio.sleep(0.1)

        assert engine.is_waiting_for_input

        engine.provide_input("last 7 days")
        await asyncio.wait_for(task, timeout=5.0)

        full_output = "".join(collected)
        assert "Found results" in full_output or "results" in full_output.lower()

    @pytest.mark.asyncio
    async def test_no_pause_on_non_question(self):
        """Normal step completion doesn't pause."""
        kernel = _make_mock_kernel()
        agent = _make_mock_agent(["Done with step 1.", "Done with step 2."])

        wf = Workflow(
            name="simple",
            steps=[
                WorkflowStep(instruction="step one", index=0),
                WorkflowStep(instruction="step two", index=1),
            ],
        )

        engine = WorkflowEngine(agent, kernel)
        chunks = []
        async for chunk in engine.run(wf):
            chunks.append(chunk)

        # Should complete without pausing
        assert not engine.is_waiting_for_input
        emitted_types = [c.args[0].type for c in kernel.emit.call_args_list]
        assert "workflow:waiting_input" not in emitted_types
        assert "workflow:complete" in emitted_types

    @pytest.mark.asyncio
    async def test_provide_input_returns_false_when_not_waiting(self):
        """provide_input returns False when no workflow is waiting."""
        kernel = _make_mock_kernel()
        agent = MagicMock()
        engine = WorkflowEngine(agent, kernel)
        assert engine.provide_input("something") is False

    @pytest.mark.asyncio
    async def test_no_pause_on_last_step(self):
        """Even with wait_for_input=True, last step doesn't pause."""
        kernel = _make_mock_kernel()
        agent = _make_mock_agent(["Final answer."])

        wf = Workflow(
            name="single",
            steps=[
                WorkflowStep(
                    instruction="do it",
                    index=0,
                    wait_for_input=True,  # should be ignored on last step
                ),
            ],
        )

        engine = WorkflowEngine(agent, kernel)
        chunks = []
        async for chunk in engine.run(wf):
            chunks.append(chunk)

        assert not engine.is_waiting_for_input
        emitted_types = [c.args[0].type for c in kernel.emit.call_args_list]
        assert "workflow:waiting_input" not in emitted_types
        assert "workflow:complete" in emitted_types


async def test_engine_user_message_as_context():
    """Original user message is passed as context to step 1."""
    prompts_received = []

    async def capture_run(user_input: str):
        prompts_received.append(user_input)
        yield "OK"

    agent = MagicMock()
    agent.run = capture_run
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="test",
        steps=[WorkflowStep(instruction="investigate the ticket", index=0)],
    )

    engine = WorkflowEngine(agent, kernel)
    async for _ in engine.run(wf, user_message="investigate PAYMENTS-1234"):
        pass

    assert "PAYMENTS-1234" in prompts_received[0]


async def test_engine_emits_kernel_events():
    """Engine emits proper lifecycle events through the kernel."""
    agent = _make_mock_agent(["Result A", "Result B"])
    kernel = _make_mock_kernel()

    wf = Workflow(
        name="lifecycle-test",
        steps=[
            WorkflowStep(instruction="first step", index=0),
            WorkflowStep(instruction="second step", index=1),
        ],
    )

    engine = WorkflowEngine(agent, kernel)
    async for _ in engine.run(wf):
        pass

    events = [c.args[0] for c in kernel.emit.call_args_list]
    types = [e.type for e in events]

    # Lifecycle order: start → step_start → step_complete → step_start → step_complete → complete
    assert types[0] == "workflow:start"
    assert types[1] == "workflow:step_start"
    assert types[2] == "workflow:step_complete"
    assert types[3] == "workflow:step_start"
    assert types[4] == "workflow:step_complete"
    assert types[5] == "workflow:complete"

    # Check event data
    assert events[0].data["workflow"] == "lifecycle-test"
    assert events[0].data["total_steps"] == 2
    assert events[1].data["step"] == 1
    assert events[3].data["step"] == 2
    assert events[5].data["completed_steps"] == 2

    # All events sourced as "workflow"
    for e in events:
        assert e.source == "workflow"
