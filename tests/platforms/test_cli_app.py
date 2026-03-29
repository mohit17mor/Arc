from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from arc.core.events import Event, EventType
from arc.platforms.cli.app import CLIPlatform


class _FakeApprovalFlow:
    def __init__(self) -> None:
        self.resolved: list[tuple[str, str]] = []

    def resolve_approval(self, request_id: str, decision: str) -> None:
        self.resolved.append((request_id, decision))


class _FakeEscalationBus:
    def __init__(self) -> None:
        self.resolved: list[tuple[str, str]] = []

    def resolve_escalation(self, escalation_id: str, answer: str) -> None:
        self.resolved.append((escalation_id, answer))


class _FakeTool:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


class _FakeManifest:
    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        tools: list[_FakeTool],
        *,
        always_available: bool,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.tools = tools
        self.always_available = always_available


class _FakeSkillManager:
    def __init__(self) -> None:
        self._manifests = {
            "core": _FakeManifest(
                "core",
                "1.0",
                "Always available helpers.",
                [_FakeTool("shell", "Run shell commands safely.")],
                always_available=True,
            ),
            "browser": _FakeManifest(
                "browser",
                "2.0",
                "Browser automation tools.",
                [_FakeTool("browser_go", "Navigate to a page."), _FakeTool("browser_look", "Inspect the DOM.")],
                always_available=False,
            ),
        }
        self.skill_names = set(self._manifests)
        self.tool_names = {tool.name for manifest in self._manifests.values() for tool in manifest.tools}

    def get_manifest(self, name: str):
        return self._manifests.get(name)


@pytest.mark.asyncio
async def test_cli_approval_prompt_suspends_interrupt_monitor_and_reads_input(monkeypatch):
    import threading

    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    flow = _FakeApprovalFlow()
    cli.set_approval_flow(flow)
    cli._interrupt_stop_event = threading.Event()

    monkeypatch.setattr(cli._session, "prompt", lambda _message: (_ for _ in ()).throw(AssertionError("prompt session should not be used for approvals")))
    monkeypatch.setattr("builtins.input", lambda: "y")

    event = Event(
        type=EventType.SECURITY_APPROVAL,
        source="security",
        data={
            "request_id": "approval_1",
            "tool_name": "write_file",
            "tool_description": "Write a file",
            "arguments": {"path": "/tmp/x.txt"},
            "capabilities": ["file:write"],
        },
    )

    await cli._handle_approval_request(event)

    assert cli._interrupt_stop_event.is_set()
    assert flow.resolved == [("approval_1", "allow_once")]


def test_cli_setters_store_dependencies():
    cli = CLIPlatform(console=Console(record=True, width=120))

    approval_flow = object()
    escalation_bus = object()
    skill_manager = object()
    skill_router = object()
    memory_manager = object()
    scheduler_store = object()
    pending_queue = asyncio.Queue()
    mcp_manager = object()
    workflow_skill = object()
    turn_controller = object()
    cost_tracker = {"requests": 1}

    cli.set_cost_tracker(cost_tracker)
    cli.set_approval_flow(approval_flow)
    cli.set_escalation_bus(escalation_bus)
    cli.set_skill_manager(skill_manager)
    cli.set_skill_router(skill_router)
    cli.set_memory_manager(memory_manager)
    cli.set_scheduler_store(scheduler_store)
    cli.set_pending_queue(pending_queue)
    cli.set_mcp_manager(mcp_manager)
    cli.set_workflow_skill(workflow_skill)
    cli.set_turn_controller(turn_controller)

    assert cli._cost_tracker is cost_tracker
    assert cli._approval_flow is approval_flow
    assert cli._escalation_bus is escalation_bus
    assert cli._skill_manager is skill_manager
    assert cli._skill_router is skill_router
    assert cli._memory_manager is memory_manager
    assert cli._scheduler_store is scheduler_store
    assert cli._pending_queue is pending_queue
    assert cli._mcp_manager is mcp_manager
    assert cli._workflow_skill is workflow_skill
    assert cli._turn_controller is turn_controller


def test_cli_renders_tool_results_workers_and_workflow_events():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    cli.on_event(Event(
        type=EventType.SKILL_TOOL_RESULT,
        source="main",
        data={"success": True, "output_preview": "This output is long enough to be truncated after sixty characters for display."},
    ))
    cli.on_event(Event(
        type=EventType.SKILL_TOOL_RESULT,
        source="main",
        data={"success": False, "output_preview": ""},
    ))
    cli.on_event(Event(type=EventType.AGENT_SPAWNED, source="main", data={"task_name": "research"}))
    cli.on_event(Event(type=EventType.AGENT_TASK_COMPLETE, source="main", data={"task_name": "research", "success": False}))
    cli.on_event(Event(type=EventType.WORKFLOW_START, source="main", data={"workflow": "deploy", "total_steps": 3}))
    cli.on_event(Event(type=EventType.WORKFLOW_STEP_START, source="main", data={"step": 1, "total_steps": 3, "instruction": "Build artifact"}))
    cli.on_event(Event(type=EventType.WORKFLOW_STEP_COMPLETE, source="main", data={"step": 1}))
    cli.on_event(Event(type=EventType.WORKFLOW_STEP_FAILED, source="main", data={"step": 2, "error": "boom"}))
    cli.on_event(Event(type=EventType.WORKFLOW_COMPLETE, source="main", data={"workflow": "deploy", "completed_steps": 2, "total_steps": 3}))
    cli.on_event(Event(
        type=EventType.WORKFLOW_PAUSED,
        source="main",
        data={
            "step": 2,
            "total_steps": 3,
            "instruction": "Deploy prod",
            "error": "approval needed",
            "completed_count": 1,
            "remaining": ["Verify", "Announce"],
        },
    ))

    text = console.export_text()
    assert "Done" in text
    assert "Worker 'research' started" in text
    assert "Worker 'research' done" in text
    assert "Workflow:" in text
    assert "Step 1/3" in text
    assert "Step 2 failed" in text
    assert "Workflow 'deploy' complete" in text
    assert "Workflow paused at step 2/3" in text


@pytest.mark.asyncio
async def test_cli_prompt_and_input_helpers_trim_and_track_prompt_state(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    prompt_messages: list[str] = []
    monkeypatch.setattr(cli._session, "prompt", lambda message: prompt_messages.append(message) or "  typed  ")

    prompt_line = await cli._prompt_line("Question? ")
    user_input = await cli._get_input()

    assert prompt_line == "  typed  "
    assert user_input == "typed"
    assert cli._interactive_prompt_active is False
    assert prompt_messages == ["Question? ", "\nYou > "]


@pytest.mark.asyncio
async def test_cli_get_input_returns_none_on_interrupt(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    monkeypatch.setattr(cli._session, "prompt", lambda _message: (_ for _ in ()).throw(KeyboardInterrupt()))

    result = await cli._get_input()

    assert result is None


def test_cli_inject_pending_results_adds_background_context():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    queue: asyncio.Queue = asyncio.Queue()
    cli.set_pending_queue(queue)
    now = time.time()

    queue.put_nowait(SimpleNamespace(
        job_name="nightly-index",
        fired_at=now,
        content="✅ Worker 'nightly-index' completed:\n\nIndexed 42 files.",
    ))
    queue.put_nowait(SimpleNamespace(
        job_name="daily-summary",
        fired_at=now,
        content="❌ Scheduled job daily-summary failed:\n\nNetwork timeout",
    ))

    injected = cli._inject_pending_results("What changed?")
    text = console.export_text()

    assert "background task(s) completed" in text
    assert "nightly-index" in injected
    assert "Indexed 42 files." in injected
    assert "Network timeout" in injected
    assert "Worker 'nightly-index' completed" not in injected
    assert queue.empty()


@pytest.mark.asyncio
async def test_cli_handle_command_renders_help_cost_skills_and_mcp():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    cli.set_skill_manager(_FakeSkillManager())
    cli.set_mcp_manager(SimpleNamespace(server_info=lambda: [
        {"name": "filesystem", "connected": True, "transport": "stdio", "tools": 3},
        {"name": "search", "connected": False, "transport": "http", "tools": 2},
    ]))
    cli.set_cost_tracker(
        {
            "requests": 2,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "worker_requests": 1,
            "worker_input_tokens": 10,
            "worker_output_tokens": 5,
            "worker_total_tokens": 15,
            "grand_total_tokens": 165,
            "cost_usd": 0.1234,
            "context_window": 1000,
            "turn_peak_input": 400,
        }
    )

    assert await cli._handle_command("/help") is True
    assert await cli._handle_command("/cost") is True
    assert await cli._handle_command("/skills") is True
    assert await cli._handle_command("/mcp") is True
    assert await cli._handle_command("/clear") is True
    assert await cli._handle_command("/perms") is True
    assert await cli._handle_command("/exit") is False

    text = console.export_text()
    assert "Commands" in text
    assert "Current Context" in text
    assert "Latest request" in text
    assert "Tier 1" in text
    assert "Tier 2" in text
    assert "MCP Servers" in text
    assert "Conversation cleared" in text
    assert "Permission memory" in text
    assert "Grand Total" not in text


@pytest.mark.asyncio
async def test_cli_handle_command_falls_back_when_features_unavailable():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    assert await cli._handle_command("/cost") is True
    assert await cli._handle_command("/skills") is True
    assert await cli._handle_command("/mcp") is True
    assert await cli._handle_command("/unknown") is True

    text = console.export_text()
    assert "Cost tracking not available" in text
    assert "No skill manager available" in text
    assert "No MCP servers configured" in text
    assert "Unknown command: /unknown" in text


@pytest.mark.asyncio
async def test_cli_handle_jobs_command_lists_and_cancels_jobs(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    deleted: list[str] = []
    job = SimpleNamespace(
        id="job-1",
        name="Nightly Sync",
        active=True,
        trigger={"kind": "cron"},
        next_run=time.time() + 60,
        prompt="sync the nightly datasets",
    )

    store = SimpleNamespace(
        get_all=AsyncMock(return_value=[job]),
        get_by_name=AsyncMock(return_value=job),
        delete=AsyncMock(side_effect=lambda job_id: deleted.append(job_id)),
    )
    cli.set_scheduler_store(store)
    monkeypatch.setattr("arc.scheduler.triggers.make_trigger", lambda trigger: SimpleNamespace(description="Every night"))

    await cli._handle_jobs_command("/jobs")
    await cli._handle_jobs_command("/jobs cancel Nightly Sync")

    text = console.export_text()
    assert "Scheduled Jobs" in text
    assert "Every night" in text
    assert "Cancelled job" in text
    assert deleted == ["job-1"]


@pytest.mark.asyncio
async def test_cli_handle_jobs_command_reports_missing_scheduler_and_errors():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    await cli._handle_jobs_command("/jobs")

    failing_store = SimpleNamespace(
        get_all=AsyncMock(side_effect=RuntimeError("list failed")),
        get_by_name=AsyncMock(return_value=None),
        delete=AsyncMock(),
    )
    cli.set_scheduler_store(failing_store)

    await cli._handle_jobs_command("/jobs")
    await cli._handle_jobs_command("/jobs cancel")

    missing_store = SimpleNamespace(
        get_all=AsyncMock(return_value=[]),
        get_by_name=AsyncMock(return_value=None),
        delete=AsyncMock(),
    )
    cli.set_scheduler_store(missing_store)
    await cli._handle_jobs_command("/jobs cancel missing")

    text = console.export_text()
    assert "Scheduler is not available" in text
    assert "Scheduler error: list failed" in text
    assert "Usage: /jobs cancel <name_or_id>" in text
    assert "No job found: missing" in text


@pytest.mark.asyncio
async def test_cli_handle_workflow_command_lists_and_streams():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    class _WorkflowSkill:
        async def execute_tool(self, name, args):
            assert name == "list_workflows"
            assert args == {}
            return SimpleNamespace(output="daily-report")

        async def stream_workflow(self, name, context):
            assert name == "deploy"
            assert context == "prod"
            for chunk in ("running ", "done"):
                yield chunk

    cli.set_workflow_skill(_WorkflowSkill())

    await cli._handle_workflow_command("/workflow")
    await cli._handle_workflow_command("/workflow deploy prod")

    text = console.export_text()
    assert "daily-report" in text
    assert "running done" in text


@pytest.mark.asyncio
async def test_cli_handle_workflow_command_reports_unavailable_engine():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    await cli._handle_workflow_command("/workflow list")

    assert "Workflow engine not available" in console.export_text()


@pytest.mark.asyncio
async def test_cli_handle_memory_command_supports_core_episodic_and_forget():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    memory_manager = SimpleNamespace(
        get_all_core=AsyncMock(return_value=[SimpleNamespace(id="fact-1", content="User likes tea", confidence=0.75)]),
        list_episodic=AsyncMock(return_value=[SimpleNamespace(id="ep-1", content="Discussed release plan", importance=0.8, created_at=time.time())]),
        delete_core=AsyncMock(),
    )
    cli.set_memory_manager(memory_manager)

    await cli._handle_memory_command("/memory")
    await cli._handle_memory_command("/memory episodic")
    await cli._handle_memory_command("/memory forget fact-1")

    text = console.export_text()
    assert "Core Memories" in text
    assert "User likes tea" in text
    assert "Recent Episodic Memories" in text
    assert "Discussed release plan" in text
    assert "Deleted core memory" in text


@pytest.mark.asyncio
async def test_cli_handle_memory_command_reports_missing_memory_and_errors():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    await cli._handle_memory_command("/memory")

    memory_manager = SimpleNamespace(
        get_all_core=AsyncMock(side_effect=RuntimeError("core failed")),
        list_episodic=AsyncMock(return_value=[]),
        delete_core=AsyncMock(side_effect=RuntimeError("delete failed")),
    )
    cli.set_memory_manager(memory_manager)

    await cli._handle_memory_command("/memory episodic")
    await cli._handle_memory_command("/memory forget")
    await cli._handle_memory_command("/memory forget fact-1")
    await cli._handle_memory_command("/memory")

    text = console.export_text()
    assert "Long-term memory is not available" in text
    assert "No episodic memories yet" in text
    assert "Usage: /memory forget <id>" in text
    assert "Memory error: delete failed" in text
    assert "Memory error: core failed" in text


@pytest.mark.asyncio
async def test_cli_process_message_renders_response_interrupt_and_cost(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console, agent_name="Arc")

    class _Controller:
        is_active = True
        last_outcome = SimpleNamespace(interrupted=True)

    cli.set_turn_controller(_Controller())
    cli.set_cost_tracker(
        {
            "turn_input_tokens": 12,
            "turn_output_tokens": 8,
            "turn_total_tokens": 20,
            "turn_requests": 2,
            "turn_peak_input": 50,
            "context_window": 200,
        }
    )

    async def fake_monitor(stop_event):
        while not stop_event.is_set():
            await asyncio.sleep(0)

    cli._monitor_escape_interrupt = fake_monitor  # type: ignore[method-assign]

    async def handler(_message: str):
        yield "Hello"
        yield " world"

    await cli._process_message("hi", handler)

    text = console.export_text()
    assert "Arc" in text
    assert "Hello world" in text
    assert "Interrupted." in text
    assert "50 / 200 ctx" in text
    assert "2 calls" in text


@pytest.mark.asyncio
async def test_cli_process_message_handles_tool_only_and_errors(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    async def fake_monitor(stop_event):
        while not stop_event.is_set():
            await asyncio.sleep(0)

    cli._monitor_escape_interrupt = fake_monitor  # type: ignore[method-assign]
    cli.set_turn_controller(SimpleNamespace(is_active=True, last_outcome=None))

    async def tool_only_handler(_message: str):
        cli._tool_call_count = 1
        if False:
            yield ""

    async def failing_handler(_message: str):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    await cli._process_message("tools only", tool_only_handler)
    await cli._process_message("bad", failing_handler)

    text = console.export_text()
    assert "Done." in text
    assert "Error: boom" in text


@pytest.mark.asyncio
async def test_cli_run_processes_workflow_resume_message_and_exit():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    processed_messages: list[str] = []
    provided_inputs: list[str] = []
    inputs = iter([None, "resume workflow", "hello agent", "/exit"])

    class _WorkflowSkill:
        def __init__(self) -> None:
            self.is_waiting_for_input = True

        def provide_input(self, value: str) -> None:
            provided_inputs.append(value)
            self.is_waiting_for_input = False

    cli.set_workflow_skill(_WorkflowSkill())

    async def fake_get_input():
        return next(inputs)

    async def fake_process_message(user_input: str, handler):
        processed_messages.append(user_input)

    async def fake_watcher_loop(handler):
        while True:
            await asyncio.sleep(0)

    cli._get_input = fake_get_input  # type: ignore[method-assign]
    cli._process_message = fake_process_message  # type: ignore[method-assign]
    cli._watcher_loop = fake_watcher_loop  # type: ignore[method-assign]

    async def handler(_message: str):
        yield "unused"

    await cli.run(handler)

    text = console.export_text()
    assert "Arc is ready." in text
    assert "Input received" in text
    assert "Goodbye!" in text
    assert provided_inputs == ["resume workflow"]
    assert processed_messages == ["hello agent"]


@pytest.mark.asyncio
async def test_cli_approval_waits_for_interrupt_monitor_shutdown_before_reading_input():
    import threading

    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    flow = _FakeApprovalFlow()
    cli.set_approval_flow(flow)
    cli._interrupt_stop_event = threading.Event()

    monitor_stopped = asyncio.Event()
    read_started_too_early = False

    async def fake_monitor() -> None:
        while not cli._interrupt_stop_event.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        monitor_stopped.set()

    async def fake_read_approval_response() -> str:
        nonlocal read_started_too_early
        read_started_too_early = not monitor_stopped.is_set()
        return "y"

    interrupt_task = asyncio.create_task(fake_monitor())
    cli._interrupt_task = interrupt_task
    cli._read_approval_response = fake_read_approval_response  # type: ignore[method-assign]

    event = Event(
        type=EventType.SECURITY_APPROVAL,
        source="security",
        data={
            "request_id": "approval_2",
            "tool_name": "write_file",
            "tool_description": "Write a file",
            "arguments": {"path": "/tmp/y.txt"},
            "capabilities": ["file:write"],
        },
    )

    await cli._handle_approval_request(event)
    await asyncio.wait_for(interrupt_task, timeout=1)

    assert cli._interrupt_task is None
    assert read_started_too_early is False
    assert flow.resolved == [("approval_2", "allow_once")]


@pytest.mark.asyncio
async def test_cli_approval_escape_dismisses_prompt_and_interrupts_turn():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    flow = _FakeApprovalFlow()
    cli.set_approval_flow(flow)

    class _Controller:
        is_active = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def interrupt_current(self, *, reason: str) -> bool:
            self.calls.append(reason)
            return True

    controller = _Controller()
    cli.set_turn_controller(controller)

    async def fake_read_approval_response() -> str:
        raise cli_module.PromptInputInterrupted()

    event = Event(
        type=EventType.SECURITY_APPROVAL,
        source="security",
        data={
            "request_id": "approval_escape",
            "tool_name": "browser_go",
            "tool_description": "Navigate browser",
            "arguments": {"url": "https://example.com"},
            "capabilities": ["browser"],
        },
    )

    from arc.platforms.cli import app as cli_module

    cli._read_approval_response = fake_read_approval_response  # type: ignore[method-assign]

    await cli._handle_approval_request(event)

    text = console.export_text()
    assert "User interrupted" in text
    assert "Permission prompt dismissed" in text
    assert flow.resolved == [("approval_escape", "deny")]
    assert controller.calls == ["cli_escape"]
    assert cli._waiting_for_approval is False


def test_cli_renders_main_agent_plan_updates():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    cli.on_event(Event(
        type=EventType.AGENT_PLAN_UPDATE,
        source="main",
        data={
            "plan": [
                {"step": "Inspect repo", "status": "completed"},
                {"step": "Patch bug", "status": "in_progress"},
                {"step": "Verify fix", "status": "pending"},
            ],
            "lifecycle_status": "active",
        },
    ))

    text = console.export_text()
    assert "Plan" in text
    assert "Patch bug" in text


def test_cli_browser_tool_call_shows_terminal_interrupt_hint():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    cli.on_event(Event(
        type=EventType.SKILL_TOOL_CALL,
        source='main',
        data={'tool': 'browser_go', 'arguments': {'url': 'example.com'}},
    ))

    text = console.export_text()
    assert 'browser_go' in text
    assert 'Press Esc in this terminal' in text


@pytest.mark.asyncio
async def test_cli_approval_restarts_interrupt_monitor_for_active_turn(monkeypatch):
    import threading

    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    flow = _FakeApprovalFlow()
    cli.set_approval_flow(flow)
    cli._interrupt_stop_event = threading.Event()

    started = asyncio.Event()

    class _Controller:
        is_active = True

    cli.set_turn_controller(_Controller())

    async def fake_monitor(stop_event):
        started.set()
        return None

    cli._monitor_escape_interrupt = fake_monitor  # type: ignore[method-assign]
    monkeypatch.setattr('builtins.input', lambda: 'y')

    event = Event(
        type=EventType.SECURITY_APPROVAL,
        source='security',
        data={
            'request_id': 'approval_restart',
            'tool_name': 'browser_go',
            'tool_description': 'Navigate browser',
            'arguments': {'url': 'https://example.com'},
            'capabilities': ['browser'],
        },
    )

    await cli._handle_approval_request(event)
    await asyncio.wait_for(started.wait(), timeout=1)

    assert cli._interrupt_task is not None
    assert flow.resolved == [('approval_restart', 'allow_once')]

    cli._interrupt_stop_event.set()
    await asyncio.gather(cli._interrupt_task, return_exceptions=True)
    cli._interrupt_task = None


@pytest.mark.asyncio
async def test_process_message_waits_for_restarted_interrupt_monitor_after_approval(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    flow = _FakeApprovalFlow()
    cli.set_approval_flow(flow)

    class _Controller:
        is_active = True
        last_outcome = None

    cli.set_turn_controller(_Controller())

    monitor_calls = 0
    restarted_started = asyncio.Event()
    restarted_release = asyncio.Event()
    restarted_task: asyncio.Task | None = None

    async def fake_monitor(stop_event):
        nonlocal monitor_calls, restarted_task
        monitor_calls += 1
        call_number = monitor_calls
        if call_number == 2:
            restarted_task = asyncio.current_task()
            restarted_started.set()
        while not stop_event.is_set():
            await asyncio.sleep(0)
        if call_number == 2:
            await restarted_release.wait()

    cli._monitor_escape_interrupt = fake_monitor  # type: ignore[method-assign]
    monkeypatch.setattr("builtins.input", lambda: "y")

    approval_event = Event(
        type=EventType.SECURITY_APPROVAL,
        source="security",
        data={
            "request_id": "approval_restart_during_turn",
            "tool_name": "browser_go",
            "tool_description": "Navigate browser",
            "arguments": {"url": "https://example.com"},
            "capabilities": ["browser"],
        },
    )

    async def handler(_message: str):
        approval_task = asyncio.create_task(cli._handle_approval_request(approval_event))
        await asyncio.wait_for(restarted_started.wait(), timeout=1)
        yield "done"
        await approval_task

    process_task = asyncio.create_task(cli._process_message("hello", handler))
    await asyncio.wait_for(restarted_started.wait(), timeout=1)
    await asyncio.sleep(0.05)

    assert restarted_task is not None
    assert process_task.done() is False
    assert restarted_task.done() is False

    restarted_release.set()
    await asyncio.wait_for(process_task, timeout=1)

    assert restarted_task.done() is True
    assert flow.resolved == [("approval_restart_during_turn", "allow_once")]


@pytest.mark.asyncio
async def test_process_message_waits_for_restarted_interrupt_monitor_after_escalation(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    bus = _FakeEscalationBus()
    cli.set_escalation_bus(bus)

    class _Controller:
        is_active = True
        last_outcome = None

    cli.set_turn_controller(_Controller())

    monitor_calls = 0
    restarted_started = asyncio.Event()
    restarted_release = asyncio.Event()
    restarted_task: asyncio.Task | None = None

    async def fake_monitor(stop_event):
        nonlocal monitor_calls, restarted_task
        monitor_calls += 1
        call_number = monitor_calls
        if call_number == 2:
            restarted_task = asyncio.current_task()
            restarted_started.set()
        while not stop_event.is_set():
            await asyncio.sleep(0)
        if call_number == 2:
            await restarted_release.wait()

    cli._monitor_escape_interrupt = fake_monitor  # type: ignore[method-assign]
    monkeypatch.setattr("builtins.input", lambda: "continue")

    escalation_event = Event(
        type=EventType.AGENT_ESCALATION,
        source="worker",
        data={
            "escalation_id": "worker_question_1",
            "from_agent": "worker",
            "question": "Should I continue?",
        },
    )

    async def handler(_message: str):
        escalation_task = asyncio.create_task(cli._handle_escalation(escalation_event))
        await asyncio.wait_for(restarted_started.wait(), timeout=1)
        yield "done"
        await escalation_task

    process_task = asyncio.create_task(cli._process_message("hello", handler))
    await asyncio.wait_for(restarted_started.wait(), timeout=1)
    await asyncio.sleep(0.05)

    assert restarted_task is not None
    assert process_task.done() is False
    assert restarted_task.done() is False

    restarted_release.set()
    await asyncio.wait_for(process_task, timeout=1)

    assert restarted_task.done() is True
    assert bus.resolved == [("worker_question_1", "continue")]
