from __future__ import annotations

import asyncio
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
