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

    async def fake_read_plain_line() -> str:
        nonlocal read_started_too_early
        read_started_too_early = not monitor_stopped.is_set()
        return "y"

    interrupt_task = asyncio.create_task(fake_monitor())
    cli._interrupt_task = interrupt_task
    cli._read_plain_line = fake_read_plain_line  # type: ignore[method-assign]

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
