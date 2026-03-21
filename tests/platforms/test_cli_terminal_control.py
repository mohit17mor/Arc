from __future__ import annotations

import pytest

import threading

from rich.console import Console

from arc.platforms.cli.app import CLIPlatform


class _FakeTurnController:
    def __init__(self) -> None:
        self.is_active = True
        self.calls: list[str] = []

    async def interrupt_current(self, *, reason: str) -> bool:
        self.calls.append(reason)
        return True


def test_escape_monitor_ignores_input_while_waiting_for_approval(monkeypatch):
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    cli.set_turn_controller(_FakeTurnController())
    cli._waiting_for_approval = True

    monkeypatch.setattr("arc.platforms.cli.app.termios", object())
    monkeypatch.setattr("arc.platforms.cli.app.tty", object())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    consumed = {"read": 0}

    class _FakeOS:
        @staticmethod
        def read(fd: int, n: int) -> bytes:
            consumed["read"] += 1
            return b"y"

    monkeypatch.setattr("arc.platforms.cli.app.os", _FakeOS)

    class _FakeSelect:
        @staticmethod
        def select(r, w, x, timeout):
            return ([0], [], [])

    monkeypatch.setattr("arc.platforms.cli.app.select", _FakeSelect)

    class _FakeTermios:
        TCSADRAIN = 0

        @staticmethod
        def tcgetattr(fd):
            return []

        @staticmethod
        def tcsetattr(fd, when, settings):
            return None

    class _FakeTTY:
        @staticmethod
        def setcbreak(fd):
            return None

    monkeypatch.setattr("arc.platforms.cli.app.termios", _FakeTermios)
    monkeypatch.setattr("arc.platforms.cli.app.tty", _FakeTTY)

    stop_event = threading.Event()
    stop_event.set()
    pressed = cli._wait_for_escape_blocking(stop_event)

    assert pressed is False
    assert consumed["read"] == 0


def test_approval_reader_raises_interrupt_on_escape(monkeypatch):
    from arc.platforms.cli import app as cli_module

    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    class _FakeStdin:
        @staticmethod
        def fileno() -> int:
            return 0

        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _FakeStdin())

    class _FakeOS:
        @staticmethod
        def read(fd: int, n: int) -> bytes:
            return b"\x1b"

    monkeypatch.setattr("arc.platforms.cli.app.os", _FakeOS)

    class _FakeTermios:
        TCSADRAIN = 0

        @staticmethod
        def tcgetattr(fd):
            return []

        @staticmethod
        def tcsetattr(fd, when, settings):
            return None

    class _FakeTTY:
        @staticmethod
        def setraw(fd):
            return None

    monkeypatch.setattr("arc.platforms.cli.app.termios", _FakeTermios)
    monkeypatch.setattr("arc.platforms.cli.app.tty", _FakeTTY)

    with pytest.raises(cli_module.PromptInputInterrupted):
        cli._read_approval_response_blocking()


def test_cli_does_not_repeat_analyzing_status():
    from arc.core.events import Event, EventType

    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    cli.on_event(Event(type=EventType.AGENT_THINKING, source="main", data={"iteration": 1}))
    cli.on_event(Event(type=EventType.AGENT_THINKING, source="main", data={"iteration": 2}))
    cli.on_event(Event(type=EventType.AGENT_THINKING, source="main", data={"iteration": 3}))

    text = console.export_text()
    assert text.count("Thinking...") == 1
    assert text.count("Analyzing...") <= 1


def test_cli_renders_user_interrupt_feedback():
    from arc.core.events import Event, EventType

    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)

    cli.on_event(Event(type=EventType.USER_INTERRUPT, source='foreground_turns', data={'reason': 'cli_escape', 'source': 'cli'}))

    text = console.export_text()
    assert 'Interrupt requested' in text


@pytest.mark.asyncio
async def test_cli_interrupt_command_renders_feedback_immediately():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    cli.set_turn_controller(_FakeTurnController())

    await cli._handle_command('/interrupt')

    text = console.export_text()
    assert 'Interrupt requested' in text


def test_reset_state_clears_interrupt_flags():
    console = Console(record=True, width=120)
    cli = CLIPlatform(console=console)
    cli._interrupt_requested = True
    cli._interrupt_notice_shown = True

    cli._reset_state()

    assert cli._interrupt_requested is False
    assert cli._interrupt_notice_shown is False
