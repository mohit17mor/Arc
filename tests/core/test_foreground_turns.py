from __future__ import annotations

import asyncio

import pytest

from arc.core.run_control import RunControlAction, RunControlManager, RunStatus


class _SlowForegroundAgent:
    def __init__(self, run_control: RunControlManager) -> None:
        self._run_control = run_control
        self.current_run_id: str | None = None
        self.last_run_id: str | None = None

    async def run(self, user_input: str):
        handle = self._run_control.start_run(
            kind="agent",
            source="main",
            metadata={"agent_id": "main", "input_preview": user_input[:50]},
        )
        self.current_run_id = handle.run_id
        self.last_run_id = handle.run_id
        try:
            for chunk in ["hello", " ", "world", "!"]:
                await asyncio.sleep(0.02)
                await handle.checkpoint()
                yield chunk
            await handle.finish_completed()
        finally:
            self.current_run_id = None


class _PromptAwareForegroundAgent:
    def __init__(self, run_control: RunControlManager) -> None:
        self._run_control = run_control
        self.current_run_id: str | None = None
        self.last_run_id: str | None = None
        self.received_overrides: list[str | None] = []

    async def run(self, user_input: str, *, system_prompt_override: str | None = None):
        self.received_overrides.append(system_prompt_override)
        handle = self._run_control.start_run(
            kind="agent",
            source="main",
            metadata={"agent_id": "main", "input_preview": user_input[:50]},
        )
        self.current_run_id = handle.run_id
        self.last_run_id = handle.run_id
        try:
            yield "ok"
            await handle.finish_completed()
        finally:
            self.current_run_id = None


@pytest.mark.asyncio
async def test_controller_interrupts_active_turn():
    from arc.core.foreground_turns import ForegroundTurnController

    run_control = RunControlManager()
    agent = _SlowForegroundAgent(run_control)
    controller = ForegroundTurnController(agent=agent, run_control=run_control)

    chunks: list[str] = []

    async def consume() -> None:
        async for chunk in controller.stream_message("hello", source="cli"):
            chunks.append(chunk)

    task = asyncio.create_task(consume())

    deadline = asyncio.get_running_loop().time() + 1.0
    while controller.active_turn is None and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.005)

    assert controller.active_turn is not None
    await asyncio.sleep(0.03)
    assert await controller.interrupt_current(reason="cli_escape")

    await task

    outcome = controller.last_outcome
    assert outcome is not None
    assert outcome.interrupted is True
    assert outcome.reason == RunControlAction.CANCEL.value

    snapshot = run_control.get_run(agent.last_run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.CANCELLED
    assert "hello" in "".join(chunks)


@pytest.mark.asyncio
async def test_controller_serializes_foreground_turns():
    from arc.core.foreground_turns import ForegroundTurnController

    run_control = RunControlManager()
    agent = _SlowForegroundAgent(run_control)
    controller = ForegroundTurnController(agent=agent, run_control=run_control)

    order: list[str] = []

    async def first() -> None:
        async for _ in controller.stream_message("first", source="cli"):
            order.append("first")

    async def second() -> None:
        async for _ in controller.stream_message("second", source="cli"):
            order.append("second")

    task1 = asyncio.create_task(first())
    await asyncio.sleep(0.01)
    task2 = asyncio.create_task(second())

    await asyncio.gather(task1, task2)

    assert order[:1] == ["first"]
    assert order[-1:] == ["second"]
    runs = run_control.list_runs()
    assert len(runs) == 2
    assert all(run.status == RunStatus.COMPLETED for run in runs)


@pytest.mark.asyncio
async def test_controller_passes_source_specific_prompt_override():
    from arc.core.foreground_turns import ForegroundTurnController

    run_control = RunControlManager()
    agent = _PromptAwareForegroundAgent(run_control)
    controller = ForegroundTurnController(
        agent=agent,
        run_control=run_control,
        system_prompt_for_source=lambda source: "voice prompt" if source == "voice" else None,
    )

    async for _ in controller.stream_message("hello", source="voice"):
        pass

    assert agent.received_overrides == ["voice prompt"]


@pytest.mark.asyncio
async def test_controller_skips_prompt_override_for_default_sources():
    from arc.core.foreground_turns import ForegroundTurnController

    run_control = RunControlManager()
    agent = _PromptAwareForegroundAgent(run_control)
    controller = ForegroundTurnController(
        agent=agent,
        run_control=run_control,
        system_prompt_for_source=lambda source: "voice prompt" if source == "voice" else None,
    )

    async for _ in controller.stream_message("hello", source="cli"):
        pass

    assert agent.received_overrides == [None]
