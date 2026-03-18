from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class RunControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    FORCE_STOP = "force_stop"


class RunCancelledError(asyncio.CancelledError):
    def __init__(self, run_id: str, action: RunControlAction) -> None:
        super().__init__(f"Run {run_id} cancelled via {action.value}")
        self.run_id = run_id
        self.action = action


@dataclass(slots=True)
class RunSnapshot:
    run_id: str
    kind: str
    source: str
    status: RunStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    requested_action: RunControlAction | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass(slots=True)
class _RunState:
    snapshot: RunSnapshot
    resume_event: asyncio.Event


class RunHandle:
    def __init__(self, manager: "RunControlManager", run_id: str) -> None:
        self._manager = manager
        self.run_id = run_id

    async def checkpoint(self) -> None:
        await self._manager.checkpoint(self.run_id)

    async def finish_completed(self) -> None:
        self._manager.finish_completed(self.run_id)

    def finish_failed(self) -> None:
        self._manager.finish_failed(self.run_id)


class RunControlManager:
    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}

    def start_run(
        self,
        *,
        kind: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunHandle:
        run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        resume_event = asyncio.Event()
        resume_event.set()
        snapshot = RunSnapshot(
            run_id=run_id,
            kind=kind,
            source=source,
            status=RunStatus.RUNNING,
            metadata=dict(metadata or {}),
        )
        self._runs[run_id] = _RunState(snapshot=snapshot, resume_event=resume_event)
        return RunHandle(self, run_id)

    def get_run(self, run_id: str) -> RunSnapshot | None:
        state = self._runs.get(run_id)
        return None if state is None else state.snapshot

    def list_runs(self, *, active_only: bool = False) -> list[RunSnapshot]:
        runs = [state.snapshot for state in self._runs.values()]
        if active_only:
            runs = [run for run in runs if run.status not in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}]
        runs.sort(key=lambda run: run.created_at)
        return runs

    def request(self, run_id: str, action: RunControlAction) -> bool:
        state = self._runs.get(run_id)
        if state is None:
            return False

        snapshot = state.snapshot
        if snapshot.status in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}:
            return False

        snapshot.updated_at = time.time()
        snapshot.requested_action = action

        if action == RunControlAction.PAUSE and snapshot.status == RunStatus.RUNNING:
            snapshot.status = RunStatus.PAUSED
            state.resume_event.clear()
            return True

        if action == RunControlAction.RESUME and snapshot.status == RunStatus.PAUSED:
            snapshot.status = RunStatus.RUNNING
            snapshot.requested_action = None
            state.resume_event.set()
            return True

        if action in {RunControlAction.CANCEL, RunControlAction.FORCE_STOP}:
            snapshot.status = RunStatus.CANCELLING
            state.resume_event.set()
            return True

        return False

    async def checkpoint(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return

        while state.snapshot.status == RunStatus.PAUSED:
            await state.resume_event.wait()

        if state.snapshot.requested_action in {RunControlAction.CANCEL, RunControlAction.FORCE_STOP}:
            self._mark_cancelled(run_id)
            action = state.snapshot.requested_action or RunControlAction.CANCEL
            raise RunCancelledError(run_id, action)

    def finish_completed(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        if state.snapshot.status == RunStatus.CANCELLED:
            return
        state.snapshot.status = RunStatus.COMPLETED
        state.snapshot.completed_at = time.time()
        state.snapshot.updated_at = state.snapshot.completed_at

    def finish_failed(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        if state.snapshot.status == RunStatus.CANCELLED:
            return
        state.snapshot.status = RunStatus.FAILED
        state.snapshot.completed_at = time.time()
        state.snapshot.updated_at = state.snapshot.completed_at

    def _mark_cancelled(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        state.snapshot.status = RunStatus.CANCELLED
        state.snapshot.completed_at = time.time()
        state.snapshot.updated_at = state.snapshot.completed_at
        state.resume_event.set()
