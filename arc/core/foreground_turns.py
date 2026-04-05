from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import AsyncIterator, Any, Callable

from arc.core.events import Event, EventType
from arc.core.run_control import RunCancelledError, RunControlAction, RunControlManager, RunStatus


@dataclass(slots=True)
class ActiveTurn:
    source: str
    user_input_preview: str


@dataclass(slots=True)
class TurnOutcome:
    interrupted: bool = False
    reason: str | None = None
    run_id: str | None = None


class ForegroundTurnController:
    def __init__(
        self,
        *,
        agent: Any,
        run_control: RunControlManager,
        kernel: Any | None = None,
        system_prompt_for_source: Callable[[str], str | None] | None = None,
    ) -> None:
        self._agent = agent
        self._run_control = run_control
        self._kernel = kernel
        self._system_prompt_for_source = system_prompt_for_source
        self._lock = asyncio.Lock()
        self._active_turn: ActiveTurn | None = None
        self._last_outcome: TurnOutcome | None = None
        self._interrupt_pending = False
        self._interrupt_reason: str | None = None

    @property
    def active_turn(self) -> ActiveTurn | None:
        return self._active_turn

    @property
    def last_outcome(self) -> TurnOutcome | None:
        return self._last_outcome

    @property
    def is_active(self) -> bool:
        return self._active_turn is not None

    async def stream_message(self, user_input: str, *, source: str = "interactive") -> AsyncIterator[str]:
        async with self._lock:
            self._active_turn = ActiveTurn(
                source=source,
                user_input_preview=user_input[:200],
            )
            self._last_outcome = None
            self._interrupt_pending = False
            self._interrupt_reason = None
            system_prompt_override = None
            if self._system_prompt_for_source is not None:
                system_prompt_override = self._system_prompt_for_source(source)

            try:
                try:
                    try:
                        run_params = inspect.signature(self._agent.run).parameters.values()
                        supports_override = any(
                            param.kind == inspect.Parameter.VAR_KEYWORD
                            or param.name == "system_prompt_override"
                            for param in run_params
                        )
                    except (TypeError, ValueError):
                        supports_override = False
                    if supports_override:
                        run_iter = self._agent.run(
                            user_input,
                            system_prompt_override=system_prompt_override,
                        )
                    else:
                        run_iter = self._agent.run(user_input)
                    async for chunk in run_iter:
                        await self._apply_pending_interrupt_if_possible()
                        yield chunk
                except RunCancelledError:
                    pass
            finally:
                snapshot = None
                if getattr(self._agent, "last_run_id", None):
                    snapshot = self._run_control.get_run(self._agent.last_run_id)

                interrupted = snapshot is not None and snapshot.status == RunStatus.CANCELLED
                reason = None
                if interrupted:
                    reason = (snapshot.requested_action or RunControlAction.CANCEL).value

                self._last_outcome = TurnOutcome(
                    interrupted=interrupted,
                    reason=reason,
                    run_id=getattr(self._agent, "last_run_id", None),
                )
                self._active_turn = None
                self._interrupt_pending = False
                self._interrupt_reason = None

    async def interrupt_current(self, *, reason: str = "user_interrupt") -> bool:
        if self._active_turn is None:
            return False

        self._interrupt_pending = True
        self._interrupt_reason = reason
        await self._emit_interrupt_event(reason)

        for _ in range(100):
            run_id = getattr(self._agent, "current_run_id", None)
            if run_id:
                ok = self._run_control.request(run_id, RunControlAction.CANCEL)
                if ok:
                    return True
            if self._active_turn is None:
                return False
            await asyncio.sleep(0.01)

        return False

    async def _apply_pending_interrupt_if_possible(self) -> None:
        if not self._interrupt_pending:
            return
        run_id = getattr(self._agent, "current_run_id", None)
        if not run_id:
            return
        if self._run_control.request(run_id, RunControlAction.CANCEL):
            self._interrupt_pending = False

    async def _emit_interrupt_event(self, reason: str) -> None:
        if self._kernel is None:
            return
        await self._kernel.emit(Event(
            type=EventType.USER_INTERRUPT,
            source="foreground_turns",
            data={"reason": reason, "source": self._active_turn.source if self._active_turn else ""},
        ))
