"""
EscalationBus — lets background agents ask the main agent a question.

When a worker or expert agent needs user input mid-task, it calls
``ask_manager(from_agent, question)`` which:

1. Emits an ``agent:escalation`` event on the kernel EventBus
2. Blocks (with timeout) waiting for an answer
3. CLIPlatform sees the event, shows the question to the user
4. User types their answer
5. CLIPlatform calls ``resolve_escalation(id, answer)``
6. ``ask_manager`` returns the answer and the worker continues

This mirrors the existing ``ApprovalFlow`` pattern exactly — same
Future-based handoff, same event-driven decoupling.

If no answer arrives within ``timeout`` seconds, ``ask_manager`` returns
a safe fallback string so the worker isn't stuck forever.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from arc.core.events import Event, EventType

if TYPE_CHECKING:
    from arc.core.kernel import Kernel

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300.0  # 5 minutes — same as ApprovalFlow


@dataclass
class EscalationRequest:
    """A pending escalation waiting for the main agent / user to answer."""

    escalation_id: str
    from_agent: str         # name of the agent that raised the question
    question: str
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )


class EscalationBus:
    """
    Request/response bridge between background agents and the main agent.

    Background agents (workers, experts, scheduler jobs) use this to surface
    a blocking question to the user without halting the whole event loop and
    without touching the terminal directly.

    Usage (in a worker skill)::

        answer = await escalation_bus.ask_manager(
            from_agent="worker-research",
            question="Which date range should I search? (e.g. 'last 7 days')",
        )

    Usage (in CLIPlatform event handler)::

        elif event.type == EventType.AGENT_ESCALATION:
            asyncio.create_task(self._handle_escalation(event))

        async def _handle_escalation(self, event):
            # show question to user, collect answer
            answer = ...
            self._escalation_bus.resolve_escalation(event.data["escalation_id"], answer)
    """

    def __init__(self, kernel: "Kernel", timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._kernel = kernel
        self._timeout = timeout
        self._pending: dict[str, EscalationRequest] = {}
        self._counter = 0

    # ------------------------------------------------------------------ #
    # Worker-side API                                                      #
    # ------------------------------------------------------------------ #

    async def ask_manager(self, from_agent: str, question: str) -> str:
        """
        Ask the main agent (and therefore the user) a question.

        Blocks until an answer is provided or the timeout expires.
        Returns the answer string, or a safe fallback on timeout.
        """
        self._counter += 1
        escalation_id = f"esc_{self._counter}"

        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()

        req = EscalationRequest(
            escalation_id=escalation_id,
            from_agent=from_agent,
            question=question,
            future=future,
        )
        self._pending[escalation_id] = req

        logger.info(
            f"Escalation {escalation_id} from '{from_agent}': {question[:80]}"
        )

        # Emit event — CLIPlatform (or any handler) will pick this up
        await self._kernel.emit(
            Event(
                type=EventType.AGENT_ESCALATION,
                source=from_agent,
                data={
                    "escalation_id": escalation_id,
                    "from_agent": from_agent,
                    "question": question,
                },
            )
        )

        try:
            # shield() so cancelling the outer task doesn't destroy the future
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            self._pending.pop(escalation_id, None)
            logger.warning(
                f"Escalation {escalation_id} timed out after {self._timeout}s"
            )
            return "[No answer received — proceeding with best judgement]"

    # ------------------------------------------------------------------ #
    # Main-agent-side API                                                  #
    # ------------------------------------------------------------------ #

    def resolve_escalation(self, escalation_id: str, answer: str) -> bool:
        """
        Resolve a pending escalation with the user's answer.

        Returns True if the escalation was found and resolved, False otherwise.
        """
        req = self._pending.pop(escalation_id, None)
        if req is None:
            logger.debug(f"Escalation {escalation_id} not found (already resolved?)")
            return False
        if req.future.done():
            return False
        req.future.set_result(answer)
        logger.info(f"Escalation {escalation_id} resolved")
        return True

    @property
    def pending(self) -> list[EscalationRequest]:
        """All escalations currently waiting for an answer."""
        return list(self._pending.values())

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)
