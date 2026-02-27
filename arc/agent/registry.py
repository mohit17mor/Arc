"""
Agent Registry — lifecycle management for expert and worker agents.

Tracks:
- Expert agents (persistent within a session, each with their own VirtualPlatform)
- Worker tasks (ephemeral asyncio.Task handles)

Shutdown:
    await registry.shutdown_all()

cancels every running task and stops every expert platform cleanly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arc.agent.loop import AgentLoop
    from arc.platforms.virtual.app import VirtualPlatform

logger = logging.getLogger(__name__)


@dataclass
class ExpertEntry:
    """A running expert agent bound to a VirtualPlatform."""

    name: str
    loop: "AgentLoop"
    platform: "VirtualPlatform"
    task: asyncio.Task        # the asyncio.Task running platform.run()
    specialty: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class AgentRegistry:
    """
    Central registry for all sub-agents in a session.

    Expert agents:
        - Long-lived within the session
        - Each has a VirtualPlatform for message routing
        - Main agent routes user messages to them when active

    Worker tasks:
        - Ephemeral — one asyncio.Task per job
        - Registered so they can be cancelled on shutdown
        - Cleaned up automatically when they complete

    Usage::

        registry = AgentRegistry()

        # Spawn an expert
        registry.register_expert("research", loop, platform, task, specialty="web research")

        # Route a message to an expert
        response = await registry.send_to_expert("research", "summarise AI news today")

        # Clean up everything on exit
        await registry.shutdown_all()
    """

    def __init__(self) -> None:
        self._experts: dict[str, ExpertEntry] = {}
        self._worker_tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ #
    # Expert management                                                    #
    # ------------------------------------------------------------------ #

    def register_expert(
        self,
        name: str,
        loop: "AgentLoop",
        platform: "VirtualPlatform",
        task: asyncio.Task,
        specialty: str = "",
    ) -> None:
        """Register a running expert agent."""
        if name in self._experts:
            logger.warning(f"Expert '{name}' already registered — replacing")
            # Don't await here; caller is responsible for stopping old entry
        self._experts[name] = ExpertEntry(
            name=name,
            loop=loop,
            platform=platform,
            task=task,
            specialty=specialty,
        )
        logger.info(f"Expert '{name}' registered (specialty: {specialty or 'general'})")

    def get_expert(self, name: str) -> ExpertEntry | None:
        """Return the entry for a named expert, or None if not found."""
        return self._experts.get(name)

    def has_expert(self, name: str) -> bool:
        return name in self._experts

    def list_experts(self) -> list[ExpertEntry]:
        return list(self._experts.values())

    async def remove_expert(self, name: str) -> bool:
        """Stop and remove a named expert. Returns True if it existed."""
        entry = self._experts.pop(name, None)
        if entry is None:
            return False
        await self._stop_entry(entry)
        logger.info(f"Expert '{name}' removed")
        return True

    async def send_to_expert(self, name: str, message: str) -> str | None:
        """
        Route a message to a named expert and return its response.

        Returns None if the expert does not exist.
        """
        entry = self._experts.get(name)
        if entry is None:
            return None
        return await entry.platform.send_message(message)

    # ------------------------------------------------------------------ #
    # Worker task management                                               #
    # ------------------------------------------------------------------ #

    def list_worker_ids(self) -> list[str]:
        """Return ids of all currently-running (not yet done) worker tasks."""
        return [
            tid
            for tid, task in self._worker_tasks.items()
            if not task.done()
        ]

    def register_worker(self, task_id: str, task: asyncio.Task) -> None:
        """Track an ephemeral worker task."""
        self._worker_tasks[task_id] = task
        # Auto-remove when done so the dict stays clean
        task.add_done_callback(lambda _: self._worker_tasks.pop(task_id, None))
        logger.debug(f"Worker task '{task_id}' registered")

    def cancel_worker(self, task_id: str) -> bool:
        """Cancel a specific worker. Returns True if it existed."""
        task = self._worker_tasks.pop(task_id, None)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def list_workers(self) -> list[str]:
        return list(self._worker_tasks.keys())

    # ------------------------------------------------------------------ #
    # Shutdown                                                             #
    # ------------------------------------------------------------------ #

    async def shutdown_all(self) -> None:
        """
        Cancel all workers and stop all experts.

        Called on clean exit (Ctrl+C, /exit).
        Safe to call multiple times.
        """
        logger.info(
            f"AgentRegistry shutting down — "
            f"{len(self._worker_tasks)} workers, {len(self._experts)} experts"
        )

        # Cancel all worker tasks
        worker_tasks = list(self._worker_tasks.values())
        self._worker_tasks.clear()
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Stop all expert agents
        experts = list(self._experts.values())
        self._experts.clear()
        for entry in experts:
            await self._stop_entry(entry)

        logger.info("AgentRegistry shutdown complete")

    async def _stop_entry(self, entry: ExpertEntry) -> None:
        """Stop a single expert entry's platform and cancel its task."""
        try:
            await entry.platform.stop()
        except Exception as e:  # pragma: no cover
            logger.debug(f"Error stopping expert '{entry.name}' platform: {e}")

        if not entry.task.done():
            entry.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(entry.task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # expected

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgentRegistry("
            f"experts={list(self._experts)}, "
            f"workers={list(self._worker_tasks)})"
        )
