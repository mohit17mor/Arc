"""
SchedulerEngine — the background asyncio task that fires jobs.

Design:
- Polls the store every POLL_INTERVAL seconds
- For each due job: spins up a fresh AgentLoop (sub-agent) with full tool
  access, collects its complete response, then routes the result via the
  NotificationRouter (Telegram, file) AND puts it on the pending_queue
  so the main-agent CLI can inject it into the next conversation turn
- On startup: computes next_run for jobs that have next_run=0
- No missed-job replay: if Arc was off when a job was due, it is skipped
  and the next_run is advanced to the next scheduled time

Using an agent_factory (callable) rather than a hard-coded AgentLoop
instance makes it straightforward to swap in specialised agents per job
type in a future multi-agent setup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

from arc.scheduler.store import SchedulerStore
from arc.scheduler.triggers import make_trigger
from arc.scheduler.job import Job
from arc.notifications.base import Notification
from arc.notifications.router import NotificationRouter

if TYPE_CHECKING:
    from arc.agent.loop import AgentLoop
    from arc.llm.base import LLMProvider
    from arc.core.kernel import Kernel
    from arc.agent.registry import AgentRegistry

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30   # seconds between due-job checks


class SchedulerEngine:
    """
    Background scheduler.

    Execution modes per job (controlled by job.use_tools):

    use_tools=False (default):
        Plain LLM text generation — no tools, no security prompts, no surprises.
        Best for reminders, tips, summaries from LLM knowledge.

    use_tools=True:
        Full sub-agent with tool access (web search, file ops, etc.).
        Use for tasks that genuinely need live data, e.g. "fetch AI news at 8am".
        Future: this is where a specialist agent gets delegated to.
    """

    def __init__(
        self,
        store: SchedulerStore,
        llm: "LLMProvider",
        agent_factory: "Callable[[str], AgentLoop]",
        router: NotificationRouter,
        kernel: "Kernel | None" = None,
        agent_registry: "AgentRegistry | None" = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._agent_factory = agent_factory
        self._router = router
        self._kernel = kernel
        self._agent_registry = agent_registry
        self._task: asyncio.Task | None = None
        self._running = False
        self._in_flight: set[str] = set()  # job IDs currently executing

    async def start(self) -> None:
        """Start the background polling loop."""
        await self._compute_initial_next_runs()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="scheduler")
        logger.info("SchedulerEngine started")

    async def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SchedulerEngine stopped")

    # ── Internal loop ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.warning(f"Scheduler tick error (non-fatal): {e}")
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        """Check for due jobs and fire them."""
        now = time.time()
        due_jobs = await self._store.get_due_jobs(now=now)
        for job in due_jobs:
            if job.id in self._in_flight:
                logger.debug(f"Job {job.name!r} still executing, skipping tick")
                continue
            self._in_flight.add(job.id)
            task = asyncio.create_task(self._fire_job(job))
            # Register ALL jobs so list_workers and arc workers --follow can see them
            if self._agent_registry is not None:
                self._agent_registry.register_worker(f"scheduler:{job.name}", task)

    async def _fire_job(self, job: Job) -> None:
        """
        Execute a job in the appropriate mode:
          - use_tools=False: plain LLM call, no tools (safe default)
          - use_tools=True:  full sub-agent on a VirtualPlatform
            (same execution path as WorkerSkill so behaviour is identical)
        """
        logger.info(f"Firing scheduled job: {job.name!r} (id={job.id}, use_tools={job.use_tools})")
        now = int(time.time())
        # Use a distinct source per job so the worker log shows the job name
        source = f"scheduler:{job.name}"
        error: str | None = None
        try:
            if self._kernel:
                from arc.core.events import Event, EventType
                await self._kernel.emit(Event(
                    type=EventType.AGENT_SPAWNED,
                    source=source,
                    data={"task_id": job.id, "task_name": job.name, "use_tools": job.use_tools},
                ))
            if job.use_tools:
                from arc.agent.runner import run_agent_on_virtual_platform
                from arc.core.events import Event, EventType
                agent = self._agent_factory(source)  # agent_id = "scheduler:<job.name>"
                content, error = await run_agent_on_virtual_platform(
                    agent=agent,
                    prompt=job.prompt,
                    name=source,
                    timeout_seconds=300.0,  # 5 min — enough for web search + reads
                )
                if error:
                    logger.warning(f"Job {job.name!r} failed: {error}")
                    content = f"(job failed: {error})"
            else:
                content = await self._run_prompt(job.prompt)
        except Exception as e:
            logger.warning(f"Job {job.name!r} unexpected error: {e}")
            content = f"(job failed: {e})"
            error = str(e)

        # Emit completion event
        if self._kernel:
            from arc.core.events import Event, EventType
            await self._kernel.emit(Event(
                type=EventType.AGENT_TASK_COMPLETE,
                source=source,
                data={"task_id": job.id, "task_name": job.name, "success": error is None},
            ))

        # Update store BEFORE releasing _in_flight guard — prevents the scheduler
        # from picking up the job again in the tick window between discard and update.
        trigger = make_trigger(job.trigger)
        next_run = trigger.next_fire_time(last_run=now, now=now)
        if next_run == 0:
            await self._store.delete(job.id)
            logger.debug(f"Oneshot job {job.name!r} deleted after firing")
        else:
            await self._store.update_after_run(job.id, next_run=next_run, last_run=now)
            logger.debug(f"Job {job.name!r} next_run set to {next_run}")

        # Release guard only after store is updated
        self._in_flight.discard(job.id)

        notification = Notification(
            job_id=job.id,
            job_name=job.name,
            content=content,
            fired_at=now,
        )
        await self._router.route(notification)

    async def _run_prompt(self, prompt: str) -> str:
        """
        Plain LLM text generation — no tools, no approval prompts.
        Used by all use_tools=False jobs.
        """
        from arc.core.types import Message

        messages = [
            Message.system(
                "You are a helpful proactive assistant completing a scheduled task. "
                "Be concise and clear. Do not ask follow-up questions."
            ),
            Message.user(prompt),
        ]
        parts: list[str] = []
        async for chunk in self._llm.generate(
            messages=messages,
            tools=None,
            temperature=0.5,
        ):
            if chunk.text:
                parts.append(chunk.text)
        return "".join(parts).strip()

    # ── Startup helper ────────────────────────────────────────────────────────

    async def _compute_initial_next_runs(self) -> None:
        """
        On startup, populate next_run for jobs that have next_run=0.
        Also advance any jobs whose next_run is in the past (no replay —
        just push forward to the next scheduled time from now).
        """
        jobs = await self._store.get_all(active_only=True)
        now = time.time()
        for job in jobs:
            trigger = make_trigger(job.trigger)
            if job.next_run == 0 or job.next_run < now:
                # Compute next_run from now, not from last_run
                # This skips any missed executions while Arc was off
                next_run = trigger.next_fire_time(last_run=int(now), now=now)
                await self._store.update_after_run(
                    job.id, next_run=next_run, last_run=job.last_run
                )
                logger.debug(f"Job {job.name!r}: next_run initialised to {next_run}")
