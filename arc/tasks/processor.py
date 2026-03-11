"""
TaskProcessor — background daemon that processes queued tasks.

Runs inside ``arc gateway`` alongside the SchedulerEngine.  Polls
the TaskStore for actionable tasks, dispatches them to the correct
agent with the correct LLM, handles workflow progression (step
advancement, review routing, bounce-backs), and delivers results
via the NotificationRouter.

Design principles:
    - Agents are ephemeral — spun up per task execution, killed after.
    - The DB is the single source of truth — all state lives in tasks
      + comments tables.
    - One task at a time per agent (respects max_concurrent).
    - Status + comment writes always happen in one transaction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from arc.tasks.store import TaskStore
from arc.tasks.types import AgentDef, Task, TaskComment, TaskStatus

if TYPE_CHECKING:
    from arc.agent.loop import AgentLoop
    from arc.core.kernel import Kernel
    from arc.llm.base import LLMProvider
    from arc.notifications.router import NotificationRouter
    from arc.skills.manager import SkillManager

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5  # seconds between checks for new tasks

# Skills that task agents must never have — prevents recursion
_ALWAYS_EXCLUDED: frozenset[str] = frozenset({"worker", "scheduler", "task_board"})

# Max characters of comment history to inject as context.
# Older comments are summarized to stay within token budgets.
_MAX_CONTEXT_CHARS = 12_000
_FULL_COMMENT_COUNT = 4  # keep last N comments in full, summarize the rest


class TaskProcessor:
    """
    Background loop that processes queued tasks.

    Usage::

        processor = TaskProcessor(
            store=task_store,
            agents=agent_defs,
            skill_manager=skill_manager,
            default_llm=llm,
            notification_router=router,
            kernel=kernel,
        )
        await processor.start()
        ...
        await processor.stop()
    """

    def __init__(
        self,
        store: TaskStore,
        agents: dict[str, AgentDef],
        skill_manager: "SkillManager",
        default_llm: "LLMProvider",
        notification_router: "NotificationRouter",
        kernel: "Kernel",
        llm_factory: "Callable[..., LLMProvider] | None" = None,
    ) -> None:
        self._store = store
        self._agents = agents
        self._skill_manager = skill_manager
        self._default_llm = default_llm
        self._router = notification_router
        self._kernel = kernel
        self._llm_factory = llm_factory
        self._task: asyncio.Task | None = None
        self._running = False
        # Track in-flight tasks per agent to enforce max_concurrent
        self._in_flight: dict[str, int] = {}
        # Cache of per-agent LLM providers (created lazily)
        self._agent_llms: dict[str, "LLMProvider"] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="task_processor")
        logger.info(
            f"TaskProcessor started — {len(self._agents)} agent(s) registered"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Close per-agent LLM providers
        for llm in self._agent_llms.values():
            if hasattr(llm, "close"):
                await llm.close()
        self._agent_llms.clear()
        logger.info("TaskProcessor stopped")

    def reload_agents(self, agents: dict[str, AgentDef]) -> None:
        """Hot-reload agent definitions without restart."""
        self._agents = agents
        logger.info(f"TaskProcessor reloaded {len(agents)} agent(s)")

    # ── Main loop ────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.warning(f"TaskProcessor tick error (non-fatal): {e}")
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        """Check for actionable tasks and dispatch them."""
        agent_names = list(self._agents.keys())
        if not agent_names:
            return

        # Phase 1: dispatch queued / revision_needed tasks to their agents
        tasks = await self._store.get_actionable_tasks(agent_names)
        for task in tasks:
            agent_name = task.current_agent
            agent_def = self._agents.get(agent_name)
            if not agent_def:
                logger.warning(
                    f"Task {task.id} assigned to unknown agent '{agent_name}', skipping"
                )
                continue

            # Enforce max_concurrent
            current = self._in_flight.get(agent_name, 0)
            if current >= agent_def.max_concurrent:
                continue

            # Dispatch
            self._in_flight[agent_name] = current + 1
            asyncio.create_task(
                self._process_task(task, agent_def),
                name=f"task:{task.id}:{agent_name}",
            )

        # Phase 2: dispatch reviewer agents for tasks in IN_REVIEW
        await self._tick_reviews()

    # ── Task execution ───────────────────────────────────────────────────────

    async def _process_task(self, task: Task, agent_def: AgentDef) -> None:
        """Execute a single task with the given agent."""
        agent_name = agent_def.name
        try:
            logger.info(f"Processing task {task.id} ({task.title!r}) with agent '{agent_name}'")

            # Emit start event
            await self._emit("task:start", {
                "task_id": task.id,
                "task_title": task.title,
                "agent": agent_name,
                "step": task.current_step,
            })

            # Mark in_progress
            await self._store.update_status_with_comment(
                task.id,
                TaskStatus.IN_PROGRESS,
                "system",
                f"Agent '{agent_name}' picked up this task.",
                step_index=task.current_step,
                extra_updates={"started_at": int(time.time())},
            )

            # Build context from comments
            comments = await self._store.get_comments(task.id)
            context_text = self._build_context(task, comments)

            # Build the prompt
            prompt = self._build_prompt(task, agent_name, context_text)

            # Check if this is a reviewer step
            is_reviewer = self._is_review_step(task, agent_name)

            # Run the agent
            result, error = await self._run_agent(
                task, agent_def, prompt, is_reviewer
            )

            if error:
                await self._handle_failure(task, agent_name, error)
                return

            # Save the agent's output as a comment
            await self._store.add_comment(
                task.id, agent_name, result, step_index=task.current_step
            )

            # Route to next state
            if is_reviewer:
                await self._handle_review_result(task, agent_name, result)
            else:
                await self._advance_task(task, agent_name, result)

        except Exception as e:
            logger.error(f"Task {task.id} unexpected error: {e}", exc_info=True)
            await self._handle_failure(task, agent_name, str(e))
        finally:
            current = self._in_flight.get(agent_name, 1)
            self._in_flight[agent_name] = max(0, current - 1)

    def _is_review_step(self, task: Task, agent_name: str) -> bool:
        """Check if the current agent is the reviewer for the current step."""
        if not task.steps:
            return False
        step = task.steps[task.current_step]
        return step.review_by is not None and agent_name == step.review_by

    # ── Agent execution ──────────────────────────────────────────────────────

    async def _run_agent(
        self,
        task: Task,
        agent_def: AgentDef,
        prompt: str,
        is_reviewer: bool,
    ) -> tuple[str, str | None]:
        """Spin up a sub-agent and run it. Returns (result, error)."""
        from arc.agent.loop import AgentLoop, AgentConfig
        from arc.agent.runner import run_agent_on_virtual_platform
        from arc.security.engine import SecurityEngine
        from arc.skills.router import SkillRouter

        # Get or create LLM for this agent
        llm = await self._get_agent_llm(agent_def)

        # Compute excluded skills
        excluded = self._compute_excluded(agent_def)

        # Build system prompt
        system_prompt = agent_def.build_system_prompt()
        if is_reviewer:
            system_prompt += (
                "\n\nYou are reviewing another agent's work. "
                "Evaluate the quality, correctness, and completeness. "
                "At the END of your review, you MUST include exactly one of:\n"
                "  VERDICT: APPROVED\n"
                "  VERDICT: NEEDS_REVISION\n\n"
                "If NEEDS_REVISION, explain clearly what needs to change."
            )

        # Build agent with its own router
        router = SkillRouter(self._skill_manager, excluded_skills=excluded)
        agent = AgentLoop(
            kernel=self._kernel,
            llm=llm,
            skill_manager=self._skill_manager,
            security=SecurityEngine.make_permissive(self._kernel),
            system_prompt=system_prompt,
            config=AgentConfig(
                max_iterations=25,
                temperature=0.5,
                excluded_skills=excluded,
            ),
            memory_manager=None,
            agent_id=f"task:{task.id}:{agent_def.name}",
            router=router,
        )

        return await run_agent_on_virtual_platform(
            agent=agent,
            prompt=prompt,
            name=f"task-{task.id}-{agent_def.name}",
            timeout_seconds=300.0,
        )

    def _compute_excluded(self, agent_def: AgentDef) -> frozenset[str]:
        """Compute the set of skills to exclude for this agent."""
        base = set(_ALWAYS_EXCLUDED)

        if agent_def.skills is not None:
            # Whitelist mode: exclude everything NOT in the list
            all_skills = set(self._skill_manager.manifests.keys())
            base = all_skills - set(agent_def.skills)
            base.update(_ALWAYS_EXCLUDED)  # always exclude dangerous ones
        elif agent_def.exclude_skills:
            base.update(agent_def.exclude_skills)

        return frozenset(base)

    async def _get_agent_llm(self, agent_def: AgentDef) -> "LLMProvider":
        """Get the LLM provider for an agent (cached, lazy-created)."""
        if not agent_def.has_llm_override:
            return self._default_llm

        if agent_def.name in self._agent_llms:
            return self._agent_llms[agent_def.name]

        if self._llm_factory is None:
            logger.warning(
                f"Agent '{agent_def.name}' has LLM override but no factory — "
                f"using default LLM"
            )
            return self._default_llm

        llm = self._llm_factory(
            agent_def.llm_provider,
            model=agent_def.llm_model,
            base_url=agent_def.llm_base_url or None,
            api_key=agent_def.llm_api_key or None,
        )
        self._agent_llms[agent_def.name] = llm
        logger.info(
            f"Created LLM for agent '{agent_def.name}': "
            f"{agent_def.llm_provider}/{agent_def.llm_model}"
        )
        return llm

    # ── Prompt building ──────────────────────────────────────────────────────

    def _build_prompt(
        self, task: Task, agent_name: str, context_text: str,
    ) -> str:
        """Build the full prompt for the agent."""
        parts = [f"## Task: {task.title}", "", task.instruction]

        if context_text:
            parts.append("")
            parts.append("## Previous Activity on This Task")
            parts.append(context_text)

        if task.status == TaskStatus.REVISION_NEEDED:
            parts.append("")
            parts.append(
                "**The reviewer requested changes.** "
                "Read the reviewer's feedback above and address it in your revised output."
            )

        return "\n".join(parts)

    def _build_context(self, task: Task, comments: list[TaskComment]) -> str:
        """Format comment history for injection into agent context."""
        if not comments:
            return ""

        # Keep last N comments in full, summarize older ones
        if len(comments) <= _FULL_COMMENT_COUNT:
            lines = [f"[{c.agent_name}] {c.content}" for c in comments]
        else:
            old = comments[:-_FULL_COMMENT_COUNT]
            recent = comments[-_FULL_COMMENT_COUNT:]
            lines = [f"(... {len(old)} earlier entries summarized ...)"]
            for c in old:
                # One-line summary of each older comment
                preview = c.content[:150].replace("\n", " ")
                lines.append(f"  [{c.agent_name}] {preview}...")
            lines.append("")
            for c in recent:
                lines.append(f"[{c.agent_name}] {c.content}")

        text = "\n\n".join(lines)
        # Truncate if still too large
        if len(text) > _MAX_CONTEXT_CHARS:
            text = text[-_MAX_CONTEXT_CHARS:]
            text = "(... truncated ...)\n" + text

        return text

    # ── Workflow progression ─────────────────────────────────────────────────

    async def _advance_task(self, task: Task, agent_name: str, result: str) -> None:
        """Advance the task after a non-reviewer agent completes."""
        step = task.steps[task.current_step] if task.steps else None

        # Does this step have a reviewer?
        reviewer = step.review_by if step else None
        if reviewer and reviewer != "human":
            # Hand off to reviewer agent
            await self._store.update_status_with_comment(
                task.id,
                TaskStatus.IN_REVIEW,
                "system",
                f"Submitted for review by '{reviewer}'.",
                step_index=task.current_step,
            )
            # The reviewer agent will be picked up on the next tick
            # (get_actionable_tasks returns in_review tasks? No — we need
            # a separate query. Let's handle this by changing the task's
            # current_agent to the reviewer temporarily.)
            #
            # Actually, a cleaner approach: set status to 'queued' but
            # swap who's executing. The _tick method identifies the reviewer
            # by checking _is_review_step. But the task's current_agent
            # returns the step's agent_name, not the reviewer.
            #
            # Simplest fix: we store a separate "pending_reviewer" flag.
            # But even simpler: for the review phase, we temporarily set
            # assigned_agent to the reviewer. After review, we reset it.
            #
            # Actually the cleanest approach for the processor: the tick
            # also checks for IN_REVIEW tasks and routes them to the reviewer.
            return

        if reviewer == "human":
            # Human review step
            await self._store.update_status_with_comment(
                task.id,
                TaskStatus.AWAITING_HUMAN,
                "system",
                f"Awaiting human review.",
                step_index=task.current_step,
            )
            await self._notify(
                task,
                f"🔍 Task #{task.id} awaiting your review: {task.title}\n\n"
                f"Latest result by '{agent_name}':\n{result[:500]}",
            )
            return

        # No reviewer — advance to next step or complete
        await self._move_to_next_step(task, result)

    async def _move_to_next_step(self, task: Task, result: str) -> None:
        """Move to next step in workflow, or mark done if no more steps."""
        next_step = task.current_step + 1

        if task.steps and next_step < len(task.steps):
            # Advance to next step
            await self._store.update_status_with_comment(
                task.id,
                TaskStatus.QUEUED,
                "system",
                f"Step {task.current_step + 1} complete. Moving to step {next_step + 1}.",
                step_index=task.current_step,
                extra_updates={"current_step": next_step, "bounce_count": 0},
            )
        else:
            # All steps done — task complete
            now = int(time.time())
            await self._store.update_status_with_comment(
                task.id,
                TaskStatus.DONE,
                "system",
                "Task completed successfully.",
                step_index=task.current_step,
                extra_updates={"result": result, "completed_at": now},
            )
            await self._emit("task:complete", {
                "task_id": task.id,
                "task_title": task.title,
            })
            await self._notify(
                task,
                f"✅ Task completed: {task.title}\n\n{result[:1000]}",
            )

    async def _handle_review_result(
        self, task: Task, reviewer_name: str, review_text: str,
    ) -> None:
        """Process a reviewer's verdict."""
        approved = "VERDICT: APPROVED" in review_text.upper()

        if approved:
            await self._store.update_status_with_comment(
                task.id,
                TaskStatus.QUEUED,  # will be picked up for next step
                reviewer_name,
                review_text,
                step_index=task.current_step,
            )
            await self._move_to_next_step(task, review_text)
        else:
            # Bounce back
            if task.bounce_count >= task.max_bounces:
                # Hit bounce limit — complete with what we have
                now = int(time.time())
                await self._store.update_status_with_comment(
                    task.id,
                    TaskStatus.DONE,
                    "system",
                    f"Max review iterations ({task.max_bounces}) reached. "
                    f"Completing with last result.",
                    step_index=task.current_step,
                    extra_updates={"result": review_text, "completed_at": now},
                )
                await self._notify(
                    task,
                    f"⚠️ Task {task.title} completed after {task.max_bounces} "
                    f"review iterations (limit reached).\n\n{review_text[:500]}",
                )
            else:
                # Send back to the step's agent
                await self._store.update_status_with_comment(
                    task.id,
                    TaskStatus.REVISION_NEEDED,
                    reviewer_name,
                    review_text,
                    step_index=task.current_step,
                    extra_updates={"bounce_count": task.bounce_count + 1},
                )
                logger.info(
                    f"Task {task.id} bounced back (count: {task.bounce_count + 1}/"
                    f"{task.max_bounces})"
                )

    async def _handle_failure(
        self, task: Task, agent_name: str, error: str,
    ) -> None:
        """Handle agent execution failure."""
        now = int(time.time())
        await self._store.update_status_with_comment(
            task.id,
            TaskStatus.FAILED,
            "system",
            f"Agent '{agent_name}' failed: {error}",
            step_index=task.current_step,
            extra_updates={"completed_at": now},
        )
        await self._emit("task:failed", {
            "task_id": task.id,
            "task_title": task.title,
            "error": error,
        })
        await self._notify(
            task,
            f"❌ Task failed: {task.title}\nAgent: {agent_name}\nError: {error}",
        )

    # ── Human reply handling ─────────────────────────────────────────────────

    async def handle_human_reply(
        self, task_id: str, reply: str, action: str = "approve",
    ) -> str:
        """
        Process a human's reply to a blocked or awaiting_human task.

        Args:
            task_id: The task ID.
            reply: The human's text response.
            action: "approve" to advance, "revise" to bounce back.

        Returns:
            Status message for the responding platform.
        """
        task = await self._store.get_blocked_task(task_id)
        if not task:
            return f"Task {task_id} is not waiting for human input."

        if task.status == TaskStatus.AWAITING_HUMAN:
            if action == "approve":
                await self._store.add_comment(
                    task_id, "human", f"APPROVED: {reply}", task.current_step
                )
                # Get last agent result for passing forward
                comments = await self._store.get_comments(task_id)
                last_result = ""
                for c in reversed(comments):
                    if c.agent_name != "system" and c.agent_name != "human":
                        last_result = c.content
                        break
                await self._move_to_next_step(task, last_result)
                return f"Task {task_id} approved. Moving to next step."
            else:
                # Revise — bounce back to the step's agent
                if task.bounce_count >= task.max_bounces:
                    return (
                        f"Task {task_id} has already hit the bounce limit "
                        f"({task.max_bounces}). Cannot send back."
                    )
                await self._store.update_status_with_comment(
                    task_id,
                    TaskStatus.REVISION_NEEDED,
                    "human",
                    f"Revision requested: {reply}",
                    step_index=task.current_step,
                    extra_updates={"bounce_count": task.bounce_count + 1},
                )
                return f"Task {task_id} sent back for revision."

        elif task.status == TaskStatus.BLOCKED:
            # Agent asked a question — save answer and re-queue
            await self._store.update_status_with_comment(
                task_id,
                TaskStatus.QUEUED,
                "human",
                reply,
                step_index=task.current_step,
            )
            return f"Answer delivered to task {task_id}. It will resume shortly."

        return f"Task {task_id} is in an unexpected state: {task.status.value}"

    # ── Review routing helper ────────────────────────────────────────────────

    async def _tick_reviews(self) -> None:
        """Check for tasks in IN_REVIEW status and dispatch reviewers.

        Called alongside the main _tick. Separated for clarity.
        """
        tasks = await self._store.get_all(status="in_review")
        for task in tasks:
            reviewer = task.current_reviewer
            if not reviewer or reviewer == "human":
                continue
            agent_def = self._agents.get(reviewer)
            if not agent_def:
                logger.warning(f"Task {task.id} reviewer '{reviewer}' not found")
                continue
            current = self._in_flight.get(reviewer, 0)
            if current >= agent_def.max_concurrent:
                continue
            self._in_flight[reviewer] = current + 1
            asyncio.create_task(
                self._process_review(task, agent_def),
                name=f"review:{task.id}:{reviewer}",
            )

    async def _process_review(self, task: Task, reviewer_def: AgentDef) -> None:
        """Run a reviewer agent on a task."""
        try:
            comments = await self._store.get_comments(task.id)
            context_text = self._build_context(task, comments)
            prompt = self._build_prompt(task, reviewer_def.name, context_text)

            result, error = await self._run_agent(
                task, reviewer_def, prompt, is_reviewer=True,
            )

            if error:
                await self._handle_failure(task, reviewer_def.name, error)
                return

            await self._store.add_comment(
                task.id, reviewer_def.name, result, step_index=task.current_step,
            )
            await self._handle_review_result(task, reviewer_def.name, result)

        except Exception as e:
            logger.error(f"Review for task {task.id} failed: {e}", exc_info=True)
            await self._handle_failure(task, reviewer_def.name, str(e))
        finally:
            current = self._in_flight.get(reviewer_def.name, 1)
            self._in_flight[reviewer_def.name] = max(0, current - 1)

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _emit(self, event_type: str, data: dict) -> None:
        """Emit a kernel event."""
        from arc.core.events import Event
        await self._kernel.emit(Event(
            type=event_type,
            source="task_processor",
            data=data,
        ))

    async def _notify(self, task: Task, content: str) -> None:
        """Send a notification through the router."""
        from arc.notifications.base import Notification
        notification = Notification(
            job_id=task.id,
            job_name=f"task:{task.title}",
            content=content,
        )
        await self._router.route(notification)
