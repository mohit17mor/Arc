"""
WorkerSkill — lets the active agent delegate sub-tasks to background workers.

A worker is a short-lived AgentLoop that runs on a VirtualPlatform (silent —
no terminal output).

Flow:
    1. LLM calls delegate_task(task_name, prompt, ...)
    2. delegate_task fires an asyncio.Task and returns IMMEDIATELY with a
       confirmation message — the main agent is FREE to continue talking.
    3. Worker runs in the background, uses tools, produces a result.
    4. On completion the result is pushed through the NotificationRouter
       (same pipeline as scheduler jobs) → pending_queue → watcher loop
       → injected into the main agent's next conversation turn.
    5. Retry once on failure; second failure sends an error notification.

This keeps the main agent responsive exactly like a manager who delegates
and checks the result later, rather than waiting by the printer.

Tools:
    delegate_task(task_name, prompt, allowed_skills?)

Dependencies (injected via set_dependencies() from main.py):
    - kernel:             Kernel instance (for events + security)
    - llm:               LLMProvider (shared — stateless)
    - skill_manager:     SkillManager (shared — provides tool specs)
    - escalation_bus:    EscalationBus (for worker → user questions)
    - notification_router: NotificationRouter (delivers results back)
    - agent_registry:    AgentRegistry (tracks tasks for shutdown)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# Skills that workers must never have — prevents recursion and scheduling loops
_ALWAYS_EXCLUDED: frozenset[str] = frozenset({"worker", "scheduler"})


class WorkerSkill(Skill):
    """
    Skill that lets the LLM delegate sub-tasks to background workers.

    delegate_task() returns immediately — the main agent stays free.
    Results arrive via the notification pipeline when the worker finishes.
    """

    def __init__(self) -> None:
        self._kernel: Any = None
        self._config: dict = {}

        # Injected via set_dependencies()
        self._llm: Any = None
        self._skill_manager: Any = None
        self._escalation_bus: Any = None
        self._notification_router: Any = None
        self._agent_registry: Any = None
        self._worker_system_prompt: str = (
            "You are a focused background worker completing a specific sub-task. "
            "Do not ask clarifying questions — make your best effort with the "
            "information provided. Return a clear, structured result.\n\n"
            "Web research rules (STRICT):\n"
            "- Run ONE web_search, then read at most 2-3 of the most relevant URLs. Stop there.\n"
            "- Do NOT loop: search → read → search → read. One search is always enough.\n"
            "- For live data (prices, rates, weather) prefer http_get against a known API URL.\n"
            "- Once you have enough information, STOP calling tools and write your result.\n"
            "- If a page fails to load, skip it and use what you already have.\n\n"
            "Tool use rules:\n"
            "- Use the minimum number of tool calls needed to complete the task.\n"
            "- Never call the same tool twice with the same or similar arguments.\n"
            "- If you have sufficient information to answer, do not make more tool calls."
        )

    # ------------------------------------------------------------------ #
    # Dependency injection (called from main.py after wiring)             #
    # ------------------------------------------------------------------ #

    def set_dependencies(
        self,
        llm: Any,
        skill_manager: Any,
        escalation_bus: Any,
        notification_router: Any,
        agent_registry: Any,
        system_prompt: str | None = None,
    ) -> None:
        """Inject runtime dependencies. Must be called before first use."""
        self._llm = llm
        self._skill_manager = skill_manager
        self._escalation_bus = escalation_bus
        self._notification_router = notification_router
        self._agent_registry = agent_registry
        if system_prompt:
            self._worker_system_prompt = system_prompt

    # ------------------------------------------------------------------ #
    # Skill ABC                                                            #
    # ------------------------------------------------------------------ #

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="worker",
            version="1.0.0",
            description="Delegate a sub-task to a background worker agent and continue immediately",
            capabilities=frozenset([Capability.FILE_READ]),
            tools=[
                ToolSpec(
                    name="delegate_task",
                    description=(
                        "Spawn a background worker for a focused sub-task. "
                        "Returns IMMEDIATELY — you do NOT wait for the result. "
                        "The worker runs in the background and its result will be "
                        "delivered as a notification when ready — you do NOT need to "
                        "check or poll for it. "
                        "AFTER calling this tool: respond to the user in plain text "
                        "confirming what you delegated, then STOP — do NOT call any "
                        "more tools (especially not list_workers). "
                        "Example reply: 'I've started a background worker to fetch "
                        "today's AI news. I'll share the results as soon as it's done.' "
                        "Use this for tasks that need live data or take a long time: "
                        "web research, file analysis, multi-step data gathering."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "task_name": {
                                "type": "string",
                                "description": (
                                    "Short descriptive name, e.g. 'research_ai_news'. "
                                    "Shown to the user in progress and result notifications."
                                ),
                            },
                            "prompt": {
                                "type": "string",
                                "description": (
                                    "Full instructions for the worker. Include all context — "
                                    "the worker has no conversation history."
                                ),
                            },
                            "allowed_skills": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional: skill names the worker may use "
                                    "(e.g. ['browsing', 'filesystem']). "
                                    "Omit to give the worker all available skills "
                                    "except 'worker' and 'scheduler'."
                                ),
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": (
                                    "Wall-clock timeout in seconds. Default 120. "
                                    "Increase for long-running tasks "
                                    "(e.g. 600 for a 10-minute deep research task). "
                                    "Maximum allowed: 1800 (30 minutes)."
                                ),
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": (
                                    "Maximum LLM think-act cycles the worker may use. "
                                    "Default 15. Increase to 30-50 for tasks that need "
                                    "many tool calls (large file analysis, multi-step "
                                    "research). Maximum allowed: 50."
                                ),
                            },
                        },
                        "required": ["task_name", "prompt"],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
                ToolSpec(
                    name="list_workers",
                    description=(
                        "List all background workers that are still running. "
                        "Only call this when the user EXPLICITLY asks 'what workers "
                        "are running?' or 'what's in progress?'. "
                        "Do NOT call this after delegate_task — results arrive "
                        "automatically as notifications, no polling needed."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                    required_capabilities=frozenset(),
                ),
            ],
        )

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name == "delegate_task":
            return await self._delegate_task(
                task_name=arguments.get("task_name", "unnamed_task"),
                prompt=arguments.get("prompt", ""),
                allowed_skills=arguments.get("allowed_skills"),
                timeout_seconds=arguments.get("timeout_seconds", 120),
                max_iterations=arguments.get("max_iterations", 15),
            )
        if tool_name == "list_workers":
            return self._list_workers()
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown tool: {tool_name}",
        )

    # ------------------------------------------------------------------ #
    # Core implementation                                                  #
    # ------------------------------------------------------------------ #

    def _list_workers(self) -> ToolResult:
        """Return the ids of active background worker tasks."""
        if not self._agent_registry:
            return ToolResult(success=True, output="No worker registry available.")
        active = self._agent_registry.list_worker_ids()
        if not active:
            return ToolResult(success=True, output="No background workers are currently running.")
        lines = "\n".join(f"  • {tid}" for tid in active)
        return ToolResult(
            success=True,
            output=f"{len(active)} worker(s) running:\n{lines}",
        )

    # Hard ceilings — protect against runaway tasks
    _MAX_TIMEOUT: int = 1800   # 30 minutes
    _MAX_ITERATIONS: int = 50
    _DEFAULT_TIMEOUT: int = 300
    _DEFAULT_ITERATIONS: int = 20

    async def _delegate_task(
        self,
        task_name: str,
        prompt: str,
        allowed_skills: list[str] | None,
        timeout_seconds: int = 300,
        max_iterations: int = 20,
    ) -> ToolResult:
        """
        Fire a background worker and return immediately.

        The worker's result arrives via the notification pipeline.
        """
        if not self._llm or not self._skill_manager or not self._notification_router:
            return ToolResult(
                success=False,
                output="",
                error="WorkerSkill not initialised — call set_dependencies() first",
            )

        # Clamp to safe bounds
        timeout_seconds = max(10, min(int(timeout_seconds), self._MAX_TIMEOUT))
        max_iterations = max(1, min(int(max_iterations), self._MAX_ITERATIONS))

        task_id = f"{task_name}_{uuid.uuid4().hex[:8]}"
        excluded = self._compute_excluded(allowed_skills)

        logger.info(
            f"Spawning background worker '{task_id}' "
            f"(timeout={timeout_seconds}s, max_iter={max_iterations})"
        )

        # Emit event so CLIPlatform shows "⟳ Worker 'X' started"
        if self._kernel:
            from arc.core.events import Event, EventType
            await self._kernel.emit(Event(
                type=EventType.AGENT_SPAWNED,
                source="worker_skill",
                data={"task_id": task_id, "task_name": task_name},
            ))

        # Fire and forget — main agent continues immediately
        bg_task = asyncio.create_task(
            self._run_and_notify(
                task_id, task_name, prompt, excluded,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
            ),
            name=f"worker:{task_id}",
        )

        # Register with AgentRegistry so it is cancelled on shutdown
        if self._agent_registry:
            self._agent_registry.register_worker(task_id, bg_task)

        # Human-readable time string for the confirmation message
        if timeout_seconds >= 60:
            t_str = f"{timeout_seconds // 60}m {timeout_seconds % 60}s".replace(" 0s", "")
        else:
            t_str = f"{timeout_seconds}s"

        return ToolResult(
            success=True,
            output=(
                f"Worker '{task_name}' started (id: {task_id}). "
                f"Timeout: {t_str}, up to {max_iterations} iterations. "
                f"I'll notify you when it completes."
            ),
        )

    async def _run_and_notify(
        self,
        task_id: str,
        task_name: str,
        prompt: str,
        excluded: frozenset[str],
        timeout_seconds: int = 120,
        max_iterations: int = 15,
    ) -> None:
        """
        Background coroutine: run worker → retry once on failure → notify.
        """
        from arc.notifications.base import Notification

        # Attempt 1
        content, error = await self._run_worker(
            task_id, prompt, excluded,
            timeout_seconds=timeout_seconds,
            max_iterations=max_iterations,
        )

        # Retry once on failure (same limits)
        if error:
            logger.warning(f"Worker '{task_id}' attempt 1 failed: {error} — retrying")
            content, error = await self._run_worker(
                f"{task_id}_retry", prompt, excluded,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
            )

        # Build notification content
        if error:
            logger.error(f"Worker '{task_id}' failed after retry: {error}")
            notification_content = f"❌ Worker '{task_name}' failed: {error}"
        else:
            notification_content = f"✅ Worker '{task_name}' completed:\n\n{content or '(no output)'}"

        # Emit completion event for CLIPlatform progress display
        if self._kernel:
            from arc.core.events import Event, EventType
            await self._kernel.emit(Event(
                type=EventType.AGENT_TASK_COMPLETE,
                source="worker_skill",
                data={
                    "task_id": task_id,
                    "task_name": task_name,
                    "success": error is None,
                },
            ))

        # Route result through the same pipeline as scheduler notifications
        notification = Notification(
            job_id=task_id,
            job_name=task_name,
            content=notification_content,
        )
        await self._notification_router.route(notification)
        logger.info(f"Worker '{task_id}' result delivered via notification router")

    async def _run_worker(
        self,
        task_id: str,
        prompt: str,
        excluded: frozenset[str],
        timeout_seconds: int = 120,
        max_iterations: int = 15,
    ) -> tuple[str, str | None]:
        """
        Run one worker attempt on a VirtualPlatform.

        Returns (result_str, None) on success,
                ("", error_message) on failure.
        """
        from arc.agent.loop import AgentLoop, AgentConfig
        from arc.agent.runner import run_agent_on_virtual_platform
        from arc.security.engine import SecurityEngine

        # Extract a short readable label from the task_id  ("research_a1b2c3d4" → "research")
        parts = task_id.rsplit("_", 1)
        task_label = parts[0] if len(parts) == 2 and len(parts[1]) == 8 else task_id

        agent = AgentLoop(
            kernel=self._kernel,
            llm=self._llm,
            skill_manager=self._skill_manager,
            security=SecurityEngine.make_permissive(self._kernel),
            system_prompt=self._worker_system_prompt,
            config=AgentConfig(
                max_iterations=max_iterations,
                temperature=0.4,
                excluded_skills=excluded,
            ),
            memory_manager=None,
            agent_id=f"worker:{task_label}",
        )
        return await run_agent_on_virtual_platform(
            agent=agent,
            prompt=prompt,
            name=task_id,
            timeout_seconds=float(timeout_seconds),
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _compute_excluded(self, allowed_skills: list[str] | None) -> frozenset[str]:
        """
        Compute the excluded_skills frozenset for a worker.

        If allowed_skills is given, exclude everything NOT in that list
        (plus the always-excluded set).
        If omitted, only exclude the always-excluded skills.
        """
        if allowed_skills is None:
            return _ALWAYS_EXCLUDED

        all_names: set[str] = set()
        if self._skill_manager:
            all_names = set(self._skill_manager.skill_names)

        allowed = frozenset(allowed_skills) - _ALWAYS_EXCLUDED
        return frozenset(all_names - allowed) | _ALWAYS_EXCLUDED
