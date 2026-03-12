"""
PlanningSkill — structured plan tracking for all agent types.

Gives every agent (main, worker, scheduler, task) the ``update_plan``
tool so they can create / update a step-by-step execution plan.

The plan is stored on the skill instance and read back by the agent
loop on every iteration.  This means the plan **survives across
iterations** and is always re-injected into context — the LLM
literally cannot forget what it committed to.

Design:
    - Always available (``always_available=True``).
    - Zero capabilities required (no file/network access).
    - One instance per AgentLoop (each agent gets its own plan).
    - NOT excluded from any agent type — workers, task agents, etc.
      all benefit from structured planning.
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# Valid statuses for plan steps
_VALID_STATUSES = frozenset({"pending", "in_progress", "completed"})


class PlanningSkill(Skill):
    """
    Built-in skill providing the ``update_plan`` tool.

    Each agent gets its own PlanningSkill instance, so each agent
    tracks its own plan independently.
    """

    def __init__(self) -> None:
        self._kernel: Any = None
        self._config: dict = {}
        # The current plan — list of {step: str, status: str}
        self._plan: list[dict[str, str]] = []

    # ── Public API for AgentLoop to read ─────────────────────────────────

    @property
    def plan(self) -> list[dict[str, str]]:
        """Current plan steps. Read by AgentLoop at compose time."""
        return self._plan

    @property
    def has_plan(self) -> bool:
        return bool(self._plan)

    @property
    def has_incomplete_steps(self) -> bool:
        """True if any step is not yet completed."""
        return any(s["status"] != "completed" for s in self._plan)

    def format_plan_for_context(self) -> str:
        """
        Render the plan as a short text block for injection into context.

        Uses simple icons so every model can parse it:
          ✅  completed
          🔄  in_progress
          ⬚   pending
        """
        if not self._plan:
            return ""

        lines = ["## Your Current Plan"]
        for i, step in enumerate(self._plan, 1):
            status = step["status"]
            if status == "completed":
                icon = "✅"
            elif status == "in_progress":
                icon = "🔄"
            else:
                icon = "⬚"
            suffix = "  ← you are here" if status == "in_progress" else ""
            lines.append(f"{i}. {icon} {step['step']}{suffix}")
        return "\n".join(lines)

    # ── Skill ABC ────────────────────────────────────────────────────────

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="planning",
            version="1.0.0",
            description="Track execution progress with a structured step-by-step plan",
            capabilities=frozenset(),  # no permissions needed
            always_available=True,     # every agent sees this tool
            tools=(
                ToolSpec(
                    name="update_plan",
                    description=(
                        "Create or update your execution plan. "
                        "Call this BEFORE starting work on non-trivial tasks "
                        "to break the task into clear steps, then call it again "
                        "as you complete each step. "
                        "Rules: exactly ONE step may be 'in_progress' at a time. "
                        "Mark steps 'completed' as you finish them. "
                        "If plans change, update the plan before continuing."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": (
                                    "Optional: brief reason for creating or "
                                    "updating the plan."
                                ),
                            },
                            "plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {
                                            "type": "string",
                                            "description": (
                                                "Short description of the step "
                                                "(5-10 words max)."
                                            ),
                                        },
                                        "status": {
                                            "type": "string",
                                            "description": (
                                                "One of: pending, in_progress, "
                                                "completed."
                                            ),
                                        },
                                    },
                                    "required": ["step", "status"],
                                },
                                "description": "List of plan steps with statuses.",
                            },
                        },
                        "required": ["plan"],
                    },
                    required_capabilities=frozenset(),
                ),
            ),
        )

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name == "update_plan":
            return await self._update_plan(arguments)
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown tool: {tool_name}",
        )

    # ── Implementation ───────────────────────────────────────────────────

    async def _update_plan(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate and store the updated plan."""
        plan_items = arguments.get("plan")
        explanation = arguments.get("explanation", "")

        if not plan_items or not isinstance(plan_items, list):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "The 'plan' argument must be a non-empty array of "
                    "{step: string, status: string} objects."
                ),
            )

        # Validate each step
        validated: list[dict[str, str]] = []
        in_progress_count = 0

        for i, item in enumerate(plan_items):
            if not isinstance(item, dict):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Plan item {i} must be an object with 'step' and 'status'.",
                )

            step_text = str(item.get("step", "")).strip()
            status = str(item.get("status", "")).strip().lower()

            if not step_text:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Plan item {i}: 'step' must not be empty.",
                )

            if status not in _VALID_STATUSES:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"Plan item {i}: status '{status}' is invalid. "
                        f"Must be one of: pending, in_progress, completed."
                    ),
                )

            if status == "in_progress":
                in_progress_count += 1

            validated.append({"step": step_text, "status": status})

        if in_progress_count > 1:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Only ONE step may be 'in_progress' at a time. "
                    "Mark the current step 'completed' before starting the next."
                ),
            )

        # Store the plan
        self._plan = validated

        # Emit event for UI / worker log
        if self._kernel:
            from arc.core.events import Event, EventType

            await self._kernel.emit(Event(
                type=EventType.AGENT_PLAN_UPDATE,
                source="planning",
                data={
                    "plan": validated,
                    "explanation": explanation,
                    "all_completed": all(
                        s["status"] == "completed" for s in validated
                    ),
                },
            ))

        # Return formatted confirmation (LLM sees this)
        return ToolResult(
            success=True,
            output=self.format_plan_for_context(),
        )
