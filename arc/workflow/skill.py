"""
WorkflowSkill — exposes workflow execution as an Arc skill.

Provides two tools:
  - ``run_workflow``  — execute a named workflow
  - ``list_workflows`` — show available workflows

Workflows are loaded from ``~/.arc/workflows/*.yaml`` at activation time.
The agent can trigger them automatically when user input matches a
workflow's trigger pattern, or the user can ask for them explicitly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill
from arc.workflow.engine import WorkflowEngine
from arc.workflow.loader import load_workflows, match_workflow
from arc.workflow.types import Workflow

logger = logging.getLogger(__name__)

_WORKFLOWS_DIR = Path.home() / ".arc" / "workflows"


class WorkflowSkill(Skill):
    """
    Skill that runs deterministic YAML workflows.

    Dependencies (injected from bootstrap):
      - ``agent``: the main AgentLoop (for step execution)
    """

    def __init__(self) -> None:
        self._kernel: Any = None
        self._config: dict = {}
        self._agent: Any = None
        self._workflows: list[Workflow] = []
        self._engine: WorkflowEngine | None = None
        self._activated: bool = False

    def set_dependencies(self, *, agent: Any, kernel: Any) -> None:
        """Inject the AgentLoop and Kernel (called from bootstrap)."""
        self._agent = agent
        self._kernel = kernel
        self._engine = WorkflowEngine(agent=agent, kernel=kernel)

    def manifest(self) -> SkillManifest:
        # Build dynamic tool description with available workflow names
        workflow_names = ", ".join(w.name for w in self._workflows)
        if not workflow_names:
            workflow_names = "(no workflows found — add .yaml files to ~/.arc/workflows/)"

        # NOTE: Workflows are NEVER auto-triggered. The user must explicitly
        # ask to run a workflow (e.g., "/workflow jira-rca" or
        # "run the jira-rca workflow"). This avoids confusion between
        # a simple tool call and a full multi-step workflow.

        return SkillManifest(
            name="workflow",
            version="1.0.0",
            description=(
                "List available workflows. Workflows are run via the /workflow "
                "command — tell the user to use /workflow <name> to start one."
            ),
            capabilities=frozenset([Capability.FILE_READ]),
            tools=[
                ToolSpec(
                    name="list_workflows",
                    description=(
                        "List all available workflows with their descriptions. "
                        "Workflows are run via the /workflow command, not via this tool. "
                        "If the user wants to run a workflow, tell them to use: "
                        "/workflow <name>\n"
                        f"Available workflows: {workflow_names}"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ],
            always_available=False,
        )

    async def activate(self) -> None:
        """Load workflows from disk on first use."""
        self._workflows = load_workflows()
        self._activated = True
        logger.info(
            f"Loaded {len(self._workflows)} workflow(s): "
            f"{', '.join(w.name for w in self._workflows)}"
        )

    async def _ensure_activated(self) -> None:
        """Ensure workflows are loaded (handles direct calls via /workflow)."""
        if not getattr(self, "_activated", False):
            await self.activate()

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        await self._ensure_activated()
        if tool_name == "list_workflows":
            return await self._list_workflows()
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

    async def _list_workflows(self) -> ToolResult:
        """List all available workflows."""
        if not self._workflows:
            return ToolResult(
                success=True,
                output=(
                    "No workflows found.\n"
                    "To create one, add a .yaml file to ~/.arc/workflows/\n\n"
                    "Example workflow file:\n"
                    "  name: my-workflow\n"
                    "  trigger: \"keyword1|keyword2\"\n"
                    "  steps:\n"
                    "    - do the first thing\n"
                    "    - then do the second thing\n"
                    "    - finally, summarize the results"
                ),
            )

        lines = [f"Available workflows ({len(self._workflows)}):\n"]
        for wf in self._workflows:
            lines.append(f"  {wf.name} — {len(wf.steps)} steps")
            if wf.description:
                lines.append(f"    {wf.description}")
            if wf.trigger_patterns:
                lines.append(f"    Triggers: {', '.join(wf.trigger_patterns)}")
            lines.append("")

        return ToolResult(success=True, output="\n".join(lines))

    async def _run_workflow(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute a workflow by name."""
        name = arguments.get("name", "")
        context = arguments.get("context", "")

        if not self._engine:
            return ToolResult(
                success=False,
                output="",
                error="Workflow engine not initialized (agent not set)",
            )

        # Find the workflow
        workflow = next((w for w in self._workflows if w.name == name), None)
        if not workflow:
            available = ", ".join(w.name for w in self._workflows)
            return ToolResult(
                success=False,
                output="",
                error=f"Workflow '{name}' not found. Available: {available}",
            )

        # ── All workflows run as background tasks when triggered by the agent ──
        # This ensures:
        #   1. Explicit wait_for_input steps work (user can respond)
        #   2. Implicit pauses work (LLM decides to ask a question)
        #   3. Long workflows don't block the agent loop
        #   4. Output streams to the user in real-time via kernel events
        # This matches the /workflow command behavior exactly.
        import asyncio

        asyncio.create_task(
            self._run_interactive_workflow(workflow, context),
            name=f"workflow-{name}",
        )
        return ToolResult(
            success=True,
            output=(
                f"Workflow '{name}' has been started. "
                f"Progress will appear in the chat as it runs.\n\n"
                f"If the workflow needs input, it will ask — "
                f"the user's next message will be routed to it automatically.\n\n"
                f"Tell the user the workflow has started and to watch for updates."
            ),
        )

    async def _run_interactive_workflow(
        self, workflow: Workflow, context: str,
    ) -> None:
        """
        Run an interactive workflow as a background task.

        Output flows through kernel events (workflow:step_start,
        workflow:step_complete, workflow:waiting_input, etc.) which
        platforms display in real-time.  This method just drives the
        engine's async generator to completion.
        """
        try:
            async for _chunk in self._engine.run(workflow, user_message=context):
                pass  # text output goes via kernel events to platforms
            logger.info(f"Interactive workflow '{workflow.name}' completed")
        except Exception as e:
            logger.error(
                f"Interactive workflow '{workflow.name}' failed: {e}",
                exc_info=True,
            )

    def get_workflow(self, name: str) -> Workflow | None:
        """Get a workflow by name."""
        return next((w for w in self._workflows if w.name == name), None)

    async def stream_workflow(self, name: str, context: str = ""):
        """
        Stream a workflow's output chunk by chunk.

        Use this from /workflow commands for real-time progress.
        Yields text chunks as each step starts, completes, or fails.
        """
        await self._ensure_activated()

        if not self._engine:
            yield "Workflow engine not initialized\n"
            return

        workflow = self.get_workflow(name)
        if not workflow:
            available = ", ".join(w.name for w in self._workflows)
            yield f"Workflow '{name}' not found. Available: {available}\n"
            return

        async for chunk in self._engine.run(workflow, user_message=context):
            yield chunk

    @property
    def workflow_names(self) -> list[str]:
        """Names of all loaded workflows."""
        return [w.name for w in self._workflows]

    def provide_input(self, user_input: str) -> bool:
        """
        Resume a paused workflow with user input.

        Called by platforms (CLI, Gateway) when the user responds to a
        workflow's question.  Returns True if a workflow was waiting.
        """
        if self._engine is not None:
            return self._engine.provide_input(user_input)
        return False

    @property
    def is_waiting_for_input(self) -> bool:
        """True if a workflow is currently waiting for user input."""
        if self._engine is not None:
            return self._engine.is_waiting_for_input
        return False
