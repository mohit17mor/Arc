"""
Browser Control Skill — interactive browser automation for Arc.

Tools:
    browser_go(url)              → navigate to URL, return page snapshot
    browser_look(extract?)       → snapshot current page state
    browser_act(actions)         → execute actions on current page

Design notes:
    - THREE tools, not one per element — the LLM makes page-level
      decisions ("fill form", "click button"), the engine handles
      individual element mechanics
    - Accessibility-tree-first: no screenshots, no vision — structured
      text is faster and cheaper than image analysis
    - Human handoff for obstacles: CAPTCHAs, login walls, cookie banners
      get escalated to the user via EscalationBus
    - Persistent profiles: cookies/sessions survive across conversations
    - Speed principle: ONE LLM call returns ALL actions for a page,
      engine executes them mechanically → 1-2 calls per page vs 8-12
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill

if TYPE_CHECKING:
    from arc.core.escalation import EscalationBus

logger = logging.getLogger(__name__)

# ━━━ Tool Specs ━━━

_BROWSER_GO_SPEC = ToolSpec(
    name="browser_go",
    description=(
        "Navigate to a URL in the browser and return a structured snapshot of "
        "the page. Use this to open websites for INTERACTION — clicking buttons, "
        "filling forms, navigating multi-step flows. "
        "For simply READING page content, prefer web_read instead (faster). "
        "The snapshot shows all interactive elements (forms, buttons, links) "
        "with numbered IDs you can reference in browser_act."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to navigate to (https:// added if missing)",
            },
        },
        "required": ["url"],
    },
    required_capabilities=frozenset([Capability.BROWSER]),
)

_BROWSER_LOOK_SPEC = ToolSpec(
    name="browser_look",
    description=(
        "Get a snapshot of the current browser page without navigating. "
        "Returns all interactive elements (forms, buttons, links) with "
        "numbered IDs. Use this after an action to see the updated page, "
        "or to re-examine the page before deciding next steps."
    ),
    parameters={
        "type": "object",
        "properties": {
            "extract": {
                "type": "string",
                "description": (
                    "Optional: what specific information to extract. "
                    "E.g. 'product prices', 'form fields', 'navigation links'. "
                    "If omitted, returns a full page snapshot."
                ),
            },
        },
        "required": [],
    },
    required_capabilities=frozenset([Capability.BROWSER]),
)

_BROWSER_ACT_SPEC = ToolSpec(
    name="browser_act",
    description=(
        "Execute one or more actions on the current browser page. "
        "Actions are executed sequentially. If one fails, execution stops "
        "and a new page snapshot is returned so you can course-correct.\n\n"
        "Action types:\n"
        "  click    - {type: 'click', target: '[id]' or 'element name'}\n"
        "  fill     - {type: 'fill', target: '[id]' or 'field name', value: 'text'} — "
        "works for text inputs, combobox dropdowns, autocomplete fields\n"
        "  fill_form - {type: 'fill_form', fields: {'[id]': 'value', ...}} — "
        "batch fill multiple fields at once (PREFERRED for forms)\n"
        "  select   - {type: 'select', target: '[id]', value: 'option'} — for native <select> dropdowns\n"
        "  check    - {type: 'check', target: '[id]', checked: true/false}\n"
        "  submit   - {type: 'submit'} — find and click the submit button\n"
        "  scroll   - {type: 'scroll', direction: 'down'|'up'}\n"
        "  back     - {type: 'back'} — go back in browser history\n"
        "  wait     - {type: 'wait', for: 'text or CSS selector'}\n\n"
        "CRITICAL: ALWAYS use [id] references (e.g., [3], [72]) from the page snapshot "
        "to target elements — this avoids ambiguity with fields that have similar names. "
        "For combobox dropdowns (trip type, cabin class, etc.), use 'fill' action — "
        "the engine will click to open and pick the matching option automatically.\n"
        "Prefer fill_form for forms — it fills ALL fields in one action."
    ),
    parameters={
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "description": "List of action objects to execute sequentially",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Action type: click, fill, fill_form, select, check, submit, scroll, back, wait",
                        },
                        "target": {
                            "type": "string",
                            "description": "Element to target — name, label, placeholder, or [id] from snapshot",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to fill/select (for fill, select actions)",
                        },
                        "fields": {
                            "type": "object",
                            "description": "For fill_form: mapping of 'Label' → 'value' for each field",
                        },
                        "direction": {
                            "type": "string",
                            "description": "For scroll: 'up' or 'down'",
                        },
                        "checked": {
                            "type": "boolean",
                            "description": "For check: true to check, false to uncheck",
                        },
                        "for": {
                            "type": "string",
                            "description": "For wait: text or CSS selector to wait for",
                        },
                    },
                    "required": ["type"],
                },
            },
        },
        "required": ["actions"],
    },
    required_capabilities=frozenset([Capability.BROWSER]),
)


class BrowserControlSkill(Skill):
    """
    Interactive browser automation skill for Arc.

    Provides three LLM-facing tools (browser_go, browser_look, browser_act)
    that wrap a BrowserEngine for full page interaction.

    Dependencies are injected via set_dependencies() after construction,
    following the same pattern as WorkerSkill.
    """

    def __init__(self) -> None:
        self._engine = None
        self._human = None
        self._escalation_bus: "EscalationBus | None" = None
        self._activated = False

    # ━━━ Lifecycle ━━━

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="browser_control",
            version="1.0.0",
            description="Interactive browser automation — navigate, fill forms, click buttons",
            capabilities=frozenset([Capability.BROWSER]),
            tools=(_BROWSER_GO_SPEC, _BROWSER_LOOK_SPEC, _BROWSER_ACT_SPEC),
        )

    def set_dependencies(
        self,
        escalation_bus: "EscalationBus | None" = None,
    ) -> None:
        """Inject dependencies after construction (CLI wiring pattern)."""
        self._escalation_bus = escalation_bus

    async def activate(self) -> None:
        """Launch the browser engine on first use."""
        if self._activated:
            return

        from arc.browser.engine import BrowserEngine
        from arc.browser.human import HumanAssist

        self._engine = BrowserEngine(headless=False)
        await self._engine.launch()

        self._human = HumanAssist(
            engine=self._engine,
            escalation_bus=self._escalation_bus,
        )

        self._activated = True
        logger.info("BrowserControlSkill activated — browser launched")

    async def shutdown(self) -> None:
        """Close the browser and release resources."""
        if self._engine:
            await self._engine.close()
            self._engine = None
        self._human = None
        self._activated = False
        logger.info("BrowserControlSkill shut down")

    # ━━━ Tool Dispatch ━━━

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Route tool calls to the appropriate handler."""
        if not self._activated:
            await self.activate()

        handlers = {
            "browser_go": self._handle_go,
            "browser_look": self._handle_look,
            "browser_act": self._handle_act,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        try:
            return await handler(arguments)
        except Exception as e:
            logger.exception(f"Browser tool {tool_name} failed")
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Browser error: {e}",
            )

    # ━━━ Tool Handlers ━━━

    async def _handle_go(self, args: dict[str, Any]) -> ToolResult:
        """Navigate to a URL and return a page snapshot."""
        url = args.get("url", "")
        if not url:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="URL is required",
            )

        snapshot = await self._engine.navigate(url)

        # Check for obstacles and handle them
        output = await self._process_snapshot(snapshot)

        return ToolResult(
            tool_call_id="",
            success=True,
            output=output,
        )

    async def _handle_look(self, args: dict[str, Any]) -> ToolResult:
        """Get a snapshot of the current page."""
        if not self._engine.current_url:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="No page loaded. Use browser_go first.",
            )

        snapshot = await self._engine.snapshot(force=True)
        extract = args.get("extract")

        output = await self._process_snapshot(snapshot)

        # If extract hint was given, add a note for the LLM
        if extract:
            output = f"[Looking for: {extract}]\n\n{output}"

        return ToolResult(
            tool_call_id="",
            success=True,
            output=output,
        )

    async def _handle_act(self, args: dict[str, Any]) -> ToolResult:
        """Execute actions on the current page."""
        actions = args.get("actions", [])
        if not actions:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="No actions provided",
            )

        if not self._engine.current_url:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="No page loaded. Use browser_go first.",
            )

        result = await self._engine.act(actions)

        # Build output
        parts = [result.summary]

        # Add the new page snapshot after actions
        if result.snapshot:
            snapshot_output = await self._process_snapshot(result.snapshot)
            parts.append(f"\n--- Page After Actions ---\n{snapshot_output}")

        return ToolResult(
            tool_call_id="",
            success=result.all_succeeded,
            output="\n".join(parts),
            error=None if result.all_succeeded else "Some actions failed (see output for details)",
        )

    # ━━━ Obstacle Handling ━━━

    async def _process_snapshot(self, snapshot) -> str:
        """
        Process a snapshot — handle obstacles if present, return text.
        """
        # Handle obstacles via human escalation
        if snapshot.obstacles and self._human:
            resolutions = await self._human.handle_obstacles(snapshot.obstacles)

            # Re-snapshot after human resolved the obstacles
            if resolutions:
                snapshot = await self._engine.snapshot(force=True)

                # Return the new snapshot. If there are still obstacles,
                # just note them but do NOT re-escalate (avoids infinite loop).
                if snapshot.obstacles:
                    obstacle_note = "\n⚠ Note: some obstacles may remain — "
                    obstacle_note += ", ".join(
                        f"{obs.type}" for obs in snapshot.obstacles
                    )
                    obstacle_note += (
                        ". These may be false positives (banners, analytics). "
                        "The page content is shown below — proceed if it looks normal.\n"
                    )
                    return obstacle_note + snapshot.to_text()

        return snapshot.to_text()
