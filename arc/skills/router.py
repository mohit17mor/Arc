"""
Skill Router — two-tier tool selection for the agent.

Instead of sending ALL tool specs to the LLM on every call (~2,500 tokens),
the router exposes a compact menu:

    Tier 1 (always sent):
        - ``use_skill`` meta-tool — activates a skill category
        - Tools from skills marked ``always_available=True``

    Tier 2 (on demand):
        - Once the LLM calls ``use_skill("browsing")``, that skill's tools
          are injected into the next LLM call and stay active until
          ``reset()`` is called at the start of the **next user turn**.

Auto-builds from SkillManager manifests — zero manual wiring when new
skills are added.
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.types import SkillManifest, ToolSpec
from arc.skills.manager import SkillManager

logger = logging.getLogger(__name__)

# The meta-tool name that the LLM calls to activate a skill.
USE_SKILL_TOOL = "use_skill"


class SkillRouter:
    """
    Two-tier skill routing for the agent loop.

    Usage::

        router = SkillRouter(skill_manager, excluded_skills={"worker"})

        # Per LLM call — returns always-on tools + use_skill + activated tools
        tool_specs = router.get_active_tool_specs()

        # When LLM calls use_skill:
        msg = router.activate("browsing")   # → confirmation text / error

        # At start of next user turn:
        router.reset()
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        excluded_skills: frozenset[str] | None = None,
    ) -> None:
        self._manager = skill_manager
        self._excluded = excluded_skills or frozenset()
        self._activated: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_active_tool_specs(self) -> list[ToolSpec]:
        """
        Return the tool specs the LLM should see for the current call.

        Includes:
            1. Tools from ``always_available`` skills (not excluded)
            2. Tools from explicitly activated skills (via ``use_skill``)
            3. The ``use_skill`` meta-tool itself (with a dynamic menu)
        """
        specs: list[ToolSpec] = []
        seen_skills: set[str] = set()

        for name, manifest in self._visible_manifests().items():
            if manifest.always_available or name in self._activated:
                specs.extend(manifest.tools)
                seen_skills.add(name)

        # Build the use_skill meta-tool with remaining (not-yet-active) skills
        remaining = {
            n: m for n, m in self._visible_manifests().items()
            if n not in seen_skills
        }

        if remaining:
            specs.append(self._build_use_skill_spec(remaining))

        return specs

    def activate(self, skill_name: str) -> str:
        """
        Activate a skill so its tools appear in subsequent LLM calls.

        Returns a human-readable confirmation or error message.
        """
        visible = self._visible_manifests()

        if skill_name not in visible:
            available = sorted(visible.keys())
            return (
                f"Unknown skill '{skill_name}'. "
                f"Available skills: {', '.join(available)}"
            )

        manifest = visible[skill_name]

        if manifest.always_available:
            return (
                f"Skill '{skill_name}' is already available — "
                f"its tools are active by default."
            )

        if skill_name in self._activated:
            tool_names = [t.name for t in manifest.tools]
            return (
                f"Skill '{skill_name}' is already activated. "
                f"Tools: {', '.join(tool_names)}"
            )

        self._activated.add(skill_name)
        tool_names = [t.name for t in manifest.tools]
        logger.debug(f"Skill '{skill_name}' activated → tools: {tool_names}")
        return (
            f"Skill '{skill_name}' activated. "
            f"You now have access to: {', '.join(tool_names)}. "
            f"Call the tool you need."
        )

    def reset(self) -> None:
        """Clear activated skills — called at the start of each user turn."""
        if self._activated:
            logger.debug(f"Router reset — deactivated: {self._activated}")
        self._activated.clear()

    @property
    def activated_skills(self) -> set[str]:
        """Currently activated skill names (for inspection/testing)."""
        return set(self._activated)

    def is_use_skill_call(self, tool_name: str) -> bool:
        """Check if a tool call is the use_skill meta-tool."""
        return tool_name == USE_SKILL_TOOL

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _visible_manifests(self) -> dict[str, SkillManifest]:
        """All manifests minus excluded skills."""
        return {
            name: manifest
            for name, manifest in self._manager.manifests.items()
            if name not in self._excluded
        }

    def _build_use_skill_spec(
        self, remaining: dict[str, SkillManifest]
    ) -> ToolSpec:
        """Build the use_skill ToolSpec with a dynamic enum of skill names."""
        # Build the menu description — compact, one line per skill
        menu_lines = []
        for name, manifest in sorted(remaining.items()):
            tool_names = ", ".join(t.name for t in manifest.tools)
            menu_lines.append(f"  • {name} — {manifest.description} [{tool_names}]")

        menu_text = "\n".join(menu_lines)

        return ToolSpec(
            name=USE_SKILL_TOOL,
            description=(
                "Activate a skill category to access its tools. "
                "Call this FIRST to unlock the tools you need, "
                "then call the specific tool.\n\n"
                "Available skills:\n" + menu_text
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to activate",
                        "enum": sorted(remaining.keys()),
                    },
                },
                "required": ["skill_name"],
            },
            required_capabilities=frozenset(),
        )
