"""
Skill Manager — registration, lifecycle, and tool routing.

The manager is the single point of access for all skills.
It handles lazy activation and routes tool calls to the right skill.
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.errors import SkillError
from arc.core.types import ToolResult, ToolSpec
from arc.skills.base import Skill

logger = logging.getLogger(__name__)


class SkillManager:
    """
    Manages skill registration, lifecycle, and tool routing.

    Usage:
        manager = SkillManager(kernel)
        await manager.register(filesystem_skill)
        await manager.register(terminal_skill)

        # Get all available tools (for sending to LLM)
        tools = manager.get_all_tool_specs()

        # Execute a tool (routes to correct skill)
        result = await manager.execute_tool("read_file", {"path": "x.py"})
    """

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel
        self._skills: dict[str, Skill] = {}  # name → skill
        self._tool_to_skill: dict[str, str] = {}  # tool_name → skill_name
        self._activated: set[str] = set()  # skill names that have been activated
        self._initialized: set[str] = set()  # skill names that have been initialized

    async def register(
        self,
        skill: Skill,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a skill.

        Calls skill.initialize() but NOT activate() (lazy activation).
        """
        manifest = skill.manifest()
        name = manifest.name

        if name in self._skills:
            logger.warning(f"Skill '{name}' already registered, replacing")

        self._skills[name] = skill

        # Map tools to this skill
        for tool_spec in manifest.tools:
            if tool_spec.name in self._tool_to_skill:
                other = self._tool_to_skill[tool_spec.name]
                logger.warning(
                    f"Tool '{tool_spec.name}' already registered by '{other}', "
                    f"now owned by '{name}'"
                )
            self._tool_to_skill[tool_spec.name] = name

        # Initialize
        await skill.initialize(self._kernel, config or {})
        self._initialized.add(name)

        logger.debug(f"Registered skill '{name}' with {len(manifest.tools)} tools")

    # Alias for backward compatibility
    register_async = register

    async def _ensure_activated(self, skill_name: str) -> Skill:
        """Ensure a skill is activated, activating if necessary."""
        skill = self._skills.get(skill_name)
        if not skill:
            raise SkillError(f"Skill '{skill_name}' not found", skill_name=skill_name)

        if skill_name not in self._activated:
            logger.debug(f"Activating skill '{skill_name}'")
            await skill.activate()
            self._activated.add(skill_name)

        return skill

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool by name.

        Finds the skill that owns the tool, activates it if needed,
        then calls execute_tool on the skill.
        """
        skill_name = self._tool_to_skill.get(tool_name)
        if not skill_name:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {list(self._tool_to_skill.keys())}",
            )

        try:
            skill = await self._ensure_activated(skill_name)
            return await skill.execute_tool(tool_name, arguments)
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Tool execution failed: {e}",
            )

    def get_all_tool_specs(self) -> list[ToolSpec]:
        """Get all tool specifications from all registered skills."""
        specs = []
        for skill in self._skills.values():
            specs.extend(skill.manifest().tools)
        return specs

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_tool_skill(self, tool_name: str) -> str | None:
        """Get the skill name that owns a tool."""
        return self._tool_to_skill.get(tool_name)

    async def shutdown_all(self) -> None:
        """Shutdown all activated skills."""
        for name in list(self._activated):
            skill = self._skills.get(name)
            if skill:
                try:
                    await skill.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down skill '{name}': {e}")
        self._activated.clear()

    @property
    def skill_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tool_to_skill.keys())