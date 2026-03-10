"""
Skill Manager — registration, lifecycle, and tool routing.

The manager is the single point of access for all skills.
It handles lazy activation and routes tool calls to the right skill.

Large tool results (> OUTPUT_SPILLOVER_THRESHOLD chars) are automatically
saved to disk and replaced with a summary + file path.  The LLM can
then use read_file to access specific parts if needed.  This prevents
a single tool call from blowing up the context window.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path
from typing import Any

from arc.core.errors import SkillError
from arc.core.types import SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# Minimum description length before a warning is emitted at registration time.
_MIN_DESC_LEN = 15

# Tool output larger than this (in characters) is saved to a file and
# replaced with a summary + path.  ~8K chars ≈ ~2K tokens.
OUTPUT_SPILLOVER_THRESHOLD = 8_000

# How many characters of the beginning to keep inline as a preview.
OUTPUT_PREVIEW_SIZE = 2_000

# Directory where large tool outputs are saved.
_ARTIFACTS_DIR = Path.home() / ".arc" / "artifacts" / "tool_results"

# Write buffer size — 1MB reduces syscalls for large files.
_WRITE_BUFFER = 1024 * 1024


def _write_bytes_buffered(filepath: Path, data: bytes) -> None:
    """Write bytes to a file with a large buffer. Runs in executor."""
    with open(filepath, "wb", buffering=_WRITE_BUFFER) as f:
        f.write(data)


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
        self._manifests: dict[str, SkillManifest] = {}  # name → cached manifest
        self._tool_to_skill: dict[str, str] = {}  # tool_name → skill_name
        self._tool_specs: dict[str, ToolSpec] = {}  # tool_name → ToolSpec (O(1))
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
        Caches the manifest and individual tool specs for O(1) lookup.
        """
        manifest = skill.manifest()
        name = manifest.name

        if name in self._skills:
            logger.warning(f"Skill '{name}' already registered, replacing")

        self._skills[name] = skill
        self._manifests[name] = manifest

        # Warn on short/missing skill description
        if len(manifest.description) < _MIN_DESC_LEN:
            logger.warning(
                f"Skill '{name}' has a very short description "
                f"({len(manifest.description)} chars) — "
                f"the LLM may not understand when to use it"
            )

        # Map tools to this skill
        for tool_spec in manifest.tools:
            if tool_spec.name in self._tool_to_skill:
                other = self._tool_to_skill[tool_spec.name]
                logger.warning(
                    f"Tool '{tool_spec.name}' already registered by '{other}', "
                    f"now owned by '{name}'"
                )
            self._tool_to_skill[tool_spec.name] = name
            self._tool_specs[tool_spec.name] = tool_spec

            # Warn on short tool descriptions
            if len(tool_spec.description) < _MIN_DESC_LEN:
                logger.warning(
                    f"Tool '{tool_spec.name}' (skill '{name}') has a very "
                    f"short description — LLM tool selection may suffer"
                )

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
            result = await skill.execute_tool(tool_name, arguments)
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Tool execution failed: {e}",
            )

        # ── Large output spillover ───────────────────────────────────
        # If the tool returned a lot of data, save it to a file and
        # replace the output with a summary + path.  The LLM can use
        # read_file to access specific parts if it needs more detail.
        if result.output and len(result.output) > OUTPUT_SPILLOVER_THRESHOLD:
            result = await self._spillover_to_file(result, tool_name)

        return result

    @staticmethod
    async def _spillover_to_file(result: ToolResult, tool_name: str) -> ToolResult:
        """
        Save large tool output to a file and replace with summary + path.

        The full output is preserved on disk.  The LLM sees a short preview
        + the file path, and can use read_file if it needs more detail.

        Uses buffered binary write for speed — important for 30-50MB outputs.
        """
        try:
            _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            safe_name = tool_name.replace("/", "_").replace("\\", "_")
            filename = f"{safe_name}_{timestamp}.txt"
            filepath = _ARTIFACTS_DIR / filename

            # Encode once, write as bytes with 1MB buffer for speed.
            # For 30-50MB outputs this is ~5x faster than write_text.
            data = result.output.encode("utf-8")
            await asyncio.get_running_loop().run_in_executor(
                None, _write_bytes_buffered, filepath, data
            )

            total_chars = len(result.output)
            total_lines = result.output.count("\n") + 1
            preview = result.output[:OUTPUT_PREVIEW_SIZE]

            summary = (
                f"[Output too large for context — saved to file]\n"
                f"File: {filepath}\n"
                f"Size: {total_chars:,} characters, {total_lines:,} lines\n\n"
                f"--- Preview (first {OUTPUT_PREVIEW_SIZE} chars) ---\n"
                f"{preview}\n"
                f"--- End of preview ---\n\n"
                f"Use read_file to access specific sections if needed."
            )

            logger.info(
                f"Tool '{tool_name}' output spilled to {filepath} "
                f"({total_chars:,} chars)"
            )

            return ToolResult(
                tool_call_id=result.tool_call_id,
                success=result.success,
                output=summary,
                error=result.error,
                artifacts=[str(filepath)] + result.artifacts,
                duration_ms=result.duration_ms,
            )

        except Exception as e:
            # If file write fails, fall back to truncation
            logger.warning(f"Spillover file write failed: {e}")
            truncated = result.output[:OUTPUT_SPILLOVER_THRESHOLD]
            return ToolResult(
                tool_call_id=result.tool_call_id,
                success=result.success,
                output=truncated + "\n\n[... output truncated — file save failed]",
                error=result.error,
                artifacts=result.artifacts,
                duration_ms=result.duration_ms,
            )

    def get_all_tool_specs(self) -> list[ToolSpec]:
        """Get all tool specifications from all registered skills."""
        specs = []
        for manifest in self._manifests.values():
            specs.extend(manifest.tools)
        return specs

    def get_tool_spec(self, tool_name: str) -> ToolSpec | None:
        """Get a single tool spec by name — O(1) lookup."""
        return self._tool_specs.get(tool_name)

    def get_manifest(self, skill_name: str) -> SkillManifest | None:
        """Get the cached manifest for a skill."""
        return self._manifests.get(skill_name)

    @property
    def manifests(self) -> dict[str, SkillManifest]:
        """All cached manifests keyed by skill name."""
        return dict(self._manifests)

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