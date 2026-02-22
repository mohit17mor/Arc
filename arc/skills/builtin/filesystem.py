"""
Filesystem Skill — read, write, list, search files.

This is a minimal implementation with core operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill


class FilesystemSkill(Skill):
    """
    Skill for file system operations.

    Tools:
        read_file(path) → content
        write_file(path, content) → success message
        list_directory(path) → list of entries
    """

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = workspace or Path.cwd()
        self._kernel: Any = None
        self._config: dict[str, Any] = {}

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="filesystem",
            version="1.0.0",
            description="Read, write, and list files",
            capabilities=frozenset(
                [Capability.FILE_READ, Capability.FILE_WRITE]
            ),
            tools=(
                ToolSpec(
                    name="read_file",
                    description="Read the contents of a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read",
                            },
                        },
                        "required": ["path"],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
                ToolSpec(
                    name="write_file",
                    description="Write content to a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write",
                            },
                        },
                        "required": ["path", "content"],
                    },
                    required_capabilities=frozenset([Capability.FILE_WRITE]),
                ),
                ToolSpec(
                    name="list_directory",
                    description="List contents of a directory",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory (default: current)",
                            },
                        },
                        "required": [],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
            ),
        )

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        if "workspace" in config:
            self._workspace = Path(config["workspace"]).resolve()

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            if tool_name == "read_file":
                return await self._read_file(arguments["path"])
            elif tool_name == "write_file":
                return await self._write_file(
                    arguments["path"],
                    arguments["content"],
                )
            elif tool_name == "list_directory":
                return await self._list_directory(arguments.get("path", "."))
            else:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_name}",
                )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=str(e),
            )

    async def _read_file(self, path: str) -> ToolResult:
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        if not file_path.is_file():
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Not a file: {path}",
            )

        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()

        return ToolResult(
            tool_call_id="",
            success=True,
            output=content,
        )

    async def _write_file(self, path: str, content: str) -> ToolResult:
        file_path = self._resolve_path(path)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, mode="w", encoding="utf-8") as f:
            await f.write(content)

        return ToolResult(
            tool_call_id="",
            success=True,
            output=f"Successfully wrote {len(content)} characters to {path}",
        )

    async def _list_directory(self, path: str) -> ToolResult:
        dir_path = self._resolve_path(path)

        if not dir_path.exists():
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Directory not found: {path}",
            )

        if not dir_path.is_dir():
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Not a directory: {path}",
            )

        entries = []
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir():
                entries.append(f"[DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                entries.append(f"[FILE] {entry.name} ({size} bytes)")

        return ToolResult(
            tool_call_id="",
            success=True,
            output="\n".join(entries) if entries else "(empty directory)",
        )

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self._workspace / p).resolve()