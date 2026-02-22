"""
Terminal Skill — execute shell commands.

Wraps the ShellProvider to provide tool interface.
"""

from __future__ import annotations

from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill
from arc.shell.base import ShellProvider
from arc.shell.detect import detect_shell


class TerminalSkill(Skill):
    """
    Skill for executing shell commands.

    Tools:
        execute(command) → command output
    """

    def __init__(self) -> None:
        self._kernel: Any = None
        self._config: dict[str, Any] = {}
        self._shell: ShellProvider | None = None
        self._session: Any = None

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="terminal",
            version="1.0.0",
            description="Execute shell commands",
            capabilities=frozenset([Capability.SHELL_EXEC]),
            tools=(
                ToolSpec(
                    name="execute",
                    description="Execute a shell command and return its output",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            },
                        },
                        "required": ["command"],
                    },
                    required_capabilities=frozenset([Capability.SHELL_EXEC]),
                ),
            ),
        )

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config

    async def activate(self) -> None:
        """Create shell provider and session on first use."""
        self._shell = detect_shell(
            preference=self._config.get("shell", "auto")
        )
        self._session = await self._shell.create_session()

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name != "execute":
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        command = arguments.get("command", "")
        if not command:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="No command provided",
            )

        if self._shell is None or self._session is None:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="Terminal not activated",
            )

        try:
            timeout = self._config.get("timeout", 120)
            result = await self._shell.execute(
                self._session,
                command,
                timeout=timeout,
            )

            if result.timed_out:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    output=result.stdout,
                    error=f"Command timed out after {timeout}s",
                )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            return ToolResult(
                tool_call_id="",
                success=result.exit_code == 0,
                output=output,
                error=None if result.exit_code == 0 else f"Exit code: {result.exit_code}",
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=str(e),
            )

    async def shutdown(self) -> None:
        """Close the shell session."""
        if self._shell and self._session:
            await self._shell.kill_session(self._session)
            self._session = None