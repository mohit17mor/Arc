"""Tests for the terminal skill."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arc.core.types import ShellOutput
from arc.skills.builtin.terminal import TerminalSkill


class TestTerminalSkill:
    def test_manifest_exposes_execute_tool(self):
        skill = TerminalSkill()

        manifest = skill.manifest()

        assert manifest.name == "terminal"
        assert manifest.always_available is True
        assert manifest.tools[0].name == "execute"
        assert "command" in manifest.tools[0].parameters["properties"]

    @pytest.mark.asyncio
    async def test_activate_detects_shell_and_creates_session(self):
        skill = TerminalSkill()
        shell = AsyncMock()
        shell.create_session.return_value = "session-1"
        await skill.initialize(kernel="kernel", config={"shell": "bash"})

        with patch("arc.skills.builtin.terminal.detect_shell", return_value=shell) as detect:
            await skill.activate()

        detect.assert_called_once_with(preference="bash")
        shell.create_session.assert_awaited_once()
        assert skill._shell is shell
        assert skill._session == "session-1"

    @pytest.mark.asyncio
    async def test_execute_tool_rejects_unknown_tool(self):
        skill = TerminalSkill()
        result = await skill.execute_tool("unknown", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_requires_command(self):
        skill = TerminalSkill()
        result = await skill.execute_tool("execute", {})
        assert result.success is False
        assert result.error == "No command provided"

    @pytest.mark.asyncio
    async def test_execute_tool_requires_activation(self):
        skill = TerminalSkill()
        result = await skill.execute_tool("execute", {"command": "echo hi"})
        assert result.success is False
        assert result.error == "Terminal not activated"

    @pytest.mark.asyncio
    async def test_execute_tool_reports_timeout(self):
        skill = TerminalSkill()
        shell = AsyncMock()
        shell.execute.return_value = ShellOutput(
            stdout="partial output",
            stderr="",
            exit_code=-1,
            duration_ms=10,
            timed_out=True,
        )
        skill._shell = shell
        skill._session = "session-1"
        skill._config = {"timeout": 9}

        result = await skill.execute_tool("execute", {"command": "sleep 10"})

        assert result.success is False
        assert result.output == "partial output"
        assert result.error == "Command timed out after 9s"

    @pytest.mark.asyncio
    async def test_execute_tool_includes_stderr_and_exit_code(self):
        skill = TerminalSkill()
        shell = AsyncMock()
        shell.execute.return_value = ShellOutput(
            stdout="stdout text",
            stderr="stderr text",
            exit_code=2,
            duration_ms=10,
            timed_out=False,
        )
        skill._shell = shell
        skill._session = "session-1"

        result = await skill.execute_tool("execute", {"command": "bad-cmd"})

        assert result.success is False
        assert "stdout text" in result.output
        assert "[stderr]: stderr text" in result.output
        assert result.error == "Exit code: 2"

    @pytest.mark.asyncio
    async def test_execute_tool_returns_successful_output(self):
        skill = TerminalSkill()
        shell = AsyncMock()
        shell.execute.return_value = ShellOutput(
            stdout="ok",
            stderr="",
            exit_code=0,
            duration_ms=5,
            timed_out=False,
        )
        skill._shell = shell
        skill._session = "session-1"

        result = await skill.execute_tool("execute", {"command": "echo ok"})

        assert result.success is True
        assert result.output == "ok"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_tool_surfaces_shell_exceptions(self):
        skill = TerminalSkill()
        shell = AsyncMock()
        shell.execute.side_effect = RuntimeError("boom")
        skill._shell = shell
        skill._session = "session-1"

        result = await skill.execute_tool("execute", {"command": "echo ok"})

        assert result.success is False
        assert result.error == "boom"

    @pytest.mark.asyncio
    async def test_shutdown_kills_session_when_active(self):
        skill = TerminalSkill()
        shell = AsyncMock()
        skill._shell = shell
        skill._session = "session-1"

        await skill.shutdown()

        shell.kill_session.assert_awaited_once_with("session-1")
        assert skill._session is None
