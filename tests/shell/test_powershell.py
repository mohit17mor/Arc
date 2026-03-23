"""Tests for the PowerShell shell provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from arc.core.errors import ShellError
from arc.shell.powershell import PowerShellProvider


class FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False
        self.waited = False

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.waited = True


class SlowProcess(FakeProcess):
    async def communicate(self):
        await asyncio.sleep(0.05)
        return self._stdout, self._stderr


class TestFindPowerShell:
    def test_prefers_pwsh(self):
        with patch("arc.shell.powershell.shutil.which", side_effect=lambda name: "/bin/pwsh" if name == "pwsh" else None):
            assert PowerShellProvider._find_powershell() == "pwsh"

    def test_falls_back_to_powershell(self):
        def which(name: str):
            if name == "pwsh":
                return None
            if name == "powershell":
                return "/bin/powershell"
            return None

        with patch("arc.shell.powershell.shutil.which", side_effect=which):
            assert PowerShellProvider._find_powershell() == "powershell"

    def test_uses_powershell_exe_as_last_resort(self):
        with patch("arc.shell.powershell.shutil.which", return_value=None):
            assert PowerShellProvider._find_powershell() == "powershell.exe"


class TestPowerShellProvider:
    @pytest.mark.asyncio
    async def test_create_session_tracks_cwd_and_env(self, tmp_path):
        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()

        session = await provider.create_session(cwd=tmp_path, env={"ARC_MODE": "test"})

        assert session.provider == "powershell"
        assert session.cwd == str(tmp_path.resolve())
        assert session.env == {"ARC_MODE": "test"}
        assert provider._sessions[session.id]["cwd"] == str(tmp_path.resolve())
        assert provider._sessions[session.id]["env"]["ARC_MODE"] == "test"

    @pytest.mark.asyncio
    async def test_execute_raises_when_session_missing(self):
        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        session = await provider.create_session()

        await provider.kill_session(session)

        with pytest.raises(ShellError, match="not found"):
            await provider.execute(session, "Write-Output hello")
        assert session.is_alive is False

    @pytest.mark.asyncio
    async def test_execute_parses_exit_code_and_updates_cwd(self, tmp_path):
        process = FakeProcess(
            stdout=(
                b"hello arc\n"
                b"___EXIT_CODE___:5\n"
                b"___NEW_CWD___:C:/Temp\n"
            ),
            stderr=b"warning line\n",
        )

        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        session = await provider.create_session(cwd=tmp_path)

        with patch("arc.shell.powershell.asyncio.create_subprocess_exec", return_value=process) as create_proc:
            result = await provider.execute(session, 'Write-Output "hello arc"', timeout=3)

        assert result.stdout == "hello arc"
        assert result.stderr == "warning line"
        assert result.exit_code == 5
        assert result.timed_out is False
        assert session.cwd == "C:/Temp"
        assert provider._sessions[session.id]["cwd"] == "C:/Temp"

        called_args = create_proc.call_args[0]
        assert called_args[0] == "pwsh"
        assert called_args[1:5] == ("-NoLogo", "-NoProfile", "-NonInteractive", "-Command")
        assert f"Set-Location -Path '{tmp_path.resolve()}'" in called_args[5]
        assert 'Write-Output "hello arc"' in called_args[5]

    @pytest.mark.asyncio
    async def test_execute_times_out_and_kills_process(self):
        process = SlowProcess()

        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        session = await provider.create_session()

        with patch("arc.shell.powershell.asyncio.create_subprocess_exec", return_value=process):
            result = await provider.execute(session, "Start-Sleep -Seconds 30", timeout=0.001)

        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stderr
        assert process.killed is True
        assert process.waited is True

    @pytest.mark.asyncio
    async def test_execute_wraps_missing_binary_as_shell_error(self):
        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        session = await provider.create_session()

        with patch("arc.shell.powershell.asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(ShellError, match="PowerShell not found"):
                await provider.execute(session, "Write-Output hello")

    @pytest.mark.asyncio
    async def test_kill_session_marks_session_dead(self):
        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        session = await provider.create_session()

        await provider.kill_session(session)

        assert session.is_alive is False
        assert session.id not in provider._sessions

    @pytest.mark.asyncio
    async def test_close_clears_all_sessions(self):
        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        await provider.create_session()
        await provider.create_session()

        await provider.close()

        assert provider._sessions == {}

    def test_get_info_reports_shell_metadata(self):
        with patch.object(PowerShellProvider, "_find_powershell", return_value="pwsh"):
            provider = PowerShellProvider()
        with patch("arc.shell.powershell.platform.platform", return_value="Windows-11"):
            info = provider.get_info()

        assert info["os"] == "windows"
        assert info["shell"] == "powershell"
        assert info["executable"] == "pwsh"
        assert info["platform"] == "Windows-11"
        assert info["home"]
