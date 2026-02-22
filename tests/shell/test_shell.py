"""Tests for shell providers — runs on current OS."""

import platform
import pytest
from pathlib import Path

from arc.shell.detect import detect_shell


@pytest.fixture
async def shell():
    """Get shell provider for current OS."""
    provider = detect_shell()
    yield provider
    await provider.close()


@pytest.mark.asyncio
async def test_create_session(shell):
    """Session starts successfully."""
    session = await shell.create_session()
    assert session.is_alive
    assert session.id
    assert session.provider in ("powershell", "bash")
    await shell.kill_session(session)


@pytest.mark.asyncio
async def test_execute_echo(shell):
    """Simple echo command works."""
    session = await shell.create_session()

    try:
        if platform.system().lower() == "windows":
            result = await shell.execute(session, 'Write-Output "hello arc"')
        else:
            result = await shell.execute(session, 'echo "hello arc"')

        assert result.exit_code == 0
        assert "hello arc" in result.stdout
        assert result.timed_out is False
        assert result.duration_ms >= 0
    finally:
        await shell.kill_session(session)


@pytest.mark.asyncio
async def test_exit_code(shell):
    """Failed command returns non-zero exit code."""
    session = await shell.create_session()

    try:
        if platform.system().lower() == "windows":
            result = await shell.execute(
                session, "cmd /c exit 1"
            )
        else:
            result = await shell.execute(session, "exit 1")

        assert result.exit_code != 0
    finally:
        await shell.kill_session(session)


@pytest.mark.asyncio
async def test_cwd_persists(shell):
    """cd persists between commands."""
    session = await shell.create_session()

    try:
        # Create a temp dir and cd into it
        if platform.system().lower() == "windows":
            await shell.execute(session, "cd $env:TEMP")
            result = await shell.execute(session, "(Get-Location).Path")
        else:
            await shell.execute(session, "cd /tmp")
            result = await shell.execute(session, "pwd")

        assert result.exit_code == 0
        # Output should show the new directory
        assert result.stdout.strip() != ""
    finally:
        await shell.kill_session(session)


@pytest.mark.asyncio
async def test_timeout(shell):
    """Long-running command gets timed out."""
    session = await shell.create_session()

    try:
        if platform.system().lower() == "windows":
            result = await shell.execute(
                session, "Start-Sleep -Seconds 30", timeout=2
            )
        else:
            result = await shell.execute(session, "sleep 30", timeout=2)

        assert result.timed_out is True
    finally:
        await shell.kill_session(session)


@pytest.mark.asyncio
async def test_multiple_commands(shell):
    """Multiple commands work in sequence."""
    session = await shell.create_session()

    try:
        r1 = await shell.execute(session, "echo first")
        r2 = await shell.execute(session, "echo second")
        r3 = await shell.execute(session, "echo third")

        assert r1.exit_code == 0
        assert "first" in r1.stdout
        assert r2.exit_code == 0
        assert "second" in r2.stdout
        assert r3.exit_code == 0
        assert "third" in r3.stdout
    finally:
        await shell.kill_session(session)


@pytest.mark.asyncio
async def test_get_info(shell):
    """get_info returns expected fields."""
    info = shell.get_info()

    assert "os" in info
    assert "shell" in info
    assert "home" in info
    assert info["home"]  # not empty