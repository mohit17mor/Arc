"""
Bash shell provider — for Linux and macOS.

Similar to PowerShellProvider but uses /bin/bash.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import uuid
import time
from pathlib import Path
from typing import Any

from arc.core.errors import ShellError
from arc.core.types import ShellOutput, ShellSession
from arc.shell.base import ShellProvider

logger = logging.getLogger(__name__)


class BashProvider(ShellProvider):
    """
    Shell provider using Bash.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._bash_executable = self._find_bash()

    async def create_session(
        self,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellSession:
        """Create a new Bash session."""
        session_id = uuid.uuid4().hex[:12]
        working_dir = str((cwd or Path.cwd()).resolve())

        session_env = os.environ.copy()
        if env:
            session_env.update(env)

        self._sessions[session_id] = {
            "cwd": working_dir,
            "env": session_env,
        }

        logger.debug(f"Created Bash session {session_id} at {working_dir}")

        return ShellSession(
            id=session_id,
            provider="bash",
            cwd=working_dir,
            env=env or {},
            is_alive=True,
        )

    async def execute(
        self,
        session: ShellSession,
        command: str,
        timeout: int | None = None,
    ) -> ShellOutput:
        """Execute a command in a Bash session."""
        session_data = self._sessions.get(session.id)
        if session_data is None:
            session.is_alive = False
            raise ShellError(f"Session {session.id} not found")

        cwd = session_data["cwd"]
        env = session_data["env"]
        timeout = timeout or 120

        # Build the bash script
        bash_script = f'''
cd "{cwd}" 2>/dev/null || true
{command}
echo "___EXIT_CODE___:$?"
echo "___NEW_CWD___:$(pwd)"
'''

        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                self._bash_executable,
                "-c",
                bash_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                elapsed = int((time.time() - start_time) * 1000)
                return ShellOutput(
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    exit_code=-1,
                    duration_ms=elapsed,
                    timed_out=True,
                )

        except FileNotFoundError:
            raise ShellError(
                f"Bash not found at '{self._bash_executable}'. "
                f"Please install bash or configure a different shell."
            )
        except Exception as e:
            raise ShellError(f"Failed to execute command: {e}")

        elapsed = int((time.time() - start_time) * 1000)

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Parse exit code and new cwd from output
        exit_code = 0
        new_cwd = cwd
        clean_lines = []

        for line in stdout.split("\n"):
            stripped = line.strip()
            if stripped.startswith("___EXIT_CODE___:"):
                try:
                    exit_code = int(stripped.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif stripped.startswith("___NEW_CWD___:"):
                new_cwd = stripped.split(":", 1)[1].strip()
            else:
                clean_lines.append(line)

        # Update session cwd
        session_data["cwd"] = new_cwd
        session.cwd = new_cwd

        clean_stdout = "\n".join(clean_lines).strip()

        return ShellOutput(
            stdout=clean_stdout,
            stderr=stderr.strip(),
            exit_code=exit_code,
            duration_ms=elapsed,
            timed_out=False,
        )

    async def kill_session(self, session: ShellSession) -> None:
        """Kill a Bash session."""
        self._sessions.pop(session.id, None)
        session.is_alive = False

    def get_info(self) -> dict[str, str]:
        return {
            "os": platform.system().lower(),
            "shell": "bash",
            "executable": self._bash_executable,
            "home": str(Path.home()),
            "platform": platform.platform(),
        }

    async def close(self) -> None:
        """Close all sessions."""
        self._sessions.clear()

    @staticmethod
    def _find_bash() -> str:
        """Find bash executable."""
        if shutil.which("bash"):
            return "bash"
        return "/bin/bash"