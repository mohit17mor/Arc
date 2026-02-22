"""
Shell Provider interface — the contract for running commands.

The agent never calls subprocess directly. It goes through
this interface, which handles OS-specific differences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

from arc.core.types import ShellOutput, ShellSession


class ShellProvider(ABC):
    """
    Abstract base class for shell providers.

    A shell provider manages persistent shell sessions.
    State (cwd, env vars) persists between commands within a session.

    Implementations:
        PowerShellProvider — Windows (default on Windows)
        BashProvider — Unix (default on Linux/macOS)
    """

    @abstractmethod
    async def create_session(
        self,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellSession:
        """
        Create a new persistent shell session.

        Args:
            cwd: Starting directory (None = current directory)
            env: Environment variables (None = inherit current)

        Returns:
            ShellSession with unique ID
        """
        ...

    @abstractmethod
    async def execute(
        self,
        session: ShellSession,
        command: str,
        timeout: int | None = None,
    ) -> ShellOutput:
        """
        Execute a command in a session.

        State changes (cd, env vars) persist for next command.
        Streams output internally, returns complete result.

        Args:
            session: The session to run in
            command: Command string to execute
            timeout: Seconds before killing (None = no timeout)

        Returns:
            ShellOutput with stdout, stderr, exit_code
        """
        ...

    @abstractmethod
    async def kill_session(self, session: ShellSession) -> None:
        """Forcefully terminate a session and all child processes."""
        ...

    @abstractmethod
    def get_info(self) -> dict[str, str]:
        """
        Return shell environment info.

        Returns dict with: os, shell, version, home
        Used to tell the LLM what environment it's working in.
        """
        ...