"""
Shell auto-detection — picks the best shell for the current OS.
"""

from __future__ import annotations

import platform
import logging

from arc.shell.base import ShellProvider

logger = logging.getLogger(__name__)


def detect_shell(preference: str = "auto") -> ShellProvider:
    """
    Detect and return the best shell provider for this system.

    Args:
        preference: "auto", "powershell", "bash", "wsl"
                    "auto" picks the best for the current OS.

    Returns:
        An instantiated ShellProvider
    """
    system = platform.system().lower()

    if preference != "auto":
        return _create_by_name(preference)

    if system == "windows":
        logger.info("Detected Windows — using PowerShell")
        from arc.shell.powershell import PowerShellProvider

        return PowerShellProvider()
    else:
        logger.info(f"Detected {system} — using Bash")
        from arc.shell.bash import BashProvider

        return BashProvider()


def _create_by_name(name: str) -> ShellProvider:
    """Create a shell provider by name."""
    name = name.lower().strip()

    if name in ("powershell", "pwsh"):
        from arc.shell.powershell import PowerShellProvider

        return PowerShellProvider()
    elif name in ("bash", "sh"):
        from arc.shell.bash import BashProvider

        return BashProvider()
    else:
        raise ValueError(
            f"Unknown shell provider: '{name}'. "
            f"Available: powershell, bash"
        )


def get_shell_info() -> dict[str, str]:
    """Get info about the current system's shell environment."""
    provider = detect_shell()
    return provider.get_info()