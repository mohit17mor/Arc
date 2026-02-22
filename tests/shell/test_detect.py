"""Tests for shell auto-detection."""

import platform
import pytest
from arc.shell.detect import detect_shell


def test_auto_detect():
    """Auto-detect returns a valid provider for current OS."""
    provider = detect_shell("auto")
    info = provider.get_info()

    system = platform.system().lower()
    if system == "windows":
        assert info["shell"] == "powershell"
    else:
        assert info["shell"] == "bash"


def test_explicit_powershell():
    """Can explicitly request PowerShell."""
    if platform.system().lower() != "windows":
        pytest.skip("PowerShell test only runs on Windows")

    provider = detect_shell("powershell")
    assert provider.get_info()["shell"] == "powershell"


def test_explicit_bash():
    """Can explicitly request Bash."""
    if platform.system().lower() == "windows":
        pytest.skip("Bash test only runs on Unix")

    provider = detect_shell("bash")
    assert provider.get_info()["shell"] == "bash"


def test_invalid_shell():
    """Invalid shell name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown shell"):
        detect_shell("nonexistent_shell")