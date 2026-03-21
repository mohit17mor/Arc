from __future__ import annotations

from pathlib import Path

from arc.install.bootstrap import build_wheel_install_command, record_browser_setup_failure
from arc.install.python_runtime import ensure_runtime
from arc.install.types import InstallSummary


def test_install_summary_distinguishes_required_and_optional_failures():
    summary = InstallSummary(
        required_failures=["python"],
        optional_failures=["playwright"],
    )

    assert summary.has_blocking_failures is True
    assert summary.can_continue is False


def test_bootstrap_creates_runtime_virtualenv(tmp_path):
    result = ensure_runtime(tmp_path / ".arc" / "runtime" / "current")

    assert result.runtime_root == tmp_path / ".arc" / "runtime" / "current"
    assert result.venv_python.exists()


def test_bootstrap_builds_pip_install_command_from_manifest():
    cmd = build_wheel_install_command("python", "https://example.test/arc.whl")

    assert "pip" in cmd
    assert "arc.whl" in " ".join(cmd)


def test_browser_setup_failure_is_marked_optional():
    summary = record_browser_setup_failure(InstallSummary(), "playwright failed")

    assert summary.can_continue is True
    assert "playwright failed" in summary.optional_failures[0]


def test_posix_bootstrap_invokes_shared_installer():
    script = Path("scripts/install/install.sh").read_text(encoding="utf-8")

    assert "arc.install.bootstrap" in script


def test_windows_bootstrap_invokes_shared_installer():
    script = Path("scripts/install/install.ps1").read_text(encoding="utf-8")

    assert "arc.install.bootstrap" in script
