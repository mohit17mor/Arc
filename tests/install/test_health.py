from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from arc.install.health import evaluate_install_health
from arc.install.health import write_health_report


def test_health_reports_missing_launcher_as_blocking(tmp_path):
    report = evaluate_install_health(tmp_path / ".arc")

    assert report.ok is False
    assert "launcher" in " ".join(report.blocking_issues).lower()


def test_health_reports_optional_issue_for_unreadable_report(tmp_path):
    arc_home = tmp_path / ".arc"
    managed_launcher = arc_home / "bin" / "arc"
    managed_launcher.parent.mkdir(parents=True, exist_ok=True)
    managed_launcher.write_text("launcher", encoding="utf-8")
    runtime_python = arc_home / "runtime" / "current" / ".venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("python", encoding="utf-8")
    health_report = arc_home / "runtime" / "current" / "health-report.json"
    health_report.write_text("{not-json", encoding="utf-8")

    report = evaluate_install_health(arc_home)

    assert report.ok is True
    assert "could not be read" in " ".join(report.optional_issues).lower()


def test_health_reports_browser_not_ready_as_optional_issue(tmp_path):
    arc_home = tmp_path / ".arc"
    managed_launcher = arc_home / "bin" / "arc"
    managed_launcher.parent.mkdir(parents=True, exist_ok=True)
    managed_launcher.write_text("launcher", encoding="utf-8")
    runtime_python = arc_home / "runtime" / "current" / ".venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("python", encoding="utf-8")
    health_report = arc_home / "runtime" / "current" / "health-report.json"
    health_report.write_text(json.dumps({"browser_ready": False}), encoding="utf-8")

    report = evaluate_install_health(arc_home)

    assert report.ok is True
    assert "browser support is not ready" in " ".join(report.optional_issues).lower()


def test_write_health_report_persists_payload_and_checked_paths(tmp_path):
    arc_home = tmp_path / ".arc"
    report = write_health_report(arc_home, browser_ready=True)
    payload = json.loads((arc_home / "runtime" / "current" / "health-report.json").read_text(encoding="utf-8"))

    assert payload["ok"] == report.ok
    assert payload["browser_ready"] is True
    assert payload["checked_paths"]["arc_home"] == str(arc_home)


def test_venv_python_uses_windows_layout_when_os_is_nt(tmp_path):
    from arc.install import health as health_module

    with patch.object(health_module.os, "name", "nt"):
        path = health_module._venv_python(tmp_path / "runtime" / "current")

    assert path == tmp_path / "runtime" / "current" / ".venv" / "Scripts" / "python.exe"
