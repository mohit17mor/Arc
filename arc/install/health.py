from __future__ import annotations

import json
import os
from pathlib import Path

from arc.install.paths import build_install_paths
from arc.install.types import HealthReport


def evaluate_install_health(arc_home: Path | None = None) -> HealthReport:
    home = arc_home.parent if arc_home is not None and arc_home.name == ".arc" else arc_home
    paths = build_install_paths(home=home)
    blocking_issues: list[str] = []
    optional_issues: list[str] = []

    if not paths.managed_launcher.exists():
        blocking_issues.append("Managed launcher is missing.")
    venv_python = _venv_python(paths.current_runtime)
    if not venv_python.exists():
        blocking_issues.append("Managed runtime Python is missing.")

    if paths.health_report.exists():
        try:
            payload = json.loads(paths.health_report.read_text(encoding="utf-8"))
        except Exception:
            optional_issues.append("Stored health report could not be read.")
        else:
            browser_ready = payload.get("browser_ready")
            if browser_ready is False:
                optional_issues.append("Browser support is not ready.")

    return HealthReport(
        ok=not blocking_issues,
        blocking_issues=blocking_issues,
        optional_issues=optional_issues,
        checked_paths={
            "arc_home": str(paths.arc_home),
            "managed_launcher": str(paths.managed_launcher),
            "runtime_python": str(venv_python),
            "health_report": str(paths.health_report),
        },
    )


def write_health_report(arc_home: Path | None = None, *, browser_ready: bool | None = None) -> HealthReport:
    home = arc_home.parent if arc_home is not None and arc_home.name == ".arc" else arc_home
    paths = build_install_paths(home=home)
    report = evaluate_install_health(paths.arc_home)
    paths.health_report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": report.ok,
        "blocking_issues": report.blocking_issues,
        "optional_issues": report.optional_issues,
        "checked_paths": report.checked_paths,
    }
    if browser_ready is not None:
        payload["browser_ready"] = browser_ready
    paths.health_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report


def _venv_python(runtime_root: Path) -> Path:
    if os.name == "nt":
        return runtime_root / ".venv" / "Scripts" / "python.exe"
    return runtime_root / ".venv" / "bin" / "python"

