from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InstallPaths:
    arc_home: Path
    runtime_root: Path
    current_runtime: Path
    launcher_root: Path
    install_metadata: Path
    health_report: Path
    managed_launcher: Path


@dataclass(slots=True)
class RuntimeEnvironment:
    runtime_root: Path
    venv_root: Path
    venv_python: Path
    venv_pip: Path


@dataclass(slots=True)
class InstallSummary:
    required_successes: list[str] = field(default_factory=list)
    required_failures: list[str] = field(default_factory=list)
    optional_successes: list[str] = field(default_factory=list)
    optional_failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_blocking_failures(self) -> bool:
        return bool(self.required_failures)

    @property
    def can_continue(self) -> bool:
        return not self.has_blocking_failures


@dataclass(slots=True)
class HealthReport:
    ok: bool
    blocking_issues: list[str] = field(default_factory=list)
    optional_issues: list[str] = field(default_factory=list)
    checked_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class InstallerManifest:
    version: str
    wheel_url: str
    entry_module: str = "arc.install.bootstrap"
    checksums: dict[str, str] = field(default_factory=dict)

