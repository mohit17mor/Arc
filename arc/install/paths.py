from __future__ import annotations

from pathlib import Path

from arc.install.types import InstallPaths


def build_install_paths(*, home: Path | None = None) -> InstallPaths:
    """Build the managed install/runtime layout under the user's Arc home."""
    home_root = home if home is not None else Path.home()
    arc_home = home_root / ".arc"
    runtime_root = arc_home / "runtime"
    current_runtime = runtime_root / "current"
    launcher_root = arc_home / "bin"
    install_metadata = current_runtime / "install-meta.json"
    health_report = current_runtime / "health-report.json"
    managed_launcher = launcher_root / ("arc.cmd" if _is_windows() else "arc")
    return InstallPaths(
        arc_home=arc_home,
        runtime_root=runtime_root,
        current_runtime=current_runtime,
        launcher_root=launcher_root,
        install_metadata=install_metadata,
        health_report=health_report,
        managed_launcher=managed_launcher,
    )


def _is_windows() -> bool:
    import os

    return os.name == "nt"

