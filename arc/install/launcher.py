from __future__ import annotations

import os
from pathlib import Path


def render_posix_launcher(python_path: str) -> str:
    return "\n".join([
        "#!/bin/sh",
        f'exec "{python_path}" -m arc.cli.main "$@"',
        "",
    ])


def render_windows_launcher(python_path: str) -> str:
    return "\r\n".join([
        "@echo off",
        f'"{python_path}" -m arc.cli.main %*',
        "",
    ])


def write_launcher(target: Path, python_path: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" or target.suffix.lower() == ".cmd":
        target.write_text(render_windows_launcher(str(python_path)), encoding="utf-8")
    else:
        target.write_text(render_posix_launcher(str(python_path)), encoding="utf-8")
        target.chmod(target.stat().st_mode | 0o755)
    return target

