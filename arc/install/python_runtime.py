from __future__ import annotations

import os
import venv
from pathlib import Path

from arc.install.types import RuntimeEnvironment


def ensure_runtime(runtime_root: Path) -> RuntimeEnvironment:
    """Create the managed runtime venv if needed and return its paths."""
    venv_root = runtime_root / ".venv"
    venv_python = _venv_python_path(venv_root)
    if not venv_python.exists():
        runtime_root.mkdir(parents=True, exist_ok=True)
        builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade=False)
        builder.create(venv_root)
    return RuntimeEnvironment(
        runtime_root=runtime_root,
        venv_root=venv_root,
        venv_python=_venv_python_path(venv_root),
        venv_pip=_venv_pip_path(venv_root),
    )


def _venv_python_path(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _venv_pip_path(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "pip.exe"
    return venv_root / "bin" / "pip"

