from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from arc.install.launcher import render_posix_launcher, render_windows_launcher, write_launcher


def test_posix_launcher_points_to_managed_runtime_python():
    script = render_posix_launcher("/home/user/.arc/runtime/current/.venv/bin/python")

    assert "arc.cli.main" in script
    assert "/home/user/.arc/runtime/current/.venv/bin/python" in script


def test_windows_launcher_points_to_managed_runtime_python():
    script = render_windows_launcher(r"C:\\Users\\me\\.arc\\runtime\\current\\.venv\\Scripts\\python.exe")

    assert "arc.cli.main" in script
    assert r"C:\\Users\\me\\.arc\\runtime\\current\\.venv\\Scripts\\python.exe" in script


def test_write_launcher_creates_posix_script_and_marks_executable(tmp_path):
    target = tmp_path / "bin" / "arc"
    python_path = Path("/tmp/runtime/.venv/bin/python")

    written = write_launcher(target, python_path)

    assert written == target
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert 'exec "/tmp/runtime/.venv/bin/python" -m arc.cli.main "$@"' in text
    assert target.stat().st_mode & 0o755


def test_write_launcher_uses_windows_format_for_cmd_suffix(tmp_path):
    target = tmp_path / "bin" / "arc.cmd"
    python_path = Path(r"C:\runtime\.venv\Scripts\python.exe")

    written = write_launcher(target, python_path)

    assert written == target
    text = target.read_text(encoding="utf-8")
    assert "@echo off" in text
    assert '"C:\\runtime\\.venv\\Scripts\\python.exe" -m arc.cli.main %*' in text
