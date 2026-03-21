from __future__ import annotations

from arc.install.launcher import render_posix_launcher, render_windows_launcher


def test_posix_launcher_points_to_managed_runtime_python():
    script = render_posix_launcher("/home/user/.arc/runtime/current/.venv/bin/python")

    assert "arc.cli.main" in script
    assert "/home/user/.arc/runtime/current/.venv/bin/python" in script


def test_windows_launcher_points_to_managed_runtime_python():
    script = render_windows_launcher(r"C:\\Users\\me\\.arc\\runtime\\current\\.venv\\Scripts\\python.exe")

    assert "arc.cli.main" in script
    assert r"C:\\Users\\me\\.arc\\runtime\\current\\.venv\\Scripts\\python.exe" in script
