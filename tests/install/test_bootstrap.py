from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc.install import bootstrap
from arc.install.bootstrap import build_wheel_install_command, record_browser_setup_failure
from arc.install.paths import build_install_paths
from arc.install.python_runtime import ensure_runtime
from arc.install.types import HealthReport, InstallSummary, InstallerManifest, RuntimeEnvironment


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


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


def test_install_from_manifest_writes_metadata_launcher_and_browser_health(monkeypatch, tmp_path):
    home = tmp_path / "home"
    paths = build_install_paths(home=home)
    runtime = RuntimeEnvironment(
        runtime_root=paths.current_runtime,
        venv_root=paths.current_runtime / ".venv",
        venv_python=paths.current_runtime / ".venv" / "bin" / "python",
        venv_pip=paths.current_runtime / ".venv" / "bin" / "pip",
    )
    commands: list[list[str]] = []
    launcher_calls: list[tuple[Path, Path]] = []
    health_calls: list[tuple[Path, bool | None]] = []

    monkeypatch.setattr(bootstrap, "ensure_runtime", lambda _: runtime)

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "write_launcher",
        lambda launcher, python: launcher_calls.append((launcher, python)),
    )
    monkeypatch.setattr(
        bootstrap,
        "write_health_report",
        lambda arc_home, browser_ready=None: health_calls.append((arc_home, browser_ready)),
    )

    summary = bootstrap.install_from_manifest(
        InstallerManifest(
            version="1.2.3",
            wheel_url="https://example.test/arc.whl",
            entry_module="arc.cli.main",
            checksums={"sha256": "abc123"},
        ),
        home=home,
    )

    assert summary.required_successes == ["runtime_created", "wheel_installed", "launcher_created"]
    assert summary.optional_successes == ["browser_ready"]
    assert summary.required_failures == []
    assert commands == [
        [str(runtime.venv_python), "-m", "pip", "install", "--upgrade", "https://example.test/arc.whl"],
        [str(runtime.venv_python), "-m", "playwright", "install", "chromium"],
    ]
    assert launcher_calls == [(paths.managed_launcher, runtime.venv_python)]
    assert health_calls == [(paths.arc_home, True)]
    assert json.loads(paths.install_metadata.read_text(encoding="utf-8")) == {
        "version": "1.2.3",
        "package_url": "https://example.test/arc.whl",
        "entry_module": "arc.cli.main",
        "checksums": {"sha256": "abc123"},
    }


def test_install_from_manifest_reports_wheel_install_failure(monkeypatch, tmp_path):
    home = tmp_path / "home"
    paths = build_install_paths(home=home)
    runtime = RuntimeEnvironment(
        runtime_root=paths.current_runtime,
        venv_root=paths.current_runtime / ".venv",
        venv_python=paths.current_runtime / ".venv" / "bin" / "python",
        venv_pip=paths.current_runtime / ".venv" / "bin" / "pip",
    )
    health_calls: list[tuple[Path, bool | None]] = []
    launcher_calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(bootstrap, "ensure_runtime", lambda _: runtime)

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="pip failed")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "write_health_report",
        lambda arc_home, browser_ready=None: health_calls.append((arc_home, browser_ready)),
    )
    monkeypatch.setattr(
        bootstrap,
        "write_launcher",
        lambda launcher, python: launcher_calls.append((launcher, python)),
    )

    summary = bootstrap.install_from_manifest(
        {"version": "2.0.0", "wheel_url": "https://example.test/arc.whl"},
        home=home,
    )

    assert summary.required_failures == ["wheel_install_failed: pip failed"]
    assert summary.can_continue is False
    assert summary.optional_successes == []
    assert health_calls == [(paths.arc_home, False)]
    assert launcher_calls == []
    assert not paths.install_metadata.exists()


def test_install_from_manifest_marks_browser_setup_as_optional_failure(monkeypatch, tmp_path):
    home = tmp_path / "home"
    paths = build_install_paths(home=home)
    runtime = RuntimeEnvironment(
        runtime_root=paths.current_runtime,
        venv_root=paths.current_runtime / ".venv",
        venv_python=paths.current_runtime / ".venv" / "bin" / "python",
        venv_pip=paths.current_runtime / ".venv" / "bin" / "pip",
    )
    commands: list[list[str]] = []
    health_calls: list[tuple[Path, bool | None]] = []

    monkeypatch.setattr(bootstrap, "ensure_runtime", lambda _: runtime)

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if len(commands) == 1:
            return SimpleNamespace(stdout="", stderr="")
        raise subprocess.CalledProcessError(1, cmd, stderr="playwright failed")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr(bootstrap, "write_launcher", lambda *_args: None)
    monkeypatch.setattr(
        bootstrap,
        "write_health_report",
        lambda arc_home, browser_ready=None: health_calls.append((arc_home, browser_ready)),
    )

    summary = bootstrap.install_from_manifest(
        InstallerManifest(version="3.0.0", wheel_url="https://example.test/arc.whl"),
        home=home,
    )

    assert summary.required_successes == ["runtime_created", "wheel_installed", "launcher_created"]
    assert summary.optional_failures == ["browser_setup_failed: playwright failed"]
    assert health_calls == [(paths.arc_home, False)]
    assert paths.install_metadata.exists()


def test_load_manifest_reads_remote_payload(monkeypatch):
    payload = {"version": "4.0.0", "wheel_url": "https://example.test/arc.whl"}
    opened: list[str] = []

    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda url: opened.append(url) or _Response(payload),
    )

    manifest = bootstrap.load_manifest("https://example.test/manifest.json")

    assert opened == ["https://example.test/manifest.json"]
    assert manifest == InstallerManifest(version="4.0.0", wheel_url="https://example.test/arc.whl")


def test_render_summary_lists_successes_failures_and_next_step():
    summary = InstallSummary(
        required_successes=["runtime_created"],
        required_failures=["wheel_install_failed"],
        optional_successes=["browser_ready"],
        optional_failures=["browser_setup_failed"],
    )
    report = HealthReport(ok=False)

    rendered = bootstrap._render_summary(summary, report)

    assert "Required successes:" in rendered
    assert "- runtime_created" in rendered
    assert "Required failures:" in rendered
    assert "Optional successes:" in rendered
    assert "Optional failures:" in rendered
    assert "Install health: Needs attention" in rendered
    assert "Run: arc doctor" in rendered


def test_coerce_manifest_returns_existing_manifest_instance():
    manifest = InstallerManifest(version="5.0.0", wheel_url="https://example.test/arc.whl")

    assert bootstrap._coerce_manifest(manifest) is manifest


def test_default_package_url_prefers_environment_override(monkeypatch):
    monkeypatch.setenv("ARC_INSTALL_WHEEL_URL", "https://example.test/custom.zip")

    assert bootstrap.default_package_url() == "https://example.test/custom.zip"


def test_default_package_url_falls_back_to_repository_archive(monkeypatch):
    monkeypatch.delenv("ARC_INSTALL_WHEEL_URL", raising=False)

    assert bootstrap.default_package_url().endswith("/Arc/archive/refs/heads/main.zip")


def test_main_uses_direct_wheel_url_and_reports_success(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    def fake_install(manifest, *, home=None):
        captured["manifest"] = manifest
        captured["home"] = home
        return InstallSummary(required_successes=["runtime_created"])

    monkeypatch.setattr(bootstrap, "install_from_manifest", fake_install)
    monkeypatch.setattr(bootstrap, "evaluate_install_health", lambda *_args: HealthReport(ok=True))

    result = bootstrap.main(
        ["--wheel-url", "https://example.test/arc.whl", "--arc-home", str(tmp_path)],
    )

    assert result == 0
    assert captured["manifest"] == InstallerManifest(version="unknown", wheel_url="https://example.test/arc.whl")
    assert captured["home"] == tmp_path
    assert "Arc installer summary" in capsys.readouterr().out


def test_main_loads_manifest_url_and_returns_failure(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    manifest = InstallerManifest(version="6.0.0", wheel_url="https://example.test/arc.whl")

    def fake_load_manifest(url: str) -> InstallerManifest:
        captured["url"] = url
        return manifest

    monkeypatch.setattr(bootstrap, "load_manifest", fake_load_manifest)

    def fake_install(loaded_manifest, *, home=None):
        captured["manifest"] = loaded_manifest
        captured["home"] = home
        return InstallSummary(required_failures=["wheel_install_failed"])

    monkeypatch.setattr(bootstrap, "install_from_manifest", fake_install)
    monkeypatch.setattr(bootstrap, "evaluate_install_health", lambda *_args: HealthReport(ok=False))

    result = bootstrap.main(
        ["--manifest-url", "https://example.test/manifest.json", "--arc-home", str(tmp_path)],
    )

    assert result == 1
    assert captured["url"] == "https://example.test/manifest.json"
    assert captured["manifest"] == manifest
    assert captured["home"] == tmp_path


def test_main_requires_manifest_or_wheel_url():
    with pytest.raises(SystemExit, match="2"):
        bootstrap.main([])
