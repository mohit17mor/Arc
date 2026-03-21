from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from arc.install.health import evaluate_install_health, write_health_report
from arc.install.launcher import write_launcher
from arc.install.paths import build_install_paths
from arc.install.python_runtime import ensure_runtime
from arc.install.types import InstallSummary, InstallerManifest


def build_wheel_install_command(python_executable: str, wheel_url: str) -> list[str]:
    return [python_executable, "-m", "pip", "install", "--upgrade", wheel_url]


def record_browser_setup_failure(summary: InstallSummary, message: str) -> InstallSummary:
    summary.optional_failures.append(message)
    return summary


def install_from_manifest(manifest: InstallerManifest | dict[str, object], *, home: Path | None = None) -> InstallSummary:
    manifest_obj = _coerce_manifest(manifest)
    paths = build_install_paths(home=home)
    runtime = ensure_runtime(paths.current_runtime)
    summary = InstallSummary(
        required_successes=["runtime_created"],
        details={"wheel_url": manifest_obj.wheel_url, "version": manifest_obj.version},
    )

    try:
        subprocess.run(
            build_wheel_install_command(str(runtime.venv_python), manifest_obj.wheel_url),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        summary.required_failures.append(f"wheel_install_failed: {exc.stderr.strip() or exc.stdout.strip() or exc}")
        write_health_report(paths.arc_home, browser_ready=False)
        return summary

    summary.required_successes.append("wheel_installed")
    write_launcher(paths.managed_launcher, runtime.venv_python)
    summary.required_successes.append("launcher_created")
    paths.install_metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.install_metadata.write_text(
        json.dumps(
            {
                "version": manifest_obj.version,
                "package_url": manifest_obj.wheel_url,
                "entry_module": manifest_obj.entry_module,
                "checksums": manifest_obj.checksums,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        subprocess.run(
            [str(runtime.venv_python), "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        summary.optional_successes.append("browser_ready")
        write_health_report(paths.arc_home, browser_ready=True)
    except subprocess.CalledProcessError as exc:
        record_browser_setup_failure(
            summary,
            f"browser_setup_failed: {exc.stderr.strip() or exc.stdout.strip() or exc}",
        )
        write_health_report(paths.arc_home, browser_ready=False)

    return summary


def load_manifest(manifest_url: str) -> InstallerManifest:
    with urllib.request.urlopen(manifest_url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _coerce_manifest(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arc managed installer")
    parser.add_argument("--manifest-url", help="URL to installer manifest JSON")
    parser.add_argument("--wheel-url", help="Direct wheel URL override")
    parser.add_argument("--arc-home", help="Override Arc home directory")
    args = parser.parse_args(argv)

    home = Path(args.arc_home).expanduser() if args.arc_home else None
    if args.wheel_url:
        manifest = InstallerManifest(version="unknown", wheel_url=args.wheel_url)
    elif args.manifest_url:
        manifest = load_manifest(args.manifest_url)
    else:
        parser.error("Provide --manifest-url or --wheel-url.")

    summary = install_from_manifest(manifest, home=home)
    report = evaluate_install_health(build_install_paths(home=home).arc_home)
    print(_render_summary(summary, report))
    return 0 if summary.can_continue and report.ok else 1


def _render_summary(summary: InstallSummary, report) -> str:
    lines = ["Arc installer summary", ""]
    if summary.required_successes:
        lines.append("Required successes:")
        lines.extend(f"- {item}" for item in summary.required_successes)
    if summary.required_failures:
        lines.append("Required failures:")
        lines.extend(f"- {item}" for item in summary.required_failures)
    if summary.optional_successes:
        lines.append("Optional successes:")
        lines.extend(f"- {item}" for item in summary.optional_successes)
    if summary.optional_failures:
        lines.append("Optional failures:")
        lines.extend(f"- {item}" for item in summary.optional_failures)
    lines.extend([
        "",
        f"Install health: {'OK' if report.ok else 'Needs attention'}",
        "Next step: arc init" if report.ok else "Run: arc doctor",
    ])
    return "\n".join(lines)


def _coerce_manifest(manifest: InstallerManifest | dict[str, object]) -> InstallerManifest:
    if isinstance(manifest, InstallerManifest):
        return manifest
    return InstallerManifest(
        version=str(manifest["version"]),
        wheel_url=str(manifest["wheel_url"]),
        entry_module=str(manifest.get("entry_module", "arc.install.bootstrap")),
        checksums=dict(manifest.get("checksums", {})),
    )


def default_package_url() -> str:
    return os.environ.get(
        "ARC_INSTALL_WHEEL_URL",
        "https://github.com/mohit17mor/Arc/archive/refs/heads/main.zip",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
