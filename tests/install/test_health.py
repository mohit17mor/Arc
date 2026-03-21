from __future__ import annotations

from arc.install.health import evaluate_install_health


def test_health_reports_missing_launcher_as_blocking(tmp_path):
    report = evaluate_install_health(tmp_path / ".arc")

    assert report.ok is False
    assert "launcher" in " ".join(report.blocking_issues).lower()
