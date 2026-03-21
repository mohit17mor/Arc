from __future__ import annotations

from scripts.release.build_installer_artifacts import build_manifest


def test_release_artifact_builder_writes_manifest():
    manifest = build_manifest(version="0.1.0", wheel_url="https://example.test/arc.whl")

    assert manifest["version"] == "0.1.0"
    assert manifest["wheel_url"] == "https://example.test/arc.whl"
    assert manifest["entry_module"] == "arc.install.bootstrap"
