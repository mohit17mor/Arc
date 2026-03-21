from __future__ import annotations

from arc.install.paths import build_install_paths


def test_install_paths_keep_user_data_root_and_add_runtime_subtree(tmp_path):
    paths = build_install_paths(home=tmp_path)

    assert paths.arc_home == tmp_path / ".arc"
    assert paths.runtime_root == tmp_path / ".arc" / "runtime"
    assert paths.current_runtime == tmp_path / ".arc" / "runtime" / "current"
    assert paths.launcher_root == tmp_path / ".arc" / "bin"
    assert paths.install_metadata == tmp_path / ".arc" / "runtime" / "current" / "install-meta.json"
    assert paths.health_report == tmp_path / ".arc" / "runtime" / "current" / "health-report.json"
