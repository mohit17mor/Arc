from __future__ import annotations

import json
from pathlib import Path


def build_manifest(*, version: str, wheel_url: str, checksums: dict[str, str] | None = None) -> dict[str, object]:
    return {
        "version": version,
        "wheel_url": wheel_url,
        "entry_module": "arc.install.bootstrap",
        "checksums": checksums or {},
    }


def write_manifest(target: Path, *, version: str, wheel_url: str, checksums: dict[str, str] | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_manifest(version=version, wheel_url=wheel_url, checksums=checksums), indent=2),
        encoding="utf-8",
    )
    return target

