"""Regression tests for MCP import boundaries."""

from __future__ import annotations

import importlib
import sys


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_arc_mcp_package_import_is_lazy():
    _clear_modules(
        "mcp",
        "mcp.client",
        "mcp.client.sse",
        "mcp.client.stdio",
        "mcp.types",
        "arc.mcp",
        "arc.mcp.client",
    )

    importlib.import_module("arc.mcp")

    assert "mcp" not in sys.modules


def test_arc_mcp_client_import_does_not_load_sdk_until_needed():
    _clear_modules(
        "mcp",
        "mcp.client",
        "mcp.client.sse",
        "mcp.client.stdio",
        "mcp.types",
        "arc.mcp",
        "arc.mcp.client",
    )

    importlib.import_module("arc.mcp.client")

    assert "mcp" not in sys.modules
