"""Tests for the built-in clipboard skill."""

from __future__ import annotations

import subprocess

import pytest

from arc.skills.builtin.clipboard import ClipboardSkill
from arc.clipboard import ClipboardReadResult, SystemClipboardReader


class _FakeReader:
    def __init__(self, result: ClipboardReadResult) -> None:
        self._result = result
        self.calls = 0

    def read(self) -> ClipboardReadResult:
        self.calls += 1
        return self._result


class TestClipboardSkillManifest:
    def test_manifest_has_expected_shape(self) -> None:
        skill = ClipboardSkill(reader=_FakeReader(ClipboardReadResult.from_text("hello", source="test")))
        manifest = skill.manifest()

        assert manifest.name == "clipboard"
        assert manifest.always_available is True
        assert {tool.name for tool in manifest.tools} == {"get_clipboard_text"}

        tool = manifest.tools[0]
        desc = tool.description.lower()
        assert "copied text" in desc
        assert "do not call" in desc


class TestClipboardSkillExecution:
    @pytest.mark.asyncio
    async def test_get_clipboard_text_returns_current_text(self) -> None:
        skill = ClipboardSkill(reader=_FakeReader(ClipboardReadResult.from_text("Explain this", source="test")))

        result = await skill.execute_tool("get_clipboard_text", {})

        assert result.success is True
        assert "Clipboard text" in result.output
        assert "Explain this" in result.output

    @pytest.mark.asyncio
    async def test_get_clipboard_text_rejects_empty_clipboard(self) -> None:
        skill = ClipboardSkill(reader=_FakeReader(ClipboardReadResult.empty(source="test")))

        result = await skill.execute_tool("get_clipboard_text", {})

        assert result.success is False
        assert "empty" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_get_clipboard_text_rejects_non_text_clipboard(self) -> None:
        skill = ClipboardSkill(reader=_FakeReader(ClipboardReadResult.non_text(source="test")))

        result = await skill.execute_tool("get_clipboard_text", {})

        assert result.success is False
        assert "does not currently contain text" in (result.error or "").lower()


class TestSystemClipboardReader:
    def test_macos_uses_pbpaste(self, monkeypatch) -> None:
        reader = SystemClipboardReader()

        monkeypatch.setattr("arc.clipboard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("arc.clipboard.shutil.which", lambda name: "/usr/bin/pbpaste" if name == "pbpaste" else None)

        def _fake_run(cmd, capture_output, check):
            assert cmd == ["/usr/bin/pbpaste"]
            assert capture_output is True
            assert check is False
            return subprocess.CompletedProcess(cmd, 0, stdout=b"Copied from macOS", stderr=b"")

        monkeypatch.setattr("arc.clipboard.subprocess.run", _fake_run)

        result = reader.read()

        assert result.kind == "text"
        assert result.text == "Copied from macOS"

    def test_windows_uses_powershell_get_clipboard(self, monkeypatch) -> None:
        reader = SystemClipboardReader()

        monkeypatch.setattr("arc.clipboard.platform.system", lambda: "Windows")

        def _fake_run(cmd, capture_output, check):
            assert cmd == [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Clipboard -Raw",
            ]
            return subprocess.CompletedProcess(cmd, 0, stdout="Copied from Windows", stderr="")

        monkeypatch.setattr("arc.clipboard.subprocess.run", _fake_run)

        result = reader.read()

        assert result.kind == "text"
        assert result.text == "Copied from Windows"

    def test_linux_prefers_wayland_then_x11_commands(self, monkeypatch) -> None:
        reader = SystemClipboardReader()

        monkeypatch.setattr("arc.clipboard.platform.system", lambda: "Linux")
        monkeypatch.setattr(
            "arc.clipboard.shutil.which",
            lambda name: "/usr/bin/xclip" if name == "xclip" else None,
        )

        def _fake_run(cmd, capture_output, check):
            assert cmd == ["/usr/bin/xclip", "-selection", "clipboard", "-out"]
            return subprocess.CompletedProcess(cmd, 0, stdout=b"Copied from Linux", stderr=b"")

        monkeypatch.setattr("arc.clipboard.subprocess.run", _fake_run)

        result = reader.read()

        assert result.kind == "text"
        assert result.text == "Copied from Linux"

    def test_unsupported_platform_returns_error(self, monkeypatch) -> None:
        reader = SystemClipboardReader()
        monkeypatch.setattr("arc.clipboard.platform.system", lambda: "Plan9")

        result = reader.read()

        assert result.kind == "error"
        assert "unsupported" in (result.error or "").lower()
