"""Tests for the built-in screen recording skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.core.types import Capability
from arc.skills.builtin.screen_record import ScreenRecordSkill
from arc.screen.recorder import FFmpegScreenRecorder


class _FakeRecorder:
    def __init__(self) -> None:
        self.recording = False
        self.last_output = Path("/tmp/demo.mp4")
        self.last_fps = 30
        self.start_calls: list[tuple[int, str | None]] = []
        self.stop_calls = 0

    def start(self, *, fps: int = 30, output_path: str | None = None) -> Path:
        self.recording = True
        self.last_fps = fps
        self.start_calls.append((fps, output_path))
        if output_path:
            self.last_output = Path(output_path)
        return self.last_output

    def stop(self) -> dict:
        self.recording = False
        self.stop_calls += 1
        return {
            "output_path": str(self.last_output),
            "duration_seconds": 2.5,
            "fps": self.last_fps,
        }

    def status(self) -> dict:
        return {
            "recording": self.recording,
            "output_path": str(self.last_output) if self.recording else "",
            "fps": self.last_fps,
        }


class TestScreenRecordSkillManifest:
    def test_manifest_has_tools(self):
        skill = ScreenRecordSkill()
        manifest = skill.manifest()

        assert manifest.name == "screen_record"
        assert {t.name for t in manifest.tools} == {
            "screen_record_start",
            "screen_record_stop",
            "screen_record_status",
        }

    def test_manifest_capabilities(self):
        skill = ScreenRecordSkill()
        manifest = skill.manifest()

        assert Capability.SYSTEM_PROCESS in manifest.capabilities
        assert Capability.FILE_WRITE in manifest.capabilities


class TestScreenRecordSkillExecution:
    @pytest.fixture
    def recorder(self) -> _FakeRecorder:
        return _FakeRecorder()

    @pytest.fixture
    def skill(self, recorder: _FakeRecorder) -> ScreenRecordSkill:
        return ScreenRecordSkill(recorder=recorder)

    @pytest.mark.asyncio
    async def test_status_idle(self, skill: ScreenRecordSkill):
        result = await skill.execute_tool("screen_record_status", {})

        assert result.success is True
        assert "not recording" in result.output.lower()

    @pytest.mark.asyncio
    async def test_start_recording(self, skill: ScreenRecordSkill, recorder: _FakeRecorder):
        result = await skill.execute_tool("screen_record_start", {"fps": 30})

        assert result.success is True
        assert recorder.recording is True
        assert "started" in result.output.lower()
        assert "/tmp/demo.mp4" in result.output

    @pytest.mark.asyncio
    async def test_start_rejects_invalid_fps(self, skill: ScreenRecordSkill):
        result = await skill.execute_tool("screen_record_start", {"fps": 0})

        assert result.success is False
        assert "fps" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stop_recording_returns_artifact(self, skill: ScreenRecordSkill, recorder: _FakeRecorder):
        await skill.execute_tool("screen_record_start", {"fps": 24})

        result = await skill.execute_tool("screen_record_stop", {})

        assert result.success is True
        assert recorder.recording is False
        assert result.artifacts == ["/tmp/demo.mp4"]
        assert "stopped" in result.output.lower()

    @pytest.mark.asyncio
    async def test_stop_without_active_recording(self, skill: ScreenRecordSkill):
        result = await skill.execute_tool("screen_record_stop", {})

        assert result.success is False
        assert "not recording" in result.error.lower()


class TestFFmpegScreenRecorder:
    def test_missing_ffmpeg_raises_clear_error(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.shutil.which", lambda _: None)

        with pytest.raises(RuntimeError, match="ffmpeg"):
            recorder.start()
