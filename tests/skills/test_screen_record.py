"""Tests for the built-in screen recording skill."""

from __future__ import annotations

import builtins
import subprocess
import sys
from types import SimpleNamespace
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


class _FakePipe:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushed = 0

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        self.flushed += 1


class _FakeProc:
    def __init__(self, *, poll_result=None, wait_outcomes: list[object] | None = None) -> None:
        self._poll_result = poll_result
        self.stdin = _FakePipe()
        self.wait_outcomes = list(wait_outcomes or [None])
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self._poll_result

    def wait(self, timeout=None):
        outcome = self.wait_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


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

    def test_recording_property_reflects_process_state(self, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)

        assert recorder.recording is False

        recorder._proc = _FakeProc(poll_result=None)
        assert recorder.recording is True

        recorder._proc = _FakeProc(poll_result=0)
        assert recorder.recording is False

    def test_start_rejects_invalid_fps(self, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)

        with pytest.raises(ValueError, match="fps"):
            recorder.start(fps=0)

    def test_start_rejects_when_recording_is_already_active(self, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        recorder._proc = _FakeProc(poll_result=None)

        with pytest.raises(RuntimeError, match="already active"):
            recorder.start()

    def test_start_uses_bundled_ffmpeg_and_updates_status(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        proc = _FakeProc(poll_result=None)
        popen_calls: list[tuple[list[str], dict[str, object]]] = []
        built: list[tuple[str, int, Path]] = []

        monkeypatch.setattr("arc.screen.recorder.shutil.which", lambda _: None)
        monkeypatch.setattr(recorder, "_ffmpeg_from_python_package", lambda: "/pkg/ffmpeg")
        monkeypatch.setattr(
            recorder,
            "_build_command",
            lambda ffmpeg, fps, output: built.append((ffmpeg, fps, output)) or [ffmpeg, str(output)],
        )
        monkeypatch.setattr(
            "arc.screen.recorder.subprocess.Popen",
            lambda cmd, **kwargs: popen_calls.append((cmd, kwargs)) or proc,
        )

        output_path = recorder.start(fps=24, output_path=str(tmp_path / "capture.mp4"))

        assert output_path == tmp_path / "capture.mp4"
        assert built == [("/pkg/ffmpeg", 24, tmp_path / "capture.mp4")]
        assert popen_calls[0][0] == ["/pkg/ffmpeg", str(tmp_path / "capture.mp4")]
        assert recorder.status()["recording"] is True
        assert recorder.status()["fps"] == 24
        assert recorder.status()["output_path"] == str(tmp_path / "capture.mp4")

    def test_status_returns_idle_values_when_not_recording(self, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)

        assert recorder.status() == {
            "recording": False,
            "output_path": "",
            "fps": 0,
            "started_at": None,
        }

    def test_default_output_path_nests_recordings_by_date(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.time.strftime", lambda _fmt: "2026-03-23/112233")

        assert recorder._default_output_path() == tmp_path / "2026-03-23/112233.mp4"

    def test_build_command_for_macos_uses_detected_screen_device(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.platform.system", lambda: "Darwin")
        monkeypatch.setattr(recorder, "_detect_macos_screen_device", lambda ffmpeg: "7")

        command = recorder._build_command("/usr/bin/ffmpeg", 60, tmp_path / "capture.mp4")

        assert command == [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "avfoundation",
            "-framerate", "60",
            "-capture_cursor", "1",
            "-i", "7:none",
            "-pix_fmt", "yuv420p",
            str(tmp_path / "capture.mp4"),
        ]

    def test_build_command_for_windows_uses_desktop_capture(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.platform.system", lambda: "Windows")

        command = recorder._build_command("/usr/bin/ffmpeg", 30, tmp_path / "capture.mp4")

        assert command == [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "gdigrab",
            "-framerate", "30",
            "-draw_mouse", "1",
            "-i", "desktop",
            "-pix_fmt", "yuv420p",
            str(tmp_path / "capture.mp4"),
        ]

    def test_build_command_for_linux_uses_x11_display(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.platform.system", lambda: "Linux")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":99")

        command = recorder._build_command("/usr/bin/ffmpeg", 15, tmp_path / "capture.mp4")

        assert command == [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "x11grab",
            "-framerate", "15",
            "-draw_mouse", "1",
            "-i", ":99",
            "-pix_fmt", "yuv420p",
            str(tmp_path / "capture.mp4"),
        ]

    def test_build_command_rejects_wayland_without_backend(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.platform.system", lambda: "Linux")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

        with pytest.raises(RuntimeError, match="Wayland"):
            recorder._build_command("/usr/bin/ffmpeg", 30, tmp_path / "capture.mp4")

    def test_build_command_requires_display_on_linux(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.platform.system", lambda: "Linux")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)

        with pytest.raises(RuntimeError, match="DISPLAY is not set"):
            recorder._build_command("/usr/bin/ffmpeg", 30, tmp_path / "capture.mp4")

    def test_build_command_rejects_unsupported_operating_system(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        monkeypatch.setattr("arc.screen.recorder.platform.system", lambda: "Solaris")

        with pytest.raises(RuntimeError, match="Unsupported OS"):
            recorder._build_command("/usr/bin/ffmpeg", 30, tmp_path / "capture.mp4")

    def test_detect_macos_screen_device_prefers_exact_capture_screen_label(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        output = """
AVFoundation video devices:
[2] FaceTime HD Camera
[4] Capture screen 0
AVFoundation audio devices:
"""
        monkeypatch.setattr(
            "arc.screen.recorder.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=output),
        )

        assert recorder._detect_macos_screen_device("/usr/bin/ffmpeg") == "4"

    def test_detect_macos_screen_device_falls_back_to_first_video_device(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        output = """
AVFoundation video devices:
[1] FaceTime HD Camera
[3] External Display
AVFoundation audio devices:
"""
        monkeypatch.setattr(
            "arc.screen.recorder.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=output),
        )

        assert recorder._detect_macos_screen_device("/usr/bin/ffmpeg") == "1"

    def test_detect_macos_screen_device_raises_when_no_video_device_matches(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        output = """
AVFoundation video devices:
AVFoundation audio devices:
"""
        monkeypatch.setattr(
            "arc.screen.recorder.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=output),
        )

        with pytest.raises(RuntimeError, match="Could not find"):
            recorder._detect_macos_screen_device("/usr/bin/ffmpeg")

    def test_stop_returns_recording_metadata_and_resets_state(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        proc = _FakeProc(poll_result=None)
        recorder._proc = proc
        recorder._output_path = tmp_path / "capture.mp4"
        recorder._started_at = 100.0
        recorder._fps = 25
        monkeypatch.setattr("arc.screen.recorder.time.time", lambda: 104.321)

        result = recorder.stop()

        assert result == {
            "output_path": str(tmp_path / "capture.mp4"),
            "duration_seconds": 4.32,
            "fps": 25,
        }
        assert proc.stdin.writes == ["q\n"]
        assert proc.stdin.flushed == 1
        assert recorder.recording is False
        assert recorder.status()["recording"] is False

    def test_stop_terminates_process_after_wait_timeout(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        proc = _FakeProc(
            poll_result=None,
            wait_outcomes=[subprocess.TimeoutExpired("ffmpeg", 10), None],
        )
        recorder._proc = proc
        recorder._output_path = tmp_path / "capture.mp4"
        recorder._started_at = 50.0
        monkeypatch.setattr("arc.screen.recorder.time.time", lambda: 55.0)

        recorder.stop()

        assert proc.terminate_calls == 1
        assert proc.kill_calls == 0

    def test_stop_kills_process_after_second_wait_timeout(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        proc = _FakeProc(
            poll_result=None,
            wait_outcomes=[
                subprocess.TimeoutExpired("ffmpeg", 10),
                subprocess.TimeoutExpired("ffmpeg", 5),
                None,
            ],
        )
        recorder._proc = proc
        recorder._output_path = tmp_path / "capture.mp4"
        recorder._started_at = 10.0
        monkeypatch.setattr("arc.screen.recorder.time.time", lambda: 15.0)

        recorder.stop()

        assert proc.terminate_calls == 1
        assert proc.kill_calls == 1

    def test_stop_requires_active_recording(self, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)

        with pytest.raises(RuntimeError, match="not active"):
            recorder.stop()

    def test_ffmpeg_from_python_package_returns_none_when_import_fails(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "imageio_ffmpeg":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert recorder._ffmpeg_from_python_package() is None

    def test_ffmpeg_from_python_package_returns_none_when_getter_fails(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)

        class _BrokenFFmpegModule:
            @staticmethod
            def get_ffmpeg_exe():
                raise RuntimeError("boom")

        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _BrokenFFmpegModule())

        assert recorder._ffmpeg_from_python_package() is None

    def test_ffmpeg_from_python_package_returns_bundled_binary(self, monkeypatch, tmp_path):
        recorder = FFmpegScreenRecorder(recordings_dir=tmp_path)

        class _FFmpegModule:
            @staticmethod
            def get_ffmpeg_exe():
                return "/pkg/ffmpeg"

        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _FFmpegModule())

        assert recorder._ffmpeg_from_python_package() == "/pkg/ffmpeg"
