"""Desktop screen recording helpers backed by ffmpeg."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


class FFmpegScreenRecorder:
    """Cross-platform desktop recorder using ffmpeg."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        recordings_dir: Path | None = None,
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._recordings_dir = recordings_dir or (Path.home() / ".arc" / "recordings")
        self._proc: subprocess.Popen[str] | None = None
        self._output_path: Path | None = None
        self._started_at: float | None = None
        self._fps: int = 30

    @property
    def recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        *,
        fps: int = 30,
        output_path: str | None = None,
    ) -> Path:
        """Start recording the desktop."""
        if self.recording:
            raise RuntimeError("Screen recording is already active.")
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")

        ffmpeg = shutil.which(self._ffmpeg_path)
        if not ffmpeg:
            ffmpeg = self._ffmpeg_from_python_package()
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg is required for screen recording but was not found in PATH "
                "and no bundled imageio-ffmpeg binary is available."
            )

        out_path = Path(output_path).expanduser() if output_path else self._default_output_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(ffmpeg, fps, out_path)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._output_path = out_path
        self._started_at = time.time()
        self._fps = fps
        return out_path

    def stop(self) -> dict[str, object]:
        """Stop the active recording and return metadata."""
        if not self.recording or not self._proc or not self._output_path:
            raise RuntimeError("Screen recording is not active.")

        proc = self._proc
        if proc.stdin:
            try:
                proc.stdin.write("q\n")
                proc.stdin.flush()
            except Exception:
                pass

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        duration = 0.0
        if self._started_at is not None:
            duration = max(0.0, time.time() - self._started_at)

        output = {
            "output_path": str(self._output_path),
            "duration_seconds": round(duration, 2),
            "fps": self._fps,
        }

        self._proc = None
        self._output_path = None
        self._started_at = None
        return output

    def status(self) -> dict[str, object]:
        """Return the current recording status."""
        return {
            "recording": self.recording,
            "output_path": str(self._output_path) if self.recording and self._output_path else "",
            "fps": self._fps if self.recording else 0,
            "started_at": self._started_at if self.recording else None,
        }

    def _default_output_path(self) -> Path:
        stamp = time.strftime("%Y-%m-%d/%H%M%S")
        return self._recordings_dir / f"{stamp}.mp4"

    def _build_command(self, ffmpeg: str, fps: int, output_path: Path) -> list[str]:
        system = platform.system().lower()
        if system == "darwin":
            device = self._detect_macos_screen_device(ffmpeg)
            return [
                ffmpeg,
                "-y",
                "-f", "avfoundation",
                "-framerate", str(fps),
                "-capture_cursor", "1",
                "-i", f"{device}:none",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]
        if system == "windows":
            return [
                ffmpeg,
                "-y",
                "-f", "gdigrab",
                "-framerate", str(fps),
                "-draw_mouse", "1",
                "-i", "desktop",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]
        if system == "linux":
            if os.environ.get("WAYLAND_DISPLAY"):
                raise RuntimeError(
                    "Wayland screen recording is not supported yet. Use X11 or add a platform-specific backend."
                )
            display = os.environ.get("DISPLAY")
            if not display:
                raise RuntimeError("DISPLAY is not set, so X11 screen capture cannot start.")
            return [
                ffmpeg,
                "-y",
                "-f", "x11grab",
                "-framerate", str(fps),
                "-draw_mouse", "1",
                "-i", display,
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]
        raise RuntimeError(f"Unsupported OS for screen recording: {platform.system()}")

    def _detect_macos_screen_device(self, ffmpeg: str) -> str:
        """Find the first avfoundation screen device index on macOS."""
        result = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            check=False,
        )
        text = "\n".join(filter(None, [result.stdout, result.stderr]))
        in_video_section = False
        for line in text.splitlines():
            if "AVFoundation video devices" in line:
                in_video_section = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if not in_video_section:
                continue
            match = re.search(r"\[(\d+)\].*Capture screen", line, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fall back to the first listed video device index if the exact label differs.
        in_video_section = False
        for line in text.splitlines():
            if "AVFoundation video devices" in line:
                in_video_section = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if not in_video_section:
                continue
            match = re.search(r"\[(\d+)\]", line)
            if match:
                return match.group(1)

        raise RuntimeError(
            "Could not find a macOS screen capture device via ffmpeg avfoundation."
        )

    def _ffmpeg_from_python_package(self) -> str | None:
        """Return the bundled imageio-ffmpeg executable path if available."""
        try:
            import imageio_ffmpeg
        except Exception:
            return None

        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
