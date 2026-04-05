"""Cross-platform clipboard helpers."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import shutil
import subprocess


@dataclass(slots=True)
class ClipboardReadResult:
    """Result of attempting to read the system clipboard."""

    kind: str
    source: str
    text: str | None = None
    error: str | None = None

    @classmethod
    def from_text(cls, text: str, *, source: str) -> "ClipboardReadResult":
        return cls(kind="text", text=text, source=source)

    @classmethod
    def empty(cls, *, source: str) -> "ClipboardReadResult":
        return cls(kind="empty", source=source)

    @classmethod
    def non_text(cls, *, source: str) -> "ClipboardReadResult":
        return cls(kind="non_text", source=source)

    @classmethod
    def error_result(cls, error: str, *, source: str) -> "ClipboardReadResult":
        return cls(kind="error", error=error, source=source)


class SystemClipboardReader:
    """Read clipboard text using platform-native commands."""

    def read(self) -> ClipboardReadResult:
        system = platform.system()
        try:
            command, source = self._build_command(system)
        except RuntimeError as exc:
            return ClipboardReadResult.error_result(str(exc), source=system.lower())

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return ClipboardReadResult.error_result(str(exc), source=source)

        if completed.returncode != 0:
            stderr = self._coerce_output(completed.stderr).strip() or "Clipboard read failed."
            return ClipboardReadResult.error_result(stderr, source=source)

        stdout = self._coerce_output(completed.stdout)
        if "\x00" in stdout:
            return ClipboardReadResult.non_text(source=source)

        if not stdout.strip():
            return ClipboardReadResult.empty(source=source)

        return ClipboardReadResult.from_text(stdout, source=source)

    def _build_command(self, system: str) -> tuple[list[str], str]:
        if system == "Darwin":
            pbpaste = shutil.which("pbpaste")
            if not pbpaste:
                raise RuntimeError("pbpaste is not available on this macOS system.")
            return [pbpaste], "pbpaste"

        if system == "Windows":
            return [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Clipboard -Raw",
            ], "powershell"

        if system == "Linux":
            wl_paste = shutil.which("wl-paste")
            if wl_paste:
                return [wl_paste, "--no-newline"], "wl-paste"

            xclip = shutil.which("xclip")
            if xclip:
                return [xclip, "-selection", "clipboard", "-out"], "xclip"

            xsel = shutil.which("xsel")
            if xsel:
                return [xsel, "--clipboard", "--output"], "xsel"

            raise RuntimeError(
                "No supported Linux clipboard utility was found. "
                "Install wl-clipboard, xclip, or xsel."
            )

        raise RuntimeError(f"Unsupported platform for clipboard access: {system}")

    @staticmethod
    def _coerce_output(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
