"""
Speech Synthesizer — text-to-speech provider interface + backends.

Mirrors the ``SpeechProvider`` (STT) pattern in ``transcriber.py``:
one abstract interface, swappable backends, async-first.

Backends (priority order for ``auto`` mode):
    1. **KokoroProvider** — kokoro-onnx, offline, high quality, ~300 MB model.
    2. **SystemTTSProvider** — OS-native subprocess (SAPI5 / ``say`` / espeak-ng).
       Zero dependencies, always available.

Usage::

    synth = create_synthesizer("auto")
    await synth.initialize()
    await synth.speak("Hello, check chat for details.")
    await synth.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SynthesisResult:
    """Metadata returned after a speak/synthesize call."""

    duration_ms: int
    provider: str
    voice: str


class SpeechSynthesizer(ABC):
    """
    Abstract interface for text-to-speech providers.

    Same lifecycle pattern as ``SpeechProvider`` (STT) and
    ``LLMProvider`` — initialize once, call many times.
    """

    async def initialize(self) -> None:
        """Prepare the provider (load models, warm up)."""

    @abstractmethod
    async def speak(self, text: str) -> SynthesisResult:
        """
        Speak *text* through the default audio output.

        Blocks until speech finishes or :meth:`stop` is called.
        """
        ...

    async def stop(self) -> None:
        """Interrupt speech currently in progress."""

    async def shutdown(self) -> None:
        """Release resources. Safe to call multiple times."""

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return provider metadata for introspection."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kokoro ONNX provider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class KokoroProvider(SpeechSynthesizer):
    """
    High-quality offline TTS via kokoro-onnx.

    Requires ``pip install kokoro-onnx`` and two model files:
        - ``kokoro-v1.0.onnx``  (~300 MB or ~80 MB quantized)
        - ``voices-v1.0.bin``   (~5 MB)

    Both can be downloaded from:
        https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

    The provider looks for the files in ``model_dir`` (default:
    ``~/.arc/models/kokoro/``).
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> None:
        self._model_dir = model_dir or (Path.home() / ".arc" / "models" / "kokoro")
        self._voice = voice
        self._speed = speed
        self._lang = lang
        self._kokoro: Any = None  # kokoro_onnx.Kokoro
        self._play_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        model_path = self._model_dir / "kokoro-v1.0.onnx"
        voices_path = self._model_dir / "voices-v1.0.bin"

        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found in {self._model_dir}. "
                "Download them from: "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0\n"
                f"  Expected: {model_path}\n"
                f"           {voices_path}"
            )

        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise ImportError(
                "kokoro-onnx is not installed. Install with: pip install kokoro-onnx"
            )

        loop = asyncio.get_running_loop()
        self._kokoro = await loop.run_in_executor(
            None, lambda: Kokoro(str(model_path), str(voices_path))
        )
        logger.info(f"Kokoro TTS ready (voice={self._voice}, lang={self._lang})")

    async def speak(self, text: str) -> SynthesisResult:
        if self._kokoro is None:
            raise RuntimeError("KokoroProvider not initialized. Call initialize() first.")

        import time
        start = time.monotonic()

        loop = asyncio.get_running_loop()
        samples, sample_rate = await loop.run_in_executor(
            None,
            lambda: self._kokoro.create(
                text, voice=self._voice, speed=self._speed, lang=self._lang,
            ),
        )

        # Play audio via sounddevice (already a voice dependency).
        import sounddevice as sd

        play_done = asyncio.Event()

        def _on_finished() -> None:
            loop.call_soon_threadsafe(play_done.set)

        # We need a way to interrupt, so play non-blocking and wait
        # on an event we can cancel.
        await loop.run_in_executor(
            None,
            lambda: sd.play(samples, samplerate=sample_rate, blocking=False),
        )

        # Wait for playback to finish (approximate via duration).
        duration_s = len(samples) / sample_rate
        try:
            await asyncio.sleep(duration_s + 0.1)
        except asyncio.CancelledError:
            sd.stop()
            raise

        duration_ms = int((time.monotonic() - start) * 1000)
        return SynthesisResult(
            duration_ms=duration_ms,
            provider="kokoro",
            voice=self._voice,
        )

    async def stop(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    async def shutdown(self) -> None:
        self._kokoro = None

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": "kokoro",
            "voice": self._voice,
            "lang": self._lang,
            "speed": self._speed,
            "model_dir": str(self._model_dir),
            "loaded": self._kokoro is not None,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OS-native System TTS provider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SystemTTSProvider(SpeechSynthesizer):
    """
    OS-native TTS via subprocess — zero dependencies.

    - **Windows**: PowerShell → ``System.Speech.Synthesis``
    - **macOS**: ``say`` command
    - **Linux**: ``espeak-ng`` (or ``espeak``)

    Quality varies by platform but it always works without
    installing anything extra.
    """

    def __init__(self, rate: int | None = None) -> None:
        # rate: words/min on macOS (say -r), 0–10 scale on Windows.
        self._rate = rate
        self._process: subprocess.Popen[bytes] | None = None
        self._system = platform.system()
        self._engine: str = ""

    async def initialize(self) -> None:
        self._engine = self._detect_engine()
        if not self._engine:
            raise RuntimeError(
                "No system TTS engine found. "
                "On Linux, install espeak-ng: sudo apt install espeak-ng"
            )
        logger.info(f"System TTS ready (engine={self._engine})")

    async def speak(self, text: str) -> SynthesisResult:
        if not self._engine:
            raise RuntimeError("SystemTTSProvider not initialized.")

        import time
        start = time.monotonic()

        cmd = self._build_command(text)
        loop = asyncio.get_running_loop()

        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )
        self._process = proc

        # Wait for the process in a thread so we don't block the event loop.
        await loop.run_in_executor(None, proc.wait)
        self._process = None

        duration_ms = int((time.monotonic() - start) * 1000)
        return SynthesisResult(
            duration_ms=duration_ms,
            provider="system",
            voice=self._engine,
        )

    async def stop(self) -> None:
        proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            self._process = None

    async def shutdown(self) -> None:
        await self.stop()

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": "system",
            "engine": self._engine,
            "platform": self._system,
        }

    # ── Internal ─────────────────────────────────────────────

    def _detect_engine(self) -> str:
        if self._system == "Windows":
            return "sapi5"
        if self._system == "Darwin":
            if shutil.which("say"):
                return "say"
            return ""
        # Linux / other
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        return ""

    def _build_command(self, text: str) -> list[str]:
        # Sanitise text: remove characters that could break shell commands.
        safe_text = text.replace('"', "'").replace("\n", " ").strip()

        if self._engine == "sapi5":
            rate = self._rate if self._rate is not None else 2
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {rate}; "
                f'$s.Speak("{safe_text}"); '
                "$s.Dispose()"
            )
            return ["powershell", "-NoProfile", "-Command", ps_script]

        if self._engine == "say":
            cmd = ["say"]
            if self._rate is not None:
                cmd.extend(["-r", str(self._rate)])
            cmd.append(safe_text)
            return cmd

        # espeak-ng / espeak
        cmd = [self._engine]
        rate = self._rate if self._rate is not None else 170
        cmd.extend(["-s", str(rate)])
        cmd.append(safe_text)
        return cmd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_synthesizer(
    provider: str = "auto",
    *,
    kokoro_model_dir: Path | None = None,
    kokoro_voice: str = "af_heart",
    kokoro_speed: float = 1.0,
    kokoro_lang: str = "en-us",
    system_rate: int | None = None,
) -> SpeechSynthesizer:
    """
    Create a TTS provider by name.

    Args:
        provider: ``"auto"`` | ``"kokoro"`` | ``"system"``.
            ``auto`` tries Kokoro first, falls back to System TTS.
        kokoro_*: Kokoro-specific settings.
        system_rate: Speech rate for system TTS.

    Returns:
        An uninitialised ``SpeechSynthesizer``.  Call ``initialize()``
        before ``speak()``.
    """
    if provider == "kokoro":
        return KokoroProvider(
            model_dir=kokoro_model_dir,
            voice=kokoro_voice,
            speed=kokoro_speed,
            lang=kokoro_lang,
        )

    if provider == "system":
        return SystemTTSProvider(rate=system_rate)

    if provider == "auto":
        return _AutoProvider(
            kokoro_model_dir=kokoro_model_dir,
            kokoro_voice=kokoro_voice,
            kokoro_speed=kokoro_speed,
            kokoro_lang=kokoro_lang,
            system_rate=system_rate,
        )

    raise ValueError(f"Unknown TTS provider: {provider!r}. Use 'auto', 'kokoro', or 'system'.")


class _AutoProvider(SpeechSynthesizer):
    """Try Kokoro → System TTS.  Transparent fallback."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._active: SpeechSynthesizer | None = None

    async def initialize(self) -> None:
        # Try Kokoro first
        try:
            kokoro = KokoroProvider(
                model_dir=self._kwargs.get("kokoro_model_dir"),
                voice=self._kwargs.get("kokoro_voice", "af_heart"),
                speed=self._kwargs.get("kokoro_speed", 1.0),
                lang=self._kwargs.get("kokoro_lang", "en-us"),
            )
            await kokoro.initialize()
            self._active = kokoro
            logger.info("Auto-TTS: using Kokoro (high quality, offline)")
            return
        except (ImportError, FileNotFoundError) as exc:
            logger.info(f"Auto-TTS: Kokoro unavailable ({exc}), trying system TTS")
        except Exception as exc:
            logger.warning(f"Auto-TTS: Kokoro failed ({exc}), trying system TTS")

        # Fall back to system TTS
        try:
            system = SystemTTSProvider(rate=self._kwargs.get("system_rate"))
            await system.initialize()
            self._active = system
            logger.info("Auto-TTS: using system TTS")
            return
        except RuntimeError as exc:
            logger.warning(f"Auto-TTS: system TTS unavailable ({exc})")

        raise RuntimeError(
            "No TTS provider available. Install kokoro-onnx or ensure "
            "a system speech engine is present."
        )

    async def speak(self, text: str) -> SynthesisResult:
        if self._active is None:
            raise RuntimeError("Auto TTS provider not initialized.")
        return await self._active.speak(text)

    async def stop(self) -> None:
        if self._active:
            await self._active.stop()

    async def shutdown(self) -> None:
        if self._active:
            await self._active.shutdown()
            self._active = None

    def get_info(self) -> dict[str, Any]:
        if self._active:
            info = self._active.get_info()
            info["wrapper"] = "auto"
            return info
        return {"provider": "auto", "active": None}
