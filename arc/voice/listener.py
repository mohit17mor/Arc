"""
Voice Listener — microphone capture + wake word + state machine.

Manages the entire audio pipeline:
    1. Continuous microphone capture via sounddevice
    2. Wake word detection via openwakeword
    3. Silence detection (energy-based)
    4. State machine controlling when to record

States:
    SLEEPING    — only wake word detector runs (~2% CPU)
    ACTIVE      — recording speech, silence detection active
    PROCESSING  — microphone paused, waiting for agent response
    LISTENING   — post-response window, follow-up without wake word

The listener yields VoiceEvent objects so the daemon can react
without the listener knowing about WebSockets or notifications.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)

# Audio constants — openwakeword expects 16kHz mono, 80ms chunks
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz


class VoiceState(str, Enum):
    """Current state of the voice listener."""

    SLEEPING = "sleeping"
    ACTIVE = "active"
    PROCESSING = "processing"
    LISTENING = "listening"


@dataclass
class VoiceEvent:
    """An event produced by the voice listener."""

    type: str  # "wake_word", "speech_ready", "state_change", "error"
    state: VoiceState
    data: dict[str, Any] = field(default_factory=dict)


class VoiceListener:
    """
    Microphone capture + wake word detection + silence detection.

    Yields VoiceEvent objects as an async iterator.  The caller
    (VoiceDaemon) handles transcription and gateway communication.

    Usage::

        listener = VoiceListener(wake_model="hey_jarvis")
        await listener.initialize()

        async for event in listener.run():
            if event.type == "wake_word":
                print("Wake word detected!")
            elif event.type == "speech_ready":
                audio = event.data["audio"]
                # → transcribe audio
    """

    def __init__(
        self,
        wake_model: str = "hey_jarvis",
        wake_threshold: float = 0.5,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        listen_timeout: float = 30.0,
    ) -> None:
        self._wake_model = wake_model
        self._wake_threshold = wake_threshold
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._listen_timeout = listen_timeout

        # State
        self._state = VoiceState.SLEEPING
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Audio pipeline
        self._audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream: Any = None  # sounddevice.InputStream
        self._oww_model: Any = None  # openwakeword Model

        # Recording
        self._recording_buffer: list[np.ndarray] = []
        self._silent_samples = 0

        # Timeout management
        self._listen_timeout_task: asyncio.Task | None = None

    @property
    def state(self) -> VoiceState:
        return self._state

    # ━━━ Lifecycle ━━━

    async def initialize(self) -> None:
        """Load the wake word model. Call once before run()."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_wake_word)

    def _init_wake_word(self) -> None:
        import openwakeword
        from openwakeword.model import Model

        # Download pre-trained models if not already present
        openwakeword.utils.download_models()

        # Use ONNX runtime (cross-platform, no tflite dependency needed)
        self._oww_model = Model(
            wakeword_models=[self._wake_model],
            inference_framework="onnx",
        )
        logger.info(f"Wake word model '{self._wake_model}' loaded")

    async def run(self) -> AsyncIterator[VoiceEvent]:
        """
        Main loop — capture audio and yield events.

        Runs until stop() is called.
        """
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._state = VoiceState.SLEEPING
        self._start_mic()

        yield VoiceEvent(type="state_change", state=VoiceState.SLEEPING)

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue  # check self._running

                if self._state == VoiceState.SLEEPING:
                    if self._detect_wake_word(chunk):
                        self._state = VoiceState.ACTIVE
                        self._recording_buffer.clear()
                        self._silent_samples = 0
                        self._oww_model.reset()
                        yield VoiceEvent(
                            type="wake_word", state=VoiceState.ACTIVE
                        )

                elif self._state == VoiceState.ACTIVE:
                    self._recording_buffer.append(chunk)

                    if self._detect_silence(chunk):
                        audio = np.concatenate(self._recording_buffer)
                        self._recording_buffer.clear()
                        self._state = VoiceState.PROCESSING
                        self._pause_mic()
                        yield VoiceEvent(
                            type="speech_ready",
                            state=VoiceState.PROCESSING,
                            data={"audio": audio},
                        )

                elif self._state == VoiceState.LISTENING:
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    if rms > self._silence_threshold:
                        # User started a follow-up — no wake word needed
                        self._cancel_listen_timeout()
                        self._state = VoiceState.ACTIVE
                        self._recording_buffer.clear()
                        self._recording_buffer.append(chunk)
                        self._silent_samples = 0
                        yield VoiceEvent(
                            type="state_change", state=VoiceState.ACTIVE
                        )

                # PROCESSING: mic is paused, no chunks arrive

        finally:
            self._stop_mic()

    async def notify_response_done(self) -> None:
        """
        Called by the daemon when the agent finishes responding.

        Transitions to LISTENING and resumes the microphone so the
        user can speak a follow-up without the wake word.
        """
        self._state = VoiceState.LISTENING
        self._resume_mic()
        self._cancel_listen_timeout()
        self._listen_timeout_task = asyncio.create_task(
            self._listen_timeout_loop()
        )

    async def stop(self) -> None:
        """Stop the listener and release the microphone."""
        self._running = False
        self._cancel_listen_timeout()

    # ━━━ Wake word detection ━━━

    def _detect_wake_word(self, chunk: np.ndarray) -> bool:
        """Feed a chunk to openwakeword and check for activation."""
        if self._oww_model is None:
            return False
        # Convert float32 [-1,1] → int16 for openwakeword
        audio_int16 = (chunk * 32767).astype(np.int16)
        prediction = self._oww_model.predict(audio_int16)
        score = prediction.get(self._wake_model, 0)
        return score >= self._wake_threshold

    # ━━━ Silence detection ━━━

    def _detect_silence(self, chunk: np.ndarray) -> bool:
        """
        Track consecutive silent samples.

        Returns True when silence has lasted >= silence_duration.
        """
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < self._silence_threshold:
            self._silent_samples += len(chunk)
        else:
            self._silent_samples = 0
        required = int(self._silence_duration * SAMPLE_RATE)
        return self._silent_samples >= required

    # ━━━ Listen timeout ━━━

    async def _listen_timeout_loop(self) -> None:
        """After listen_timeout seconds without speech, go back to SLEEPING."""
        try:
            await asyncio.sleep(self._listen_timeout)
            if self._state == VoiceState.LISTENING:
                self._state = VoiceState.SLEEPING
                logger.debug(
                    f"No follow-up for {self._listen_timeout}s — sleeping"
                )
        except asyncio.CancelledError:
            pass  # cancelled because user spoke or daemon stopped

    def _cancel_listen_timeout(self) -> None:
        if self._listen_timeout_task and not self._listen_timeout_task.done():
            self._listen_timeout_task.cancel()
            self._listen_timeout_task = None

    # ━━━ Microphone ━━━

    def _start_mic(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.debug("Microphone started")

    def _pause_mic(self) -> None:
        if self._stream and self._stream.active:
            self._stream.stop()
            # Drain any queued chunks so LISTENING starts clean
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.debug("Microphone paused")

    def _resume_mic(self) -> None:
        if self._stream and not self._stream.active:
            self._stream.start()
            logger.debug("Microphone resumed")

    def _stop_mic(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
            logger.debug("Microphone stopped")

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """sounddevice callback — runs in a separate thread."""
        if status:
            logger.debug(f"Audio status: {status}")
        audio = indata[:, 0].copy()  # mono float32
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._audio_queue.put_nowait, audio
            )
