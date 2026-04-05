"""
Voice Daemon — connects the VoiceListener to the Arc Gateway.

This is the process that runs when the user types ``arc listen``.
It:
    1. Connects to the Gateway via WebSocket (same protocol as WebChat)
    2. Initializes the speech provider, voice listener, and TTS synthesizer
    3. Runs the listener's state machine
    4. Transcribes speech and forwards text to the gateway
    5. Speaks a condensed summary of the agent's response via TTS
    6. Shows desktop notifications when the agent responds
    7. Plays a confirmation chime on wake word detection

The daemon is a thin orchestrator — all audio logic lives in
VoiceListener, all STT logic lives in SpeechProvider, all TTS
logic lives in SpeechSynthesizer, and text condensation lives
in the condenser module.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import numpy as np

from arc.voice.condenser import condense
from arc.voice.listener import VoiceEvent, VoiceListener, VoiceState
from arc.voice.synthesizer import SpeechSynthesizer, create_synthesizer
from arc.voice.transcriber import SpeechProvider, WhisperLocalProvider

logger = logging.getLogger(__name__)


class VoiceDaemon:
    """
    Connects voice input to the Arc Gateway.

    Requires the gateway to be running (``arc gateway``).
    The daemon acts as a WebSocket client — same protocol as WebChat.
    Messages sent from voice appear in WebChat and Telegram with a
    ``source: "voice"`` tag.

    Usage::

        daemon = VoiceDaemon(
            gateway_url="ws://127.0.0.1:18789/ws",
            whisper_model="base.en",
        )
        await daemon.run()   # blocks until stopped
    """

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789/ws",
        whisper_model: str = "base.en",
        wake_model: str = "hey_jarvis",
        wake_threshold: float = 0.5,
        silence_duration: float = 1.5,
        listen_timeout: float = 30.0,
        status_callback: Callable[[VoiceState, str], None] | None = None,
        tts_provider: str = "auto",
        tts_voice: str = "af_heart",
        tts_speed: float = 1.0,
        synthesizer: SpeechSynthesizer | None = None,
    ) -> None:
        self._gateway_url = gateway_url
        self._transcriber: SpeechProvider = WhisperLocalProvider(
            model_name=whisper_model,
        )
        self._listener = VoiceListener(
            wake_model=wake_model,
            wake_threshold=wake_threshold,
            silence_duration=silence_duration,
            listen_timeout=listen_timeout,
        )
        self._status_callback = status_callback
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._session: Any = None  # aiohttp.ClientSession
        self._running = False
        self._current_state: VoiceState = VoiceState.SLEEPING
        self._sleep_watch_task: asyncio.Task[None] | None = None

        # TTS — if no synthesizer injected, create from provider name
        self._synthesizer: SpeechSynthesizer | None = synthesizer
        self._tts_provider = tts_provider
        self._tts_voice = tts_voice
        self._tts_speed = tts_speed
        self._tts_available = False

    # ━━━ Lifecycle ━━━

    async def run(self) -> None:
        """
        Main entry point — initialize everything, then listen.

        Blocks until stop() is called or the gateway disconnects.
        Requires ``arc gateway`` to be running.
        """
        import aiohttp

        logger.info("Voice daemon starting")

        # 1. Initialize transcriber, wake word detector, and TTS
        await self._transcriber.initialize()
        await self._listener.initialize()
        await self._init_tts()
        self._emit_status("sleep", VoiceState.SLEEPING)

        # 2. Connect to Gateway WebSocket
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self._gateway_url)
        except Exception as e:
            await self._session.close()
            raise ConnectionError(
                f"Cannot connect to Arc Gateway at {self._gateway_url}. "
                f"Is 'arc gateway' running?"
            ) from e

        logger.info(f"Connected to gateway at {self._gateway_url}")
        self._running = True

        # 3. Start gateway response listener in background
        response_task = asyncio.create_task(
            self._response_loop(), name="voice-response"
        )

        # 4. Main loop — process voice events
        try:
            async for event in self._listener.run():
                if not self._running:
                    break
                await self._handle_event(event)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            response_task.cancel()
            try:
                await response_task
            except asyncio.CancelledError:
                pass
            await self._listener.stop()
            self._cancel_sleep_watch()
            if self._synthesizer:
                await self._synthesizer.shutdown()
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session and not self._session.closed:
                await self._session.close()
            logger.info("Voice daemon stopped")

    async def stop(self) -> None:
        """Signal the daemon to shut down."""
        self._running = False
        await self._listener.stop()
        self._cancel_sleep_watch()
        self._emit_status("sleep", VoiceState.SLEEPING)

    # ━━━ Event handling ━━━

    async def _handle_event(self, event: VoiceEvent) -> None:
        """Process a single event from the voice listener."""

        if event.type == "wake_word":
            logger.info("Wake word detected")
            self._play_chime()
            self._emit_status("wake", VoiceState.ACTIVE)

        elif event.type == "speech_ready":
            self._emit_status("processing", VoiceState.PROCESSING)
            audio: np.ndarray = event.data["audio"]
            result = await self._transcriber.transcribe(audio)

            if not result.text:
                logger.debug("Empty transcription — ignoring")
                await self._listener.notify_response_done()
                self._emit_status("listen", VoiceState.LISTENING)
                self._start_sleep_watch()
                return

            logger.info(
                f"Transcribed ({result.duration_ms}ms, "
                f"conf={result.confidence:.2f}): {result.text}"
            )

            # Send to gateway — same format as WebChat
            if self._ws and not self._ws.closed:
                await self._ws.send_json({
                    "type": "message",
                    "content": result.text,
                    "source": "voice",
                })
            else:
                logger.warning("Gateway disconnected — cannot send")
                self._running = False

        elif event.type == "state_change":
            logger.debug(f"State → {event.state.value}")
            self._emit_status("state", event.state)
            if event.state == VoiceState.SLEEPING:
                self._cancel_sleep_watch()

    # ━━━ Gateway response listener ━━━

    async def _response_loop(self) -> None:
        """
        Listen for gateway responses and transition the listener.

        When the agent finishes responding (type=done), we:
        - Show a desktop notification with the response
        - Tell the listener to move to LISTENING state
        """
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")

                    if msg_type == "done":
                        full_content = data.get("full_content", "")
                        self._notify(full_content)
                        await self._speak_response(full_content)
                        await self._listener.notify_response_done()
                        self._emit_status("listen", VoiceState.LISTENING)
                        self._start_sleep_watch()

                    elif msg_type == "error":
                        error_msg = data.get("message", "Unknown error")
                        logger.warning(f"Gateway error: {error_msg}")
                        self._notify(f"Error: {error_msg}")
                        await self._listener.notify_response_done()
                        self._emit_status("listen", VoiceState.LISTENING)
                        self._start_sleep_watch()

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    logger.warning("Gateway WebSocket closed")
                    self._running = False
                    self._emit_status("sleep", VoiceState.SLEEPING)
                    break

        except asyncio.CancelledError:
            pass

    # ━━━ TTS ━━━

    async def _init_tts(self) -> None:
        """Initialize the TTS synthesizer.  Non-fatal if it fails."""
        if self._synthesizer is None:
            self._synthesizer = create_synthesizer(
                self._tts_provider,
                kokoro_voice=self._tts_voice,
                kokoro_speed=self._tts_speed,
            )
        try:
            await self._synthesizer.initialize()
            self._tts_available = True
            logger.info(f"TTS ready: {self._synthesizer.get_info()}")
        except Exception as exc:
            logger.warning(f"TTS unavailable — responses will not be spoken: {exc}")
            self._tts_available = False

    async def _speak_response(self, text: str) -> None:
        """Condense *text* and speak it.  Non-fatal on error."""
        if not self._tts_available or not self._synthesizer:
            return

        try:
            condensed = condense(text)
            if not condensed.spoken_text:
                return
            logger.debug(
                f"TTS [{condensed.source}]: {condensed.spoken_text!r}"
            )
            await self._synthesizer.speak(condensed.spoken_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"TTS speak failed: {exc}")

    # ━━━ User feedback ━━━

    @staticmethod
    def _play_chime() -> None:
        """Play a short confirmation tone when the wake word is detected."""
        try:
            import sounddevice as sd
            sample_rate = 16000
            duration = 0.35
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            base = np.sin(2 * np.pi * 660 * t)
            overtone = np.sin(2 * np.pi * 990 * t)
            envelope = np.exp(-3.5 * t)
            attack_len = int(0.02 * sample_rate)
            if attack_len > 0:
                attack = np.linspace(0.0, 1.0, attack_len)
                envelope[:attack_len] *= attack
            tone = 0.4 * (0.7 * base + 0.3 * overtone) * envelope
            sd.play(tone.astype(np.float32), samplerate=sample_rate, blocking=False)
        except Exception:
            pass  # non-critical

    @staticmethod
    def _notify(text: str) -> None:
        """Show a desktop notification with the agent's response."""
        try:
            from plyer import notification

            notification.notify(
                title="Arc",
                message=(text[:300] + "…") if len(text) > 300 else text,
                timeout=10,
            )
        except ImportError:
            pass  # plyer not installed — skip silently
        except Exception as e:
            logger.debug(f"Notification failed: {e}")

    # ━━━ Status signalling ━━━

    def _emit_status(self, event: str, state: VoiceState) -> None:
        self._current_state = state
        if self._status_callback:
            try:
                self._status_callback(state, event)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Status callback error: {exc}")

    def _start_sleep_watch(self) -> None:
        if not self._running:
            return
        self._cancel_sleep_watch()

        async def _sleep_monitor() -> None:
            try:
                while self._running:
                    if self._listener.state == VoiceState.SLEEPING:
                        self._emit_status("sleep", VoiceState.SLEEPING)
                        return
                    await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                pass

        self._sleep_watch_task = asyncio.create_task(
            _sleep_monitor(), name="voice-sleep-monitor"
        )

    def _cancel_sleep_watch(self) -> None:
        if self._sleep_watch_task and not self._sleep_watch_task.done():
            self._sleep_watch_task.cancel()
        self._sleep_watch_task = None

    @property
    def current_state(self) -> VoiceState:
        return self._current_state
