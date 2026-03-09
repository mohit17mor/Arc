"""Tests for the voice daemon."""

import asyncio
import json

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.voice.daemon import VoiceDaemon
from arc.voice.listener import VoiceEvent, VoiceState
from arc.voice.transcriber import TranscriptionResult


# ── Construction ─────────────────────────────────────────────────


class TestDaemonConstruction:

    def test_default_config(self):
        daemon = VoiceDaemon()
        assert daemon._gateway_url == "ws://127.0.0.1:18789/ws"
        assert daemon._running is False

    def test_custom_config(self):
        daemon = VoiceDaemon(
            gateway_url="ws://remote:9999/ws",
            whisper_model="tiny.en",
            wake_model="alexa",
            silence_duration=2.0,
            listen_timeout=60.0,
        )
        assert daemon._gateway_url == "ws://remote:9999/ws"
        assert daemon._listener._wake_model == "alexa"
        assert daemon._listener._silence_duration == 2.0


# ── Event handling ───────────────────────────────────────────────


class TestEventHandling:

    def _make_daemon(self, status_callback=None) -> VoiceDaemon:
        daemon = VoiceDaemon(status_callback=status_callback)
        daemon._ws = AsyncMock()
        daemon._ws.closed = False
        return daemon

    @pytest.mark.asyncio
    async def test_wake_word_plays_chime(self):
        daemon = self._make_daemon()
        event = VoiceEvent(type="wake_word", state=VoiceState.ACTIVE)

        with patch.object(daemon, "_play_chime") as mock_chime:
            await daemon._handle_event(event)
            mock_chime.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_ready_transcribes_and_sends(self):
        daemon = self._make_daemon()
        daemon._transcriber = AsyncMock()
        daemon._transcriber.transcribe.return_value = TranscriptionResult(
            text="hello world",
            confidence=0.95,
            language="en",
            duration_ms=200,
        )

        audio = np.random.randn(16000).astype(np.float32)
        event = VoiceEvent(
            type="speech_ready",
            state=VoiceState.PROCESSING,
            data={"audio": audio},
        )

        await daemon._handle_event(event)

        daemon._transcriber.transcribe.assert_called_once()
        daemon._ws.send_json.assert_called_once_with({
            "type": "message",
            "content": "hello world",
            "source": "voice",
        })
        assert daemon.current_state == VoiceState.PROCESSING

    @pytest.mark.asyncio
    async def test_empty_transcription_skipped(self):
        daemon = self._make_daemon()
        daemon._transcriber = AsyncMock()
        daemon._transcriber.transcribe.return_value = TranscriptionResult(
            text="",
            confidence=0.0,
            language="en",
            duration_ms=100,
        )
        daemon._listener = AsyncMock()

        audio = np.zeros(16000, dtype=np.float32)
        event = VoiceEvent(
            type="speech_ready",
            state=VoiceState.PROCESSING,
            data={"audio": audio},
        )

        await daemon._handle_event(event)

        # Should not send to gateway
        daemon._ws.send_json.assert_not_called()

        # Should tell listener to resume
        daemon._listener.notify_response_done.assert_called_once()
        assert daemon.current_state == VoiceState.LISTENING

    @pytest.mark.asyncio
    async def test_disconnected_ws_stops_daemon(self):
        daemon = self._make_daemon()
        daemon._ws.closed = True
        daemon._transcriber = AsyncMock()
        daemon._transcriber.transcribe.return_value = TranscriptionResult(
            text="test",
            confidence=0.9,
            language="en",
            duration_ms=100,
        )

        event = VoiceEvent(
            type="speech_ready",
            state=VoiceState.PROCESSING,
            data={"audio": np.zeros(16000, dtype=np.float32)},
        )

        await daemon._handle_event(event)

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_state_change_handled_gracefully(self):
        daemon = self._make_daemon()
        event = VoiceEvent(type="state_change", state=VoiceState.SLEEPING)
        # Should not raise
        await daemon._handle_event(event)


class TestStatusCallback:

    @pytest.mark.asyncio
    async def test_status_callback_receives_events(self):
        events: list[tuple[VoiceState, str]] = []

        def cb(state: VoiceState, event: str) -> None:
            events.append((state, event))

        daemon = VoiceDaemon(status_callback=cb)
        daemon._ws = AsyncMock()
        daemon._ws.closed = False
        daemon._transcriber = AsyncMock()
        daemon._transcriber.transcribe.return_value = TranscriptionResult(
            text="",
            confidence=0.0,
            language="en",
            duration_ms=100,
        )
        daemon._listener = AsyncMock()

        await daemon._handle_event(VoiceEvent(type="wake_word", state=VoiceState.ACTIVE))
        audio = np.zeros(16000, dtype=np.float32)
        await daemon._handle_event(
            VoiceEvent(type="speech_ready", state=VoiceState.PROCESSING, data={"audio": audio})
        )

        assert events[0] == (VoiceState.ACTIVE, "wake")
        assert (VoiceState.PROCESSING, "processing") in events
        assert events[-1] == (VoiceState.LISTENING, "listen")


# ── Gateway message format ───────────────────────────────────────


class TestGatewayProtocol:

    @pytest.mark.asyncio
    async def test_message_format_matches_webchat(self):
        """Voice messages use the same format as WebChat."""
        daemon = VoiceDaemon()
        daemon._ws = AsyncMock()
        daemon._ws.closed = False
        daemon._transcriber = AsyncMock()
        daemon._transcriber.transcribe.return_value = TranscriptionResult(
            text="search for flights",
            confidence=0.92,
            language="en",
            duration_ms=300,
        )

        event = VoiceEvent(
            type="speech_ready",
            state=VoiceState.PROCESSING,
            data={"audio": np.zeros(16000, dtype=np.float32)},
        )

        await daemon._handle_event(event)

        call_args = daemon._ws.send_json.call_args[0][0]
        assert call_args["type"] == "message"
        assert call_args["content"] == "search for flights"


# ── Notification ─────────────────────────────────────────────────


class TestNotification:

    def test_notify_with_plyer(self):
        with patch("arc.voice.daemon.logger"):
            # _notify doesn't raise even if plyer is missing
            VoiceDaemon._notify("test response")

    def test_notify_truncates_long_text(self):
        """Long responses are truncated for the notification."""
        # Just verify it doesn't crash — plyer may not be installed
        VoiceDaemon._notify("x" * 500)


# ── Chime ────────────────────────────────────────────────────────


class TestChime:

    def test_chime_does_not_raise(self):
        """Chime is non-critical — should never raise even if sounddevice is missing."""
        VoiceDaemon._play_chime()

    def test_chime_suppresses_errors(self):
        """If sounddevice isn't available, chime is silently skipped."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            # Should not raise
            VoiceDaemon._play_chime()


# ── VoiceConfig integration ──────────────────────────────────────


class TestVoiceConfig:

    def test_config_defaults(self):
        from arc.core.config import VoiceConfig

        cfg = VoiceConfig()
        assert cfg.wake_model == "hey_jarvis"
        assert cfg.whisper_model == "base.en"
        assert cfg.wake_threshold == 0.5
        assert cfg.silence_duration == 1.5
        assert cfg.listen_timeout == 30.0

    def test_config_in_arc_config(self):
        from arc.core.config import ArcConfig

        cfg = ArcConfig()
        assert hasattr(cfg, "voice")
        assert cfg.voice.wake_model == "hey_jarvis"

    def test_config_from_dict(self):
        from arc.core.config import ArcConfig

        cfg = ArcConfig(
            voice={
                "wake_model": "alexa",
                "whisper_model": "tiny.en",
                "silence_duration": 2.0,
                "listen_timeout": 60.0,
            }
        )
        assert cfg.voice.wake_model == "alexa"
        assert cfg.voice.whisper_model == "tiny.en"
        assert cfg.voice.silence_duration == 2.0
        assert cfg.voice.listen_timeout == 60.0
