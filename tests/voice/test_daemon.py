"""Tests for the voice daemon."""

import asyncio
import builtins
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.voice.daemon import VoiceDaemon
from arc.voice.listener import VoiceEvent, VoiceState
from arc.voice.transcriber import TranscriptionResult


class _AsyncIterator:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _FakeWebSocket(_AsyncIterator):
    def __init__(self, items=None):
        super().__init__(items or [])
        self.closed = False
        self.sent_json: list[dict] = []
        self.close_calls = 0

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)

    async def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class _FakeSession:
    def __init__(self, ws=None, exc: Exception | None = None):
        self._ws = ws
        self._exc = exc
        self.closed = False
        self.connected_url: str | None = None
        self.close_calls = 0

    async def ws_connect(self, url: str):
        self.connected_url = url
        if self._exc is not None:
            raise self._exc
        return self._ws

    async def close(self) -> None:
        self.closed = True
        self.close_calls += 1


async def _iter_events(items):
    for item in items:
        yield item


def _fake_aiohttp(session: _FakeSession):
    return SimpleNamespace(
        ClientSession=lambda: session,
        WSMsgType=SimpleNamespace(TEXT="text", CLOSED="closed", ERROR="error"),
    )


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


@pytest.mark.asyncio
class TestDaemonLifecycle:

    async def test_run_initializes_connects_and_cleans_up(self, monkeypatch):
        statuses: list[tuple[VoiceState, str]] = []
        ws = _FakeWebSocket([])
        session = _FakeSession(ws=ws)
        daemon = VoiceDaemon(status_callback=lambda state, event: statuses.append((state, event)))
        daemon._transcriber = SimpleNamespace(initialize=AsyncMock())
        daemon._listener = SimpleNamespace(
            initialize=AsyncMock(),
            run=lambda: _iter_events([]),
            stop=AsyncMock(),
            state=VoiceState.SLEEPING,
            notify_response_done=AsyncMock(),
        )
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(session))

        await daemon.run()

        daemon._transcriber.initialize.assert_called_once()
        daemon._listener.initialize.assert_called_once()
        daemon._listener.stop.assert_called_once()
        assert session.connected_url == daemon._gateway_url
        assert ws.closed is True
        assert session.closed is True
        assert statuses[0] == (VoiceState.SLEEPING, "sleep")

    async def test_run_raises_connection_error_when_gateway_is_unavailable(self, monkeypatch):
        session = _FakeSession(exc=RuntimeError("boom"))
        daemon = VoiceDaemon()
        daemon._transcriber = SimpleNamespace(initialize=AsyncMock())
        daemon._listener = SimpleNamespace(initialize=AsyncMock(), stop=AsyncMock())
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(session))

        with pytest.raises(ConnectionError, match="Cannot connect to Arc Gateway"):
            await daemon.run()

        assert session.closed is True

    async def test_stop_shuts_down_listener_and_emits_sleep(self):
        statuses: list[tuple[VoiceState, str]] = []
        daemon = VoiceDaemon(status_callback=lambda state, event: statuses.append((state, event)))
        daemon._running = True
        daemon._listener = SimpleNamespace(stop=AsyncMock())
        daemon._sleep_watch_task = asyncio.create_task(asyncio.sleep(10))

        await daemon.stop()
        await asyncio.sleep(0)

        daemon._listener.stop.assert_called_once()
        assert daemon._running is False
        assert daemon.current_state == VoiceState.SLEEPING
        assert statuses[-1] == (VoiceState.SLEEPING, "sleep")


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

    @pytest.mark.asyncio
    async def test_response_loop_handles_done_messages_and_closed_socket(self, monkeypatch):
        daemon = VoiceDaemon()
        daemon._running = True
        daemon._listener = SimpleNamespace(notify_response_done=AsyncMock(), state=VoiceState.LISTENING)
        daemon._ws = _FakeWebSocket([
            SimpleNamespace(type="text", data=json.dumps({"type": "done", "full_content": "Completed"})),
            SimpleNamespace(type="closed", data=""),
        ])
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(_FakeSession()))

        with patch.object(daemon, "_notify") as mock_notify, patch.object(daemon, "_start_sleep_watch") as mock_sleep_watch:
            await daemon._response_loop()

        mock_notify.assert_any_call("Completed")
        daemon._listener.notify_response_done.assert_called_once()
        mock_sleep_watch.assert_called_once()
        assert daemon._running is False
        assert daemon.current_state == VoiceState.SLEEPING

    @pytest.mark.asyncio
    async def test_response_loop_handles_error_messages(self, monkeypatch):
        daemon = VoiceDaemon()
        daemon._listener = SimpleNamespace(notify_response_done=AsyncMock(), state=VoiceState.LISTENING)
        daemon._ws = _FakeWebSocket([
            SimpleNamespace(type="text", data=json.dumps({"type": "error", "message": "Gateway failed"})),
        ])
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(_FakeSession()))

        with patch.object(daemon, "_notify") as mock_notify, patch.object(daemon, "_start_sleep_watch") as mock_sleep_watch:
            await daemon._response_loop()

        mock_notify.assert_called_once_with("Error: Gateway failed")
        daemon._listener.notify_response_done.assert_called_once()
        mock_sleep_watch.assert_called_once()
        assert daemon.current_state == VoiceState.LISTENING

    @pytest.mark.asyncio
    async def test_response_loop_swallows_cancellation(self, monkeypatch):
        daemon = VoiceDaemon()
        daemon._ws = _FakeWebSocket()
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(_FakeSession()))

        async def endless_messages():
            while True:
                await asyncio.sleep(10)
                yield None

        daemon._ws = endless_messages()
        task = asyncio.create_task(daemon._response_loop())
        await asyncio.sleep(0)
        task.cancel()

        await task


# ── Notification ─────────────────────────────────────────────────


class TestNotification:

    def test_notify_calls_plyer_and_truncates_long_text(self, monkeypatch):
        calls: list[dict[str, object]] = []
        notification = SimpleNamespace(notify=lambda **kwargs: calls.append(kwargs))
        monkeypatch.setitem(sys.modules, "plyer", SimpleNamespace(notification=notification))

        VoiceDaemon._notify("x" * 400)

        assert calls[0]["title"] == "Arc"
        assert calls[0]["message"] == ("x" * 300) + "…"
        assert calls[0]["timeout"] == 10

    def test_notify_with_plyer(self):
        with patch("arc.voice.daemon.logger"):
            # _notify doesn't raise even if plyer is missing
            VoiceDaemon._notify("test response")

    def test_notify_truncates_long_text(self):
        """Long responses are truncated for the notification."""
        # Just verify it doesn't crash — plyer may not be installed
        VoiceDaemon._notify("x" * 500)

    def test_notify_suppresses_import_error(self, monkeypatch):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "plyer":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "plyer", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        VoiceDaemon._notify("no desktop notifier")

    def test_notify_logs_debug_when_notifier_fails(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("notify failed")

        monkeypatch.setitem(sys.modules, "plyer", SimpleNamespace(notification=SimpleNamespace(notify=boom)))

        with patch("arc.voice.daemon.logger") as mock_logger:
            VoiceDaemon._notify("test response")

        mock_logger.debug.assert_called_once()


# ── Chime ────────────────────────────────────────────────────────


class TestChime:

    def test_chime_does_not_raise(self):
        """Chime is non-critical — should never raise even if sounddevice is missing."""
        VoiceDaemon._play_chime()


# ── TTS integration ─────────────────────────────────────────────


class TestDaemonTTS:

    def test_tts_params_stored(self):
        daemon = VoiceDaemon(
            tts_provider="kokoro",
            tts_voice="bf_emma",
            tts_speed=1.3,
        )
        assert daemon._tts_provider == "kokoro"
        assert daemon._tts_voice == "bf_emma"
        assert daemon._tts_speed == 1.3
        assert daemon._tts_available is False

    def test_tts_defaults(self):
        daemon = VoiceDaemon()
        assert daemon._tts_provider == "auto"
        assert daemon._tts_voice == "af_heart"
        assert daemon._tts_speed == 1.0

    def test_injected_synthesizer(self):
        mock_synth = AsyncMock()
        daemon = VoiceDaemon(synthesizer=mock_synth)
        assert daemon._synthesizer is mock_synth

    @pytest.mark.asyncio
    async def test_init_tts_success(self):
        mock_synth = AsyncMock()
        mock_synth.get_info.return_value = {"provider": "mock"}
        daemon = VoiceDaemon(synthesizer=mock_synth)

        await daemon._init_tts()

        mock_synth.initialize.assert_called_once()
        assert daemon._tts_available is True

    @pytest.mark.asyncio
    async def test_init_tts_failure_non_fatal(self):
        mock_synth = AsyncMock()
        mock_synth.initialize.side_effect = RuntimeError("no engine")
        daemon = VoiceDaemon(synthesizer=mock_synth)

        await daemon._init_tts()

        assert daemon._tts_available is False

    @pytest.mark.asyncio
    async def test_speak_response_with_spoken_tag(self):
        mock_synth = AsyncMock()
        mock_synth.speak.return_value = None
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._tts_available = True

        text = "Full explanation here.\n[spoken]Short summary.[/spoken]"
        await daemon._speak_response(text)

        mock_synth.speak.assert_called_once_with("Short summary.")

    @pytest.mark.asyncio
    async def test_speak_response_short_text_verbatim(self):
        mock_synth = AsyncMock()
        mock_synth.speak.return_value = None
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._tts_available = True

        await daemon._speak_response("The answer is 42.")

        # Should speak the text (cleaned for speech)
        call_text = mock_synth.speak.call_args[0][0]
        assert "42" in call_text

    @pytest.mark.asyncio
    async def test_speak_response_long_text_fallback(self):
        mock_synth = AsyncMock()
        mock_synth.speak.return_value = None
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._tts_available = True

        await daemon._speak_response("x " * 200)

        call_text = mock_synth.speak.call_args[0][0]
        assert "response is ready" in call_text.lower()

    @pytest.mark.asyncio
    async def test_speak_response_empty_skipped(self):
        mock_synth = AsyncMock()
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._tts_available = True

        await daemon._speak_response("")

        mock_synth.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_response_tts_unavailable_skipped(self):
        mock_synth = AsyncMock()
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._tts_available = False

        await daemon._speak_response("Hello")

        mock_synth.speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_response_error_nonfatal(self):
        mock_synth = AsyncMock()
        mock_synth.speak.side_effect = RuntimeError("audio device busy")
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._tts_available = True

        # Should not raise
        await daemon._speak_response("Hello world")

    @pytest.mark.asyncio
    async def test_response_loop_speaks_on_done(self, monkeypatch):
        """The response loop should call _speak_response when agent is done."""
        mock_synth = AsyncMock()
        mock_synth.speak.return_value = None
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._running = True
        daemon._tts_available = True
        daemon._listener = SimpleNamespace(
            notify_response_done=AsyncMock(),
            state=VoiceState.LISTENING,
        )
        daemon._ws = _FakeWebSocket([
            SimpleNamespace(
                type="text",
                data=json.dumps({
                    "type": "done",
                    "full_content": "Result here.\n[spoken]All done.[/spoken]",
                }),
            ),
            SimpleNamespace(type="closed", data=""),
        ])
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(_FakeSession()))

        with patch.object(daemon, "_notify"):
            with patch.object(daemon, "_start_sleep_watch"):
                await daemon._response_loop()

        mock_synth.speak.assert_called_once_with("All done.")

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_synthesizer(self, monkeypatch):
        mock_synth = AsyncMock()
        ws = _FakeWebSocket([])
        session = _FakeSession(ws=ws)
        daemon = VoiceDaemon(synthesizer=mock_synth)
        daemon._transcriber = SimpleNamespace(initialize=AsyncMock())
        daemon._listener = SimpleNamespace(
            initialize=AsyncMock(),
            run=lambda: _iter_events([]),
            stop=AsyncMock(),
            state=VoiceState.SLEEPING,
            notify_response_done=AsyncMock(),
        )
        daemon._tts_available = True
        monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(session))

        await daemon.run()

        mock_synth.shutdown.assert_called_once()

    def test_chime_suppresses_errors(self):
        """If sounddevice isn't available, chime is silently skipped."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            # Should not raise
            VoiceDaemon._play_chime()


@pytest.mark.asyncio
class TestSleepWatch:

    async def test_start_sleep_watch_ignores_inactive_daemon(self):
        daemon = VoiceDaemon()
        daemon._running = False

        daemon._start_sleep_watch()

        assert daemon._sleep_watch_task is None

    async def test_start_sleep_watch_emits_sleep_when_listener_returns_to_sleeping(self, monkeypatch):
        events: list[tuple[VoiceState, str]] = []
        daemon = VoiceDaemon(status_callback=lambda state, event: events.append((state, event)))
        daemon._running = True
        daemon._listener = SimpleNamespace(state=VoiceState.ACTIVE)

        async def fake_sleep(_delay):
            daemon._listener.state = VoiceState.SLEEPING

        monkeypatch.setattr("arc.voice.daemon.asyncio.sleep", fake_sleep)

        daemon._start_sleep_watch()
        await daemon._sleep_watch_task

        assert events[-1] == (VoiceState.SLEEPING, "sleep")
        assert daemon.current_state == VoiceState.SLEEPING

    async def test_cancel_sleep_watch_clears_pending_task(self):
        daemon = VoiceDaemon()
        daemon._sleep_watch_task = asyncio.create_task(asyncio.sleep(10))

        daemon._cancel_sleep_watch()
        await asyncio.sleep(0)

        assert daemon._sleep_watch_task is None


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
