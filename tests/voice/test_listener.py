"""Tests for the voice listener state machine."""

import asyncio

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.voice.listener import (
    CHUNK_SIZE,
    SAMPLE_RATE,
    VoiceEvent,
    VoiceListener,
    VoiceState,
)


# ── VoiceState enum ─────────────────────────────────────────────


class TestVoiceState:

    def test_all_states_exist(self):
        assert VoiceState.SLEEPING == "sleeping"
        assert VoiceState.ACTIVE == "active"
        assert VoiceState.PROCESSING == "processing"
        assert VoiceState.LISTENING == "listening"


# ── VoiceEvent ───────────────────────────────────────────────────


class TestVoiceEvent:

    def test_creation(self):
        e = VoiceEvent(type="wake_word", state=VoiceState.ACTIVE)
        assert e.type == "wake_word"
        assert e.state == VoiceState.ACTIVE
        assert e.data == {}

    def test_with_data(self):
        audio = np.zeros(1000, dtype=np.float32)
        e = VoiceEvent(
            type="speech_ready",
            state=VoiceState.PROCESSING,
            data={"audio": audio},
        )
        assert e.data["audio"] is audio


# ── VoiceListener construction ───────────────────────────────────


class TestListenerConstruction:

    def test_default_config(self):
        listener = VoiceListener()
        assert listener.state == VoiceState.SLEEPING
        assert listener._wake_model == "hey_jarvis"
        assert listener._wake_threshold == 0.5
        assert listener._silence_threshold == 0.01
        assert listener._silence_duration == 1.5
        assert listener._listen_timeout == 30.0

    def test_custom_config(self):
        listener = VoiceListener(
            wake_model="alexa",
            wake_threshold=0.7,
            silence_duration=2.0,
            listen_timeout=60.0,
        )
        assert listener._wake_model == "alexa"
        assert listener._wake_threshold == 0.7
        assert listener._silence_duration == 2.0
        assert listener._listen_timeout == 60.0

    @pytest.mark.asyncio
    async def test_initialize_runs_wake_word_load_in_executor(self):
        listener = VoiceListener()

        class FakeLoop:
            def __init__(self):
                self.calls = []

            async def run_in_executor(self, executor, func):
                self.calls.append((executor, func))
                return func()

        fake_loop = FakeLoop()
        listener._init_wake_word = MagicMock()

        with patch("arc.voice.listener.asyncio.get_running_loop", return_value=fake_loop):
            await listener.initialize()

        listener._init_wake_word.assert_called_once()
        assert fake_loop.calls


# ── Silence detection ────────────────────────────────────────────


class TestSilenceDetection:

    def test_silence_detected_after_duration(self):
        listener = VoiceListener(silence_threshold=0.01, silence_duration=0.5)
        # Feed enough silent chunks to exceed 0.5s at 16kHz
        silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        # 0.5s = 8000 samples, CHUNK_SIZE = 1280, need ~7 chunks
        for _ in range(6):
            assert listener._detect_silence(silent_chunk) is False
        assert listener._detect_silence(silent_chunk) is True

    def test_loud_resets_counter(self):
        listener = VoiceListener(silence_threshold=0.01, silence_duration=0.5)
        silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        loud_chunk = np.ones(CHUNK_SIZE, dtype=np.float32) * 0.5

        # Feed some silence
        for _ in range(5):
            listener._detect_silence(silent_chunk)

        # A loud chunk resets the counter
        assert listener._detect_silence(loud_chunk) is False

        # Need full duration of silence again
        for _ in range(6):
            assert listener._detect_silence(silent_chunk) is False
        assert listener._detect_silence(silent_chunk) is True


# ── Wake word detection (mocked) ─────────────────────────────────


class TestWakeWordDetection:

    def test_detected_when_above_threshold(self):
        listener = VoiceListener(wake_threshold=0.5)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.9}
        listener._oww_model = mock_model

        chunk = np.random.randn(CHUNK_SIZE).astype(np.float32)
        assert listener._detect_wake_word(chunk) is True

    def test_not_detected_below_threshold(self):
        listener = VoiceListener(wake_threshold=0.5)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.2}
        listener._oww_model = mock_model

        chunk = np.random.randn(CHUNK_SIZE).astype(np.float32)
        assert listener._detect_wake_word(chunk) is False

    def test_no_model_returns_false(self):
        listener = VoiceListener()
        listener._oww_model = None
        chunk = np.random.randn(CHUNK_SIZE).astype(np.float32)
        assert listener._detect_wake_word(chunk) is False

    def test_converts_float32_to_int16(self):
        listener = VoiceListener(wake_threshold=0.5)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.9}
        listener._oww_model = mock_model

        chunk = np.array([0.5, -0.5], dtype=np.float32)
        listener._detect_wake_word(chunk)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args.dtype == np.int16


# ── State transitions (integrated, mocked audio) ────────────────


class TestStateTransitions:

    @pytest.mark.asyncio
    async def test_sleeping_to_active_on_wake_word(self):
        """Wake word transitions SLEEPING → ACTIVE."""
        listener = VoiceListener(wake_threshold=0.5)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.9}
        mock_model.reset = MagicMock()
        listener._oww_model = mock_model

        # Simulate: start_mic is mocked, we push audio manually
        listener._start_mic = MagicMock()
        listener._stop_mic = MagicMock()
        listener._loop = asyncio.get_running_loop()
        listener._running = True

        # Push a chunk and then stop
        chunk = np.random.randn(CHUNK_SIZE).astype(np.float32)
        await listener._audio_queue.put(chunk)

        events = []
        async for event in listener.run():
            events.append(event)
            if event.type == "wake_word":
                listener._running = False
                break

        assert any(e.type == "wake_word" for e in events)
        assert listener.state == VoiceState.ACTIVE

    @pytest.mark.asyncio
    async def test_active_to_processing_on_silence(self):
        """Prolonged silence during ACTIVE → PROCESSING with audio."""
        listener = VoiceListener(
            wake_threshold=0.5,
            silence_threshold=0.01,
            silence_duration=0.1,  # short for test
        )

        # Mock wake word (always detected for first chunk)
        mock_oww = MagicMock()
        call_count = 0

        def predict_side_effect(audio):
            nonlocal call_count
            call_count += 1
            return {"hey_jarvis": 0.9 if call_count == 1 else 0.0}

        mock_oww.predict = predict_side_effect
        mock_oww.reset = MagicMock()
        listener._oww_model = mock_oww
        listener._start_mic = MagicMock()
        listener._stop_mic = MagicMock()
        listener._pause_mic = MagicMock()
        listener._loop = asyncio.get_running_loop()
        listener._running = True

        # Push wake word chunk, then enough silent chunks
        wake_chunk = np.random.randn(CHUNK_SIZE).astype(np.float32) * 0.5
        await listener._audio_queue.put(wake_chunk)

        # Silence chunks (1600 samples = 0.1s at 16kHz, need >=0.1s)
        silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        for _ in range(3):
            await listener._audio_queue.put(silent_chunk)

        events = []
        async for event in listener.run():
            events.append(event)
            if event.type == "speech_ready":
                listener._running = False
                break

        assert any(e.type == "speech_ready" for e in events)
        speech_event = [e for e in events if e.type == "speech_ready"][0]
        assert "audio" in speech_event.data
        assert listener.state == VoiceState.PROCESSING

    @pytest.mark.asyncio
    async def test_notify_response_done_transitions_to_listening(self):
        """notify_response_done() → LISTENING with mic resumed."""
        listener = VoiceListener(listen_timeout=1.0)
        listener._state = VoiceState.PROCESSING
        listener._stream = MagicMock()
        listener._stream.active = False  # mic is paused

        await listener.notify_response_done()

        assert listener.state == VoiceState.LISTENING
        assert listener._listen_timeout_task is not None
        # Clean up
        listener._cancel_listen_timeout()

    @pytest.mark.asyncio
    async def test_listen_timeout_returns_to_sleeping(self):
        """After listen_timeout, LISTENING → SLEEPING."""
        listener = VoiceListener(listen_timeout=0.1)  # 100ms
        listener._state = VoiceState.LISTENING
        listener._stream = MagicMock()
        listener._stream.active = False

        await listener.notify_response_done()
        await asyncio.sleep(0.2)  # wait for timeout

        assert listener.state == VoiceState.SLEEPING

    @pytest.mark.asyncio
    async def test_listening_follow_up_transitions_back_to_active(self):
        listener = VoiceListener(silence_threshold=0.01)
        listener._start_mic = MagicMock()
        listener._stop_mic = MagicMock()
        events = []

        async def consume():
            async for event in listener.run():
                events.append(event)
                if event.type == "state_change" and event.state == VoiceState.SLEEPING:
                    listener._state = VoiceState.LISTENING
                    loud_chunk = np.ones(CHUNK_SIZE, dtype=np.float32) * 0.5
                    await listener._audio_queue.put(loud_chunk)
                elif event.type == "state_change" and event.state == VoiceState.ACTIVE:
                    listener._running = False
                    break

        await asyncio.wait_for(consume(), timeout=2.0)

        assert listener.state == VoiceState.ACTIVE
        assert listener._recording_buffer
        assert np.array_equal(listener._recording_buffer[0], np.ones(CHUNK_SIZE, dtype=np.float32) * 0.5)


class TestWakeModelInit:

    def test_init_wake_word_downloads_models_and_creates_model(self):
        listener = VoiceListener(wake_model="hey_arc")
        fake_utils = MagicMock()
        fake_openwakeword = MagicMock(utils=fake_utils)
        fake_model_cls = MagicMock(return_value="model-instance")

        with patch.dict(
            "sys.modules",
            {
                "openwakeword": fake_openwakeword,
                "openwakeword.model": MagicMock(Model=fake_model_cls),
            },
        ):
            listener._init_wake_word()

        fake_utils.download_models.assert_called_once()
        fake_model_cls.assert_called_once_with(
            wakeword_models=["hey_arc"],
            inference_framework="onnx",
        )
        assert listener._oww_model == "model-instance"


class TestMicrophoneHelpers:

    def test_pause_mic_stops_stream_and_drains_queue(self):
        listener = VoiceListener()
        listener._stream = MagicMock()
        listener._stream.active = True
        listener._audio_queue.put_nowait(np.ones(CHUNK_SIZE, dtype=np.float32))

        listener._pause_mic()

        listener._stream.stop.assert_called_once()
        assert listener._audio_queue.empty()

    def test_resume_mic_restarts_inactive_stream(self):
        listener = VoiceListener()
        listener._stream = MagicMock()
        listener._stream.active = False

        listener._resume_mic()

        listener._stream.start.assert_called_once()

    def test_stop_mic_closes_stream_and_clears_reference(self):
        listener = VoiceListener()
        stream = MagicMock()
        listener._stream = stream

        listener._stop_mic()

        stream.close.assert_called_once()
        assert listener._stream is None

    def test_audio_callback_logs_status_and_schedules_chunk(self):
        listener = VoiceListener()
        captured = []

        class FakeLoop:
            def call_soon_threadsafe(self, func, audio):
                captured.append(audio)

        listener._loop = FakeLoop()
        indata = np.arange(CHUNK_SIZE, dtype=np.float32).reshape(-1, 1)

        with patch("arc.voice.listener.logger.debug") as debug:
            listener._audio_callback(indata, CHUNK_SIZE, None, status="overflow")

        debug.assert_called_once()
        assert len(captured) == 1
        assert np.array_equal(captured[0], indata[:, 0])

    @pytest.mark.asyncio
    async def test_stop_handles_queue_put_failure(self):
        listener = VoiceListener()
        listener._running = True
        listener._audio_queue = MagicMock()
        listener._audio_queue.put_nowait.side_effect = RuntimeError("queue closed")
        listener._stop_mic = MagicMock()
        listener._cancel_listen_timeout = MagicMock()

        await listener.stop()

        assert listener._running is False
        listener._stop_mic.assert_called_once()
        listener._cancel_listen_timeout.assert_called_once()


# ── Audio constants ──────────────────────────────────────────────


class TestAudioConstants:

    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_chunk_size_is_80ms(self):
        # 80ms at 16kHz = 1280 samples
        assert CHUNK_SIZE == int(SAMPLE_RATE * 0.08)
