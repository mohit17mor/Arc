"""Tests for the speech-to-text provider."""

import asyncio

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.voice.transcriber import (
    SpeechProvider,
    TranscriptionResult,
    WhisperLocalProvider,
)


# ── TranscriptionResult ─────────────────────────────────────────


class TestTranscriptionResult:

    def test_creation(self):
        r = TranscriptionResult(
            text="hello world",
            confidence=0.95,
            language="en",
            duration_ms=150,
        )
        assert r.text == "hello world"
        assert r.confidence == 0.95
        assert r.language == "en"
        assert r.duration_ms == 150

    def test_empty_text(self):
        r = TranscriptionResult(text="", confidence=0.0, language="en", duration_ms=0)
        assert r.text == ""


# ── WhisperLocalProvider construction ────────────────────────────


class TestWhisperLocalConstruction:

    def test_default_config(self):
        p = WhisperLocalProvider()
        info = p.get_info()
        assert info["provider"] == "whisper_local"
        assert info["model"] == "base.en"
        assert info["compute_type"] == "int8"
        assert info["loaded"] is False

    def test_custom_model(self):
        p = WhisperLocalProvider(model_name="tiny.en", compute_type="float32")
        info = p.get_info()
        assert info["model"] == "tiny.en"
        assert info["compute_type"] == "float32"

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self):
        p = WhisperLocalProvider()
        with pytest.raises(RuntimeError, match="not initialized"):
            await p.transcribe(np.zeros(16000, dtype=np.float32))


# ── WhisperLocalProvider transcribe (mocked) ────────────────────


class TestWhisperTranscribe:

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self):
        """Mocked Whisper transcription returns a proper result."""
        p = WhisperLocalProvider()

        # Mock the internal model
        mock_segment = MagicMock()
        mock_segment.text = "hello world"
        mock_segment.avg_logprob = -0.1  # high confidence

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        p._model = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        result = await p.transcribe(audio)

        assert result.text == "hello world"
        assert result.confidence > 0.8
        assert result.language == "en"
        assert result.duration_ms >= 0
        mock_model.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self):
        """Empty audio produces empty text with low confidence."""
        p = WhisperLocalProvider()

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        p._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = await p.transcribe(audio)

        assert result.text == ""
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_transcribe_multiple_segments(self):
        """Multiple segments are joined."""
        p = WhisperLocalProvider()

        seg1 = MagicMock()
        seg1.text = "hello"
        seg1.avg_logprob = -0.2

        seg2 = MagicMock()
        seg2.text = "world"
        seg2.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)
        p._model = mock_model

        audio = np.random.randn(32000).astype(np.float32)
        result = await p.transcribe(audio)

        assert result.text == "hello world"
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_clamping(self):
        """Confidence is clamped to [0, 1]."""
        p = WhisperLocalProvider()

        seg = MagicMock()
        seg.text = "test"
        seg.avg_logprob = -2.0  # very low → 1.0 + (-2.0) = -1.0 → clamped to 0.0

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], mock_info)
        p._model = mock_model

        result = await p.transcribe(np.zeros(16000, dtype=np.float32))

        assert result.confidence == 0.0


# ── SpeechProvider interface ─────────────────────────────────────


class TestSpeechProviderInterface:

    def test_whisper_is_speech_provider(self):
        assert issubclass(WhisperLocalProvider, SpeechProvider)

    @pytest.mark.asyncio
    async def test_default_initialize_is_noop(self):
        """Base class initialize does nothing by default."""

        class DummyProvider(SpeechProvider):
            async def transcribe(self, audio, sample_rate=16000):
                return TranscriptionResult(
                    text="test", confidence=1.0, language="en", duration_ms=0
                )

        p = DummyProvider()
        await p.initialize()  # should not raise
        assert p.get_info() == {}
