"""
Speech-to-text provider — converts audio to text.

Abstracted behind a provider interface (same pattern as LLMProvider,
EmbeddingProvider) so the STT backend can be swapped without
touching the voice daemon or listener.

Default implementation: WhisperLocalProvider
    - Uses faster-whisper + CTranslate2 (runs offline, no GPU needed)
    - Model: base.en (~150MB, ~95% accuracy for clear English)
    - Downloaded once on first use to ~/.cache/huggingface
    - Runs in a thread pool to avoid blocking the event loop
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from a speech-to-text transcription."""

    text: str
    confidence: float  # 0.0–1.0 derived from average log probability
    language: str
    duration_ms: int  # wall-clock time taken to transcribe


class SpeechProvider(ABC):
    """
    Abstract interface for speech-to-text providers.

    Follows the same provider pattern as LLMProvider and
    EmbeddingProvider — initialize once, call many times.
    """

    async def initialize(self) -> None:
        """Load or warm up the model. Override if startup work is needed."""

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe an audio segment to text.

        Args:
            audio: float32 numpy array, mono, values in [-1.0, 1.0]
            sample_rate: sample rate of the audio (default 16kHz)

        Returns:
            TranscriptionResult with text and confidence.
        """
        ...

    def get_info(self) -> dict[str, Any]:
        """Return provider metadata for introspection."""
        return {}


class WhisperLocalProvider(SpeechProvider):
    """
    Local Whisper STT via faster-whisper.

    Runs entirely offline after the first model download.
    CPU-only by default (int8 quantisation for speed).

    Usage::

        provider = WhisperLocalProvider(model_name="base.en")
        await provider.initialize()
        result = await provider.transcribe(audio_array)
        print(result.text, result.confidence)
    """

    def __init__(
        self,
        model_name: str = "base.en",
        compute_type: str = "int8",
    ) -> None:
        self._model_name = model_name
        self._compute_type = compute_type
        self._model: Any = None

    async def initialize(self) -> None:
        """Download (if needed) and load the Whisper model."""
        logger.info(
            f"Loading Whisper model '{self._model_name}' "
            f"(first run downloads ~150MB)..."
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)
        logger.info(f"Whisper model '{self._model_name}' ready")

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_name,
            compute_type=self._compute_type,
            device="cpu",
        )

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError(
                "WhisperLocalProvider not initialized. Call initialize() first."
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, audio, sample_rate
        )

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        start = time.monotonic()

        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=3,
            vad_filter=True,
        )

        texts: list[str] = []
        total_logprob = 0.0
        segment_count = 0
        for segment in segments:
            texts.append(segment.text)
            total_logprob += segment.avg_logprob
            segment_count += 1

        text = " ".join(texts).strip()

        # Convert log probability (negative) to a 0–1 confidence score.
        # avg_logprob of -0.0 → 1.0, -1.0 → 0.0 (clamped).
        avg_logprob = total_logprob / max(segment_count, 1)
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

        duration_ms = int((time.monotonic() - start) * 1000)

        return TranscriptionResult(
            text=text,
            confidence=confidence,
            language=getattr(info, "language", "en"),
            duration_ms=duration_ms,
        )

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": "whisper_local",
            "model": self._model_name,
            "compute_type": self._compute_type,
            "loaded": self._model is not None,
        }
