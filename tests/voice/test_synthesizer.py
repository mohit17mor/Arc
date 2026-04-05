"""Tests for the TTS synthesizer providers."""

import asyncio
import platform
import subprocess

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from arc.voice.synthesizer import (
    KokoroProvider,
    SpeechSynthesizer,
    SynthesisResult,
    SystemTTSProvider,
    _AutoProvider,
    create_synthesizer,
)


# ── SynthesisResult ──────────────────────────────────────────────


class TestSynthesisResult:

    def test_creation(self):
        r = SynthesisResult(duration_ms=500, provider="kokoro", voice="af_heart")
        assert r.duration_ms == 500
        assert r.provider == "kokoro"
        assert r.voice == "af_heart"


# ── KokoroProvider construction ──────────────────────────────────


class TestKokoroConstruction:

    def test_default_config(self):
        p = KokoroProvider()
        info = p.get_info()
        assert info["provider"] == "kokoro"
        assert info["voice"] == "af_heart"
        assert info["lang"] == "en-us"
        assert info["speed"] == 1.0
        assert info["loaded"] is False

    def test_custom_config(self):
        p = KokoroProvider(voice="bf_emma", speed=1.2, lang="en-gb")
        info = p.get_info()
        assert info["voice"] == "bf_emma"
        assert info["speed"] == 1.2
        assert info["lang"] == "en-gb"

    @pytest.mark.asyncio
    async def test_initialize_missing_model_raises(self, tmp_path):
        p = KokoroProvider(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Kokoro model files not found"):
            await p.initialize()

    @pytest.mark.asyncio
    async def test_initialize_missing_package_raises(self, tmp_path):
        # Create dummy model files
        (tmp_path / "kokoro-v1.0.onnx").write_text("fake")
        (tmp_path / "voices-v1.0.bin").write_text("fake")

        p = KokoroProvider(model_dir=tmp_path)
        with patch.dict("sys.modules", {"kokoro_onnx": None}):
            with pytest.raises(ImportError, match="kokoro-onnx is not installed"):
                await p.initialize()


# ── KokoroProvider speak (mocked) ────────────────────────────────


class TestKokoroSpeak:

    @pytest.mark.asyncio
    async def test_speak_not_initialized_raises(self):
        p = KokoroProvider()
        with pytest.raises(RuntimeError, match="not initialized"):
            await p.speak("Hello")

    @pytest.mark.asyncio
    async def test_speak_calls_create_and_plays(self, tmp_path):
        import numpy as np

        p = KokoroProvider(model_dir=tmp_path, voice="af_heart")

        # Mock internal kokoro instance
        mock_kokoro = MagicMock()
        samples = np.zeros(16000, dtype=np.float32)
        mock_kokoro.create.return_value = (samples, 16000)
        p._kokoro = mock_kokoro

        with patch("arc.voice.synthesizer.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.stop = MagicMock()

            # Patch asyncio.sleep to not actually wait
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await p.speak("Test speech")

        assert result.provider == "kokoro"
        assert result.voice == "af_heart"
        assert result.duration_ms >= 0
        mock_kokoro.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_calls_sd_stop(self):
        p = KokoroProvider()
        with patch("arc.voice.synthesizer.sd") as mock_sd:
            mock_sd.stop = MagicMock()
            await p.stop()
            mock_sd.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_clears_model(self):
        p = KokoroProvider()
        p._kokoro = MagicMock()
        await p.shutdown()
        assert p._kokoro is None


# ── SystemTTSProvider ────────────────────────────────────────────


class TestSystemTTSConstruction:

    def test_default_config(self):
        p = SystemTTSProvider()
        info = p.get_info()
        assert info["provider"] == "system"
        assert info["platform"] == platform.system()


class TestSystemTTSEngine:

    @pytest.mark.asyncio
    async def test_initialize_detects_engine(self):
        p = SystemTTSProvider()
        # Patch detection to always find something
        with patch.object(p, "_detect_engine", return_value="sapi5"):
            await p.initialize()
            assert p._engine == "sapi5"

    @pytest.mark.asyncio
    async def test_initialize_no_engine_raises(self):
        p = SystemTTSProvider()
        with patch.object(p, "_detect_engine", return_value=""):
            with pytest.raises(RuntimeError, match="No system TTS engine found"):
                await p.initialize()

    def test_build_command_sapi5(self):
        p = SystemTTSProvider(rate=3)
        p._engine = "sapi5"
        cmd = p._build_command("Hello world")
        assert cmd[0] == "powershell"
        assert "System.Speech" in cmd[-1]
        assert "Hello world" in cmd[-1]
        assert "$s.Rate = 3" in cmd[-1]

    def test_build_command_say(self):
        p = SystemTTSProvider(rate=200)
        p._engine = "say"
        cmd = p._build_command("Test")
        assert cmd[0] == "say"
        assert "-r" in cmd
        assert "200" in cmd
        assert "Test" in cmd

    def test_build_command_espeak(self):
        p = SystemTTSProvider()
        p._engine = "espeak-ng"
        cmd = p._build_command("Test")
        assert cmd[0] == "espeak-ng"
        assert "-s" in cmd
        assert "Test" in cmd

    def test_build_command_sanitises_quotes(self):
        p = SystemTTSProvider()
        p._engine = "sapi5"
        cmd = p._build_command('He said "hello" today')
        # Double quotes should be replaced with single quotes
        assert '"hello"' not in cmd[-1]
        assert "'hello'" in cmd[-1]


class TestSystemTTSSpeak:

    @pytest.mark.asyncio
    async def test_speak_not_initialized_raises(self):
        p = SystemTTSProvider()
        with pytest.raises(RuntimeError, match="not initialized"):
            await p.speak("Hello")

    @pytest.mark.asyncio
    async def test_speak_runs_subprocess(self):
        p = SystemTTSProvider()
        p._engine = "say"

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.poll.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            result = await p.speak("Hello")

        assert result.provider == "system"
        assert result.voice == "say"

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self):
        p = SystemTTSProvider()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        p._process = mock_proc

        await p.stop()

        mock_proc.terminate.assert_called_once()
        assert p._process is None


# ── create_synthesizer factory ───────────────────────────────────


class TestFactory:

    def test_create_kokoro(self):
        synth = create_synthesizer("kokoro")
        assert isinstance(synth, KokoroProvider)

    def test_create_system(self):
        synth = create_synthesizer("system")
        assert isinstance(synth, SystemTTSProvider)

    def test_create_auto(self):
        synth = create_synthesizer("auto")
        assert isinstance(synth, _AutoProvider)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_synthesizer("nonexistent")

    def test_kokoro_params_forwarded(self):
        synth = create_synthesizer(
            "kokoro",
            kokoro_voice="bf_emma",
            kokoro_speed=1.5,
            kokoro_lang="en-gb",
        )
        info = synth.get_info()
        assert info["voice"] == "bf_emma"
        assert info["speed"] == 1.5

    def test_system_rate_forwarded(self):
        synth = create_synthesizer("system", system_rate=250)
        assert synth._rate == 250


# ── _AutoProvider ────────────────────────────────────────────────


class TestAutoProvider:

    @pytest.mark.asyncio
    async def test_auto_falls_back_to_system(self):
        """When Kokoro is unavailable, auto should use system TTS."""
        auto = _AutoProvider()

        # Make Kokoro fail (ImportError) and System succeed
        with patch(
            "arc.voice.synthesizer.KokoroProvider.initialize",
            side_effect=ImportError("no kokoro"),
        ):
            with patch(
                "arc.voice.synthesizer.SystemTTSProvider.initialize",
                new_callable=AsyncMock,
            ):
                with patch(
                    "arc.voice.synthesizer.SystemTTSProvider._detect_engine",
                    return_value="say",
                ):
                    await auto.initialize()

        assert auto._active is not None
        info = auto.get_info()
        assert info["wrapper"] == "auto"

    @pytest.mark.asyncio
    async def test_auto_no_provider_raises(self):
        auto = _AutoProvider()
        with patch(
            "arc.voice.synthesizer.KokoroProvider.initialize",
            side_effect=ImportError("no kokoro"),
        ):
            with patch(
                "arc.voice.synthesizer.SystemTTSProvider.initialize",
                side_effect=RuntimeError("no engine"),
            ):
                with pytest.raises(RuntimeError, match="No TTS provider available"):
                    await auto.initialize()

    @pytest.mark.asyncio
    async def test_auto_speak_delegates(self):
        auto = _AutoProvider()
        mock_synth = AsyncMock(spec=SpeechSynthesizer)
        mock_synth.speak.return_value = SynthesisResult(
            duration_ms=100, provider="mock", voice="test"
        )
        auto._active = mock_synth

        result = await auto.speak("Hello")
        mock_synth.speak.assert_called_once_with("Hello")
        assert result.provider == "mock"

    @pytest.mark.asyncio
    async def test_auto_not_initialized_raises(self):
        auto = _AutoProvider()
        with pytest.raises(RuntimeError, match="not initialized"):
            await auto.speak("Hello")

    def test_auto_info_no_active(self):
        auto = _AutoProvider()
        info = auto.get_info()
        assert info["provider"] == "auto"
        assert info["active"] is None
