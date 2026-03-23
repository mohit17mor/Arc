"""Tests for the ambient edge glow overlay."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Force a headless-safe Qt backend for tests before importing PyQt6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── Availability check ───────────────────────────────────────────


class TestAvailability:

    def test_is_available_returns_bool(self):
        from arc.voice.overlay import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_create_overlay_without_qt_returns_none(self):
        """When PyQt6 is not installed, create_overlay returns None."""
        with patch.dict("sys.modules", {"PyQt6": None, "PyQt6.QtCore": None,
                                         "PyQt6.QtGui": None, "PyQt6.QtWidgets": None}):
            # Re-import to pick up the mocked modules
            import importlib
            import arc.voice.overlay as overlay_mod
            # _HAS_QT was set at import time; test create_overlay directly
            original = overlay_mod._HAS_QT
            overlay_mod._HAS_QT = False
            try:
                result = overlay_mod.create_overlay()
                assert result is None
                assert overlay_mod.is_available() is False
            finally:
                overlay_mod._HAS_QT = original

    def test_should_raise_macos_window_level_skips_offscreen(self):
        import arc.voice.overlay as overlay_mod

        with patch("platform.system", return_value="Darwin"):
            with patch.object(overlay_mod.QApplication, "platformName", return_value="offscreen"):
                assert overlay_mod._should_raise_macos_window_level() is False


# ── State styles ─────────────────────────────────────────────────


class TestStateStyles:

    def test_all_states_have_styles(self):
        from arc.voice.overlay import _STATE_STYLES
        for state in ("sleeping", "active", "processing", "listening"):
            assert state in _STATE_STYLES

    def test_sleeping_is_hidden(self):
        from arc.voice.overlay import _STATE_STYLES
        assert _STATE_STYLES["sleeping"]["visible"] is False

    def test_active_is_visible_with_pulse(self):
        from arc.voice.overlay import _STATE_STYLES
        style = _STATE_STYLES["active"]
        assert style["visible"] is True
        assert style["pulse"] is True
        assert "color1" in style
        assert "color2" in style

    def test_processing_has_different_colors(self):
        from arc.voice.overlay import _STATE_STYLES
        active = _STATE_STYLES["active"]
        processing = _STATE_STYLES["processing"]
        assert active["color1"] != processing["color1"]

    def test_listening_is_green(self):
        from arc.voice.overlay import _STATE_STYLES
        style = _STATE_STYLES["listening"]
        assert "#22c55e" in style["color1"]  # green


# ── Bar height constant ──────────────────────────────────────────


class TestConstants:

    def test_bar_height(self):
        from arc.voice.overlay import _BAR_HEIGHT
        assert _BAR_HEIGHT > 0
        assert _BAR_HEIGHT <= 20  # reasonable range


# ── Qt widget tests (only when PyQt6 is available) ───────────────

_qt_available = False
try:
    from PyQt6.QtWidgets import QApplication
    _qt_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _qt_available, reason="PyQt6 not installed")
class TestGlowBarWidget:

    @pytest.fixture(autouse=True)
    def qt_app(self):
        """Ensure a QApplication exists for widget tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_create_overlay_returns_tuple(self):
        from arc.voice.overlay import create_overlay
        result = create_overlay()
        assert result is not None
        bar, bridge = result
        assert bar is not None
        assert bridge is not None

    def test_bar_starts_hidden(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        assert not bar.isVisible()

    def test_set_active_shows_bar(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bar.set_voice_state("active")
        assert bar.isVisible()

    def test_set_sleeping_hides_bar(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bar.set_voice_state("active")
        assert bar.isVisible()
        bar.set_voice_state("sleeping")
        assert not bar.isVisible()

    def test_set_processing_shows_bar(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bar.set_voice_state("processing")
        assert bar.isVisible()

    def test_set_listening_shows_bar(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bar.set_voice_state("listening")
        assert bar.isVisible()

    def test_signal_bridge_updates_bar(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bridge.state_changed.emit("active")
        assert bar.isVisible()
        bridge.state_changed.emit("sleeping")
        assert not bar.isVisible()

    def test_same_state_is_noop(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bar.set_voice_state("active")
        # Calling again with same state should not crash
        bar.set_voice_state("active")
        assert bar.isVisible()

    def test_unknown_state_hides(self):
        from arc.voice.overlay import create_overlay
        bar, bridge = create_overlay()
        bar.set_voice_state("active")
        bar.set_voice_state("unknown_state")
        assert not bar.isVisible()

    def test_bar_has_correct_flags(self):
        from PyQt6.QtCore import Qt
        from arc.voice.overlay import create_overlay
        bar, _ = create_overlay()
        flags = bar.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint
        assert flags & Qt.WindowType.WindowStaysOnTopHint
        assert flags & Qt.WindowType.Tool
