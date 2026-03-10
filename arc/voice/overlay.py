"""
Ambient Edge Glow — system-wide overlay for voice state feedback.

A thin glowing bar at the top of the screen that shows Arc's
voice state. Always on top, no taskbar entry — the user never
has to interact with it.

States → colours:
    SLEEPING    → hidden
    ACTIVE      → pulsing cyan/blue  (listening to speech)
    PROCESSING  → pulsing purple/orange  (thinking)
    LISTENING   → gentle green pulse  (waiting for follow-up)

Requires PyQt6.  If not installed, ``create_overlay()`` returns None
and ``arc listen`` falls back to the terminal indicator.

Thread safety:
    The overlay runs on the Qt main thread.  Background threads
    (audio, daemon) update it via ``OverlayBridge.state_changed``
    signal — never by calling widget methods directly.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Guard: everything below needs PyQt6
try:
    from PyQt6.QtCore import (
        QPropertyAnimation,
        QSequentialAnimationGroup,
        QTimer,
        Qt,
        pyqtProperty,
        pyqtSignal,
        QObject,
        QEasingCurve,
    )
    from PyQt6.QtGui import QColor, QScreen, QLinearGradient, QPainter
    from PyQt6.QtWidgets import QApplication, QWidget

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


def is_available() -> bool:
    """Check if PyQt6 is installed."""
    return _HAS_QT


# ── State → visual mapping ────────────────────────────────────────

_STATE_STYLES: dict[str, dict[str, Any]] = {
    "sleeping": {
        "visible": False,
    },
    "active": {
        "visible": True,
        "color1": "#00d4ff",  # cyan
        "color2": "#0066ff",  # blue
        "pulse": True,
        "pulse_speed": 800,   # ms per cycle
    },
    "processing": {
        "visible": True,
        "color1": "#a855f7",  # purple
        "color2": "#ff6b2b",  # orange
        "pulse": True,
        "pulse_speed": 1200,
    },
    "listening": {
        "visible": True,
        "color1": "#22c55e",  # green
        "color2": "#86efac",  # light green
        "pulse": True,
        "pulse_speed": 1500,
    },
}

# Height of the bar in pixels
_BAR_HEIGHT = 8


if _HAS_QT:

    class OverlayBridge(QObject):
        """
        Thread-safe bridge: background thread emits signal,
        Qt main thread receives it and updates the widget.

        Usage from any thread::

            bridge.state_changed.emit("active")
        """

        state_changed = pyqtSignal(str)

    class GlowBar(QWidget):
        """
        Glow bar at the top edge of the screen.

        Uses QPainter to render the gradient directly — this ensures
        real pixels are composited on Windows.  CSS stylesheets with
        WA_TranslucentBackground are invisible on some Windows configs.
        """

        def __init__(self, screen: QScreen | None = None) -> None:
            super().__init__()

            # Window flags: frameless, always on top, no taskbar
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )

            # Size and position — full width, top edge of screen
            target_screen = screen or QApplication.primaryScreen()
            avail = target_screen.availableGeometry()
            self.setGeometry(avail.x(), avail.y(),
                             avail.width(), _BAR_HEIGHT)

            # Gradient colors for painting
            self._color1 = QColor("#00d4ff")
            self._color2 = QColor("#0066ff")

            # Internal opacity for animation
            self._opacity = 1.0
            self._current_state = "sleeping"

            # Pulse animation: fade down then fade up, looped
            self._anim_group = QSequentialAnimationGroup(self)

            fade_out = QPropertyAnimation(self, b"bar_opacity")
            fade_out.setEasingCurve(QEasingCurve.Type.InOutSine)

            fade_in = QPropertyAnimation(self, b"bar_opacity")
            fade_in.setEasingCurve(QEasingCurve.Type.InOutSine)

            self._fade_out = fade_out
            self._fade_in = fade_in
            self._anim_group.addAnimation(fade_out)
            self._anim_group.addAnimation(fade_in)
            self._anim_group.setLoopCount(-1)

            # Start hidden
            self.hide()

            # Startup flash — brief green glow to confirm overlay is active
            QTimer.singleShot(200, self._startup_flash)

        def _startup_flash(self) -> None:
            """Show a brief green flash on startup, then fade to hidden."""
            self._color1 = QColor("#22c55e")
            self._color2 = QColor("#86efac")
            self.setWindowOpacity(1.0)
            self.show()
            self.update()

            flash_out = QPropertyAnimation(self, b"bar_opacity")
            flash_out.setDuration(800)
            flash_out.setStartValue(1.0)
            flash_out.setEndValue(0.0)
            flash_out.setEasingCurve(QEasingCurve.Type.OutQuad)
            flash_out.finished.connect(self.hide)
            self._flash_anim = flash_out
            flash_out.start()

        # ── Paint the gradient ────────────────────────────────

        def paintEvent(self, event: Any) -> None:
            """Paint the gradient bar directly."""
            painter = QPainter(self)
            gradient = QLinearGradient(0, 0, self.width(), 0)
            gradient.setColorAt(0.0, self._color1)
            gradient.setColorAt(1.0, self._color2)
            painter.fillRect(self.rect(), gradient)
            painter.end()

        # ── Click-through ─────────────────────────────────────

        def mousePressEvent(self, event: Any) -> None:
            event.ignore()

        def mouseReleaseEvent(self, event: Any) -> None:
            event.ignore()

        def mouseMoveEvent(self, event: Any) -> None:
            event.ignore()

        # ── Animated property ─────────────────────────────────

        def get_bar_opacity(self) -> float:
            return self._opacity

        def set_bar_opacity(self, value: float) -> None:
            self._opacity = value
            self.setWindowOpacity(max(0.0, min(1.0, value)))

        bar_opacity = pyqtProperty(float, get_bar_opacity, set_bar_opacity)

        # ── State control ─────────────────────────────────────

        def set_voice_state(self, state: str) -> None:
            """
            Update the bar's appearance for a voice state.

            Called on the Qt main thread via OverlayBridge signal.
            """
            state = state.lower()
            if state == self._current_state:
                return
            self._current_state = state

            style = _STATE_STYLES.get(state, _STATE_STYLES["sleeping"])

            # Stop any running animation
            self._anim_group.stop()

            if not style.get("visible", False):
                self.hide()
                return

            # Set gradient colors and repaint
            self._color1 = QColor(style.get("color1", "#00d4ff"))
            self._color2 = QColor(style.get("color2", "#0066ff"))

            self.setWindowOpacity(1.0)
            self._opacity = 1.0
            self.show()
            self.update()  # trigger repaint with new colors

            # Pulse animation
            if style.get("pulse", False):
                half_speed = style.get("pulse_speed", 1000) // 2
                self._fade_out.setDuration(half_speed)
                self._fade_out.setStartValue(1.0)
                self._fade_out.setEndValue(0.35)
                self._fade_in.setDuration(half_speed)
                self._fade_in.setStartValue(0.35)
                self._fade_in.setEndValue(1.0)
                self._anim_group.start()


def create_overlay() -> tuple[Any, Any] | None:
    """
    Create the overlay bar and signal bridge.

    Returns ``(glow_bar, bridge)`` if PyQt6 is available,
    or ``None`` if it isn't.

    The caller must:
    1. Create ``QApplication`` before calling this.
    2. Connect ``bridge.state_changed`` to ``glow_bar.set_voice_state``.
    3. Run ``app.exec()`` on the main thread.
    """
    if not _HAS_QT:
        return None

    bridge = OverlayBridge()
    bar = GlowBar()
    bridge.state_changed.connect(bar.set_voice_state)

    return bar, bridge
