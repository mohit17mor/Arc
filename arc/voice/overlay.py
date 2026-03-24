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
import platform
from os import environ
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


def _should_raise_macos_window_level() -> bool:
    """Only use the native Cocoa window level on interactive macOS Qt backends."""
    if not _HAS_QT or platform.system() != "Darwin":
        return False

    platform_name = ""
    app = QApplication.instance()
    if app is not None:
        try:
            platform_name = app.platformName().lower()
        except Exception:
            platform_name = ""

    env_platform = environ.get("QT_QPA_PLATFORM", "").lower()
    return platform_name not in {"offscreen", "minimal"} and env_platform not in {"offscreen", "minimal"}


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

        macOS-specific: uses a higher window level via native API to
        ensure the bar floats above all other windows including
        full-screen apps.
        """

        def __init__(self, screen: QScreen | None = None) -> None:
            super().__init__()
            is_mac = platform.system() == "Darwin"

            # Window flags — platform-specific for best behavior
            if is_mac:
                # Avoid SplashScreen windows on macOS because they can steal
                # focus or pull the user to a different Space/Desktop.
                # Use a non-activating tool window instead.
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.Tool
                )
                # Make it non-focusable so it doesn't steal keyboard
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
                self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            else:
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.Tool
                )

            # Size and position — full width, top edge of screen
            target_screen = screen or QApplication.primaryScreen()
            avail = target_screen.availableGeometry()
            if is_mac:
                # On macOS, place just below the menu bar (avail.y() accounts for it)
                self.setGeometry(avail.x(), avail.y(),
                                 avail.width(), _BAR_HEIGHT)
            else:
                self.setGeometry(avail.x(), avail.y(),
                                 avail.width(), _BAR_HEIGHT)

            # On macOS, raise window level via native Cocoa API
            if is_mac and _should_raise_macos_window_level():
                self._raise_macos_window_level()

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
            if not is_mac:
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

        # ── macOS native window level ─────────────────────────

        def _raise_macos_window_level(self) -> None:
            """Set a high window level on macOS via Cocoa API.

            NSStatusWindowLevel (25) ensures the bar floats above
            normal windows, panels, and floating windows.

            QWidget.winId() on macOS returns a QNSView (NSView*).
            We need to get the parent NSWindow via [view window],
            then call [window setLevel:25].
            """
            try:
                from ctypes import c_void_p, cdll, c_int

                objc = cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")

                objc.sel_registerName.restype = c_void_p
                objc.objc_msgSend.restype = c_void_p

                # Get the NSView from Qt's winId
                ns_view = int(self.winId())

                # [view window] → NSWindow
                sel_window = objc.sel_registerName(b"window")
                objc.objc_msgSend.argtypes = [c_void_p, c_void_p]
                ns_window = objc.objc_msgSend(ns_view, sel_window)

                if not ns_window:
                    logger.debug("macOS: could not get NSWindow from QNSView")
                    return

                # [window setLevel:25]  (NSStatusWindowLevel)
                sel_setLevel = objc.sel_registerName(b"setLevel:")
                objc.objc_msgSend.argtypes = [c_void_p, c_void_p, c_int]
                objc.objc_msgSend(ns_window, sel_setLevel, 25)

                logger.debug("macOS: set window level to NSStatusWindowLevel (25)")
            except Exception as e:
                logger.debug(f"macOS window level setup failed (non-fatal): {e}")

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
            if platform.system() != "Darwin":
                self.raise_()  # ensure on top without stealing Spaces on macOS
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
