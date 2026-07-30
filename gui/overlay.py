"""
OverlayWindow -- the transparent, always-on-top real-time subtitle/answer
overlay window (Windows-first, cross-platform via Qt).

Design notes:
  - Frameless + translucent background + WindowStaysOnTopHint: gives the
    "floating subtitle" look used by live-caption / karaoke-style overlays --
    only the text itself is visible, no window chrome, and it sits above the
    meeting/video window it is annotating.
  - WA_TranslucentBackground requires the *widget* background to be painted
    manually (see paintEvent) rather than relying on a stylesheet background
    on the top-level window -- Qt only makes the window surface itself
    alpha-capable, child widgets still need explicit transparent styling.
  - WindowTransparentForInput is intentionally NOT set: the window stays
    click-and-drag-able so the user can reposition it. Dragging works from
    anywhere in the window body, including the text areas (see
    _DragPassthroughTextEdit below) -- text selection is disabled so mouse
    events always mean "drag the window", not "select text".
  - Transcript and answer text are shown in scrollable, auto-growing history
    panes (not single overwritten lines) so the running conversation stays
    visible instead of getting replaced sentence by sentence. History is
    capped (see _MAX_HISTORY_LINES) so a long-running meeting doesn't grow
    the widget's memory/text buffer unbounded.
  - Only renders what it receives over the DisplayReceiver signals -- knows
    nothing about STT/LLM/audio. Whether the sender is the Python agent core
    or, later, a Go rewrite, this file does not change at all.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.receiver import DisplayReceiver

_BG_ALPHA = 140            # 0-255, background panel opacity
_MAX_HISTORY_LINES = 50    # cap on confirmed transcript lines kept in memory/view
_DEFAULT_WIDTH = 900
_DEFAULT_HEIGHT = 380
_MIN_WIDTH = 320
_MIN_HEIGHT = 160

_SOURCE_LABELS = {"mic": "You", "loopback": "Audio"}

_TRANSPARENT_TEXTEDIT_STYLE = """
QPlainTextEdit {
    background: transparent;
    border: none;
    color: #ffffff;
}
"""


class _DragPassthroughTextEdit(QPlainTextEdit):
    """A read-only QPlainTextEdit that forwards mouse drag to move the
    top-level window, instead of selecting text. Needed because a frameless
    window has no title bar -- without this, clicking into the transcript/
    answer area would just place a text cursor instead of letting the user
    reposition the whole overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFrameStyle(0)
        self.setStyleSheet(_TRANSPARENT_TEXTEDIT_STYLE)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._drag_offset = None

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.window().pos()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_offset is not None:
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_offset = None

    def append_and_scroll(self, text: str):
        self.setPlainText(text)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class OverlayWindow(QWidget):
    def __init__(self, host: str, port: int, width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT):
        super().__init__()
        self._drag_offset = None
        self._window_width = max(width, _MIN_WIDTH)
        self._window_height = max(height, _MIN_HEIGHT)
        # Confirmed (is_final=True) lines, oldest first, capped at
        # _MAX_HISTORY_LINES so the view keeps scrolling instead of growing forever.
        self._transcript_history: list[str] = []
        # Per-source text that hasn't been finalized yet (still being spoken/
        # recognized) -- shown as a trailing, still-updating line under the
        # confirmed history, replaced in place until it's finalized.
        self._transcript_partial: dict[str, str] = {"mic": "", "loopback": ""}
        self._answer_text = ""

        self._setup_window()
        self._setup_ui()
        self._setup_shortcuts()

        self._receiver = DisplayReceiver(host=host, port=port)
        self._receiver.transcript_received.connect(self._on_transcript)
        self._receiver.answer_chunk_received.connect(self._on_answer_chunk)
        self._receiver.status_received.connect(self._on_status)
        self._receiver.clear_received.connect(self._on_clear)
        self._receiver.start()

    # ---- window setup ----
    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # keep it out of the taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(self._window_width, self._window_height)
        self._move_to_bottom_center()

    def _move_to_bottom_center(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + screen.height() - self.height() - 60
        self.move(x, y)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(6)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #8fd3ff; background: transparent;")
        self._status_label.setFont(QFont("Segoe UI", 10))

        self._transcript_view = _DragPassthroughTextEdit()
        self._transcript_view.setFont(QFont("Segoe UI", 14))

        self._answer_view = _DragPassthroughTextEdit()
        self._answer_view.setFont(QFont("Segoe UI", 13))
        self._answer_view.setStyleSheet(_TRANSPARENT_TEXTEDIT_STYLE.replace("#ffffff", "#ffe38f"))

        layout.addWidget(self._status_label)
        layout.addWidget(self._transcript_view, stretch=3)
        layout.addWidget(self._answer_view, stretch=2)

    def _setup_shortcuts(self):
        """Frameless window has no title bar / close button, so provide a
        keyboard-only way to quit -- Escape closes the overlay window."""
        close_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        close_shortcut.activated.connect(self.close)

    # ---- background painting (required for real transparency, see module docstring) ----
    def paintEvent(self, event):  # noqa: N802 (Qt override naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 20, 20, _BG_ALPHA))
        painter.drawRoundedRect(self.rect(), 14, 14)

    # ---- drag to reposition (frameless window has no title bar) ----
    # (covers the margins/status-label area; the text views handle their own
    # dragging via _DragPassthroughTextEdit, since they'd otherwise swallow
    # the mouse press themselves)
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_offset = None

    # ---- rendering helpers ----
    def _render_transcript(self):
        lines = list(self._transcript_history)
        for source, partial in self._transcript_partial.items():
            if partial:
                label = _SOURCE_LABELS.get(source, source)
                lines.append(f"{label}: {partial}")
        self._transcript_view.append_and_scroll("\n".join(lines))

    # ---- DisplayReceiver signal handlers ----
    def _on_transcript(self, text: str, source: str, is_final: bool):
        if is_final:
            self._transcript_partial[source] = ""
            if text:
                label = _SOURCE_LABELS.get(source, source)
                self._transcript_history.append(f"{label}: {text}")
                overflow = len(self._transcript_history) - _MAX_HISTORY_LINES
                if overflow > 0:
                    del self._transcript_history[:overflow]
        else:
            self._transcript_partial[source] = text
        self._render_transcript()

    def _on_answer_chunk(self, text: str, done: bool):
        if text:
            self._answer_text += text
            self._answer_view.append_and_scroll(self._answer_text)

    def _on_status(self, state: str, detail: str):
        self._status_label.setText(f"[{state}] {detail}" if detail else f"[{state}]")

    def _on_clear(self, target: str):
        if target in ("transcript", "all"):
            self._transcript_history = []
            self._transcript_partial = {"mic": "", "loopback": ""}
            self._transcript_view.append_and_scroll("")
        if target in ("answer", "all"):
            self._answer_text = ""
            self._answer_view.append_and_scroll("")

    def closeEvent(self, event):  # noqa: N802
        self._receiver.stop()
        super().closeEvent(event)
