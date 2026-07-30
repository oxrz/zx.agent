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
  - Frameless windows lose the OS's built-in "drag the edge to resize" affordance
    along with the title bar, so it's reimplemented manually here (see
    _EdgeResizer): hovering within _RESIZE_MARGIN pixels of an edge/corner shows
    the matching resize cursor, and dragging from there resizes instead of moves.
    Resizing takes priority over the existing "drag body to move" behavior --
    a press near the edge is treated as a resize, not a move, so the two don't
    fight over the same mouse gesture.
  - "Closing" this window (Escape, or the right-click menu's Hide item) only
    hides it -- it does NOT quit the GUI process. Since this is the only
    top-level window in the app, gui/app.py disables Qt's default "quit when
    the last window closes" behavior and adds a system tray icon instead,
    which is the only way to bring the window back afterward (and the only
    place the real "Quit" action lives). See gui/app.py.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.receiver import DisplayReceiver
from gui.settings_window import SettingsWindow

_MAX_HISTORY_LINES = 50    # cap on confirmed transcript lines kept in memory/view
_DEFAULT_WIDTH = 900
_DEFAULT_HEIGHT = 380
_MIN_WIDTH = 320
_MIN_HEIGHT = 160
_RESIZE_MARGIN = 8  # pixels from each edge/corner that count as a resize handle

_DEFAULT_OPACITY = 140    # 0-255, background panel alpha -- same range/meaning as the old _BG_ALPHA
_MIN_OPACITY = 20          # below this the panel is practically invisible, not useful
_MAX_OPACITY = 255

_SOURCE_LABELS = {"mic": "You", "loopback": "Audio"}

# Theme presets: background panel color + text colors. "dark" (black background,
# white/near-white text) is the original look; "light" is the inverse for use
# over bright backgrounds where white text would wash out. Both keep the answer
# text in a warm accent color so it's visually distinct from the transcript,
# just swapped to something that still reads clearly against each background.
_THEMES = {
    "dark": {
        "bg": (20, 20, 20),
        "status": "#8fd3ff",
        "transcript": "#ffffff",
        "answer": "#ffe38f",
    },
    "light": {
        "bg": (245, 245, 245),
        "status": "#0a63c4",
        "transcript": "#111111",
        "answer": "#8a5200",
    },
}
_DEFAULT_THEME = "dark"

_TRANSPARENT_TEXTEDIT_STYLE = """
QPlainTextEdit {{
    background: transparent;
    border: none;
    color: {color};
}}
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
    def __init__(
        self,
        host: str,
        port: int,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        opacity: int = _DEFAULT_OPACITY,
        theme: str = _DEFAULT_THEME,
    ):
        super().__init__()
        self._drag_offset = None
        self._opacity = min(max(opacity, _MIN_OPACITY), _MAX_OPACITY)
        self._theme = theme if theme in _THEMES else _DEFAULT_THEME
        self._settings_window = None
        # Edge-resize state: which edge(s) the cursor is currently over (a
        # subset of {"left", "right", "top", "bottom"}, empty = not on an
        # edge), and the geometry captured at the moment a resize drag starts
        # (so deltas are computed against a fixed reference, not accumulated
        # error from repeatedly re-reading self.geometry() mid-drag).
        self._resize_edges: set[str] = set()
        self._resize_start_geometry = None
        self._resize_start_pos = None
        self._window_width = max(width, _MIN_WIDTH)
        self._window_height = max(height, _MIN_HEIGHT)
        self.setMouseTracking(True)  # needed to update the resize cursor on hover, not just while dragging
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
        self._status_label.setFont(QFont("Segoe UI", 10))

        self._transcript_view = _DragPassthroughTextEdit()
        self._transcript_view.setFont(QFont("Segoe UI", 14))

        self._answer_view = _DragPassthroughTextEdit()
        self._answer_view.setFont(QFont("Segoe UI", 13))

        layout.addWidget(self._status_label)
        layout.addWidget(self._transcript_view, stretch=3)
        layout.addWidget(self._answer_view, stretch=2)

        self._apply_theme()  # sets each widget's colors from self._theme

    def _setup_shortcuts(self):
        """Frameless window has no title bar / close button, so provide a
        keyboard-only way to quit -- Escape closes the overlay window."""
        close_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        close_shortcut.activated.connect(self.close)

    def contextMenuEvent(self, event):  # noqa: N802
        """Right-click opens Settings, since there's no title bar / system
        menu on a frameless window to hang this off of.

        "Hide" (not "Close"/"Quit"): closing a frameless Qt window only hides
        it by default (the widget object survives), but since this is the
        application's only top-level window, Qt would otherwise treat that as
        "the last window closed" and quit the whole GUI process -- see
        gui/app.py's setQuitOnLastWindowClose(False) and the system tray icon,
        which is what actually brings the window back (left-click/double-click
        the tray icon, or its own "Show Overlay" menu item) and owns the real
        "Quit" action."""
        menu = QMenu(self)
        settings_action = menu.addAction("Settings...")
        settings_action.triggered.connect(self.open_settings)
        menu.addSeparator()
        hide_action = menu.addAction("Hide (bring back from the tray icon)")
        hide_action.triggered.connect(self.close)
        menu.exec(event.globalPos())

    def open_settings(self):
        """Open (or bring to front) the Settings window. Public so both the
        overlay's own right-click menu and the system tray icon's menu
        (gui/app.py) can trigger it -- the tray needs this too, since the
        overlay window itself may currently be hidden."""
        if self._settings_window is None or not self._settings_window.isVisible():
            self._settings_window = SettingsWindow(self)
            self._settings_window.show()
        else:
            self._settings_window.raise_()
            self._settings_window.activateWindow()

    # ---- background painting (required for real transparency, see module docstring) ----
    def paintEvent(self, event):  # noqa: N802 (Qt override naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        r, g, b = _THEMES[self._theme]["bg"]
        painter.setBrush(QColor(r, g, b, self._opacity))
        painter.drawRoundedRect(self.rect(), 14, 14)

    # ---- live-adjustable appearance (driven by the settings window, see
    # gui/settings_window.py -- these are the only two things it's currently
    # allowed to change; everything else about the window stays fixed once
    # launched) ----
    def _apply_theme(self):
        colors = _THEMES[self._theme]
        self._status_label.setStyleSheet(f"color: {colors['status']}; background: transparent;")
        self._transcript_view.setStyleSheet(
            _TRANSPARENT_TEXTEDIT_STYLE.format(color=colors["transcript"])
        )
        self._answer_view.setStyleSheet(
            _TRANSPARENT_TEXTEDIT_STYLE.format(color=colors["answer"])
        )
        self.update()  # repaint the background panel with the new theme's bg color

    def set_theme(self, theme: str):
        if theme not in _THEMES:
            return
        self._theme = theme
        self._apply_theme()

    def set_opacity(self, opacity: int):
        self._opacity = min(max(int(opacity), _MIN_OPACITY), _MAX_OPACITY)
        self.update()  # repaint with the new alpha; text colors are unaffected

    @property
    def theme(self) -> str:
        return self._theme

    @property
    def opacity(self) -> int:
        return self._opacity

    # ---- edge resize + drag to reposition (frameless window has no title bar) ----
    # (covers the margins/status-label area; the text views handle their own
    # move-dragging via _DragPassthroughTextEdit, since they'd otherwise swallow
    # the mouse press themselves -- but they don't sit right at the window edge,
    # since the layout has a margin, so edge-resize detection here is enough)
    def _edges_at(self, pos) -> set[str]:
        """Return which edge(s) `pos` (widget-local QPoint) is within
        _RESIZE_MARGIN pixels of. Can return up to two edges (a corner), e.g.
        {"left", "top"} for the top-left corner."""
        edges = set()
        if pos.x() <= _RESIZE_MARGIN:
            edges.add("left")
        elif pos.x() >= self.width() - _RESIZE_MARGIN:
            edges.add("right")
        if pos.y() <= _RESIZE_MARGIN:
            edges.add("top")
        elif pos.y() >= self.height() - _RESIZE_MARGIN:
            edges.add("bottom")
        return edges

    _EDGE_CURSORS = {
        frozenset({"left"}): Qt.CursorShape.SizeHorCursor,
        frozenset({"right"}): Qt.CursorShape.SizeHorCursor,
        frozenset({"top"}): Qt.CursorShape.SizeVerCursor,
        frozenset({"bottom"}): Qt.CursorShape.SizeVerCursor,
        frozenset({"left", "top"}): Qt.CursorShape.SizeFDiagCursor,
        frozenset({"right", "bottom"}): Qt.CursorShape.SizeFDiagCursor,
        frozenset({"right", "top"}): Qt.CursorShape.SizeBDiagCursor,
        frozenset({"left", "bottom"}): Qt.CursorShape.SizeBDiagCursor,
    }

    def _update_cursor(self, edges: set[str]):
        cursor = self._EDGE_CURSORS.get(frozenset(edges), Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edges = self._edges_at(event.position().toPoint())
        if edges:
            self._resize_edges = edges
            self._resize_start_geometry = self.geometry()
            self._resize_start_pos = event.globalPosition().toPoint()
        else:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._resize_start_geometry is not None:
            self._perform_resize(event.globalPosition().toPoint())
        elif self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        else:
            # No button held -- just update the hover cursor so the user sees
            # a resize handle before they click, same as any normal OS window.
            self._update_cursor(self._edges_at(event.position().toPoint()))

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_offset = None
        self._resize_edges = set()
        self._resize_start_geometry = None
        self._resize_start_pos = None

    def _perform_resize(self, global_pos):
        """Resize (and, for left/top edges, reposition) the window based on
        how far the mouse has moved since the resize drag started, clamped to
        the configured minimum size so the window can't be shrunk into
        uselessness or flip inside-out."""
        delta = global_pos - self._resize_start_pos
        geo = self._resize_start_geometry
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()

        if "right" in self._resize_edges:
            w = max(_MIN_WIDTH, geo.width() + delta.x())
        elif "left" in self._resize_edges:
            w = max(_MIN_WIDTH, geo.width() - delta.x())
            x = geo.x() + (geo.width() - w)

        if "bottom" in self._resize_edges:
            h = max(_MIN_HEIGHT, geo.height() + delta.y())
        elif "top" in self._resize_edges:
            h = max(_MIN_HEIGHT, geo.height() - delta.y())
            y = geo.y() + (geo.height() - h)

        self.setGeometry(x, y, w, h)

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
