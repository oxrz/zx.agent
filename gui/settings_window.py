"""
SettingsWindow -- a small, normal (framed) Qt window for adjusting the
overlay's live-adjustable appearance: background opacity and color theme
(dark/light).

Scope, deliberately narrow for now: this only touches OverlayWindow's own
in-memory state (self._opacity / self._theme via set_opacity()/set_theme()).
It does not read or write any config file, and it has no connection to the
agent core process -- switching capture mode (transcribe/mic/assist) or any
other agent-side setting is out of scope here, since that lives in a
completely separate process and would need a way to signal it (not built
yet; see PROTOCOL.md notes on this being agent-core-side work for later).

Kept as a normal (non-frameless) window on purpose -- it's a utility dialog,
not part of the "floating subtitle" look, so it should behave like a regular
window (has its own title bar/close button, doesn't need to be draggable via
its body, doesn't need to stay always-on-top).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class SettingsWindow(QDialog):
    def __init__(self, overlay):
        """overlay: the OverlayWindow instance to apply changes to live."""
        super().__init__(overlay, Qt.WindowType.Dialog)
        self._overlay = overlay
        self.setWindowTitle("Overlay Settings")
        self.setFixedWidth(320)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Background opacity"))
        opacity_row = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        from gui.overlay import _MAX_OPACITY, _MIN_OPACITY
        self._opacity_slider.setRange(_MIN_OPACITY, _MAX_OPACITY)
        self._opacity_slider.setValue(self._overlay.opacity)
        self._opacity_value_label = QLabel(str(self._overlay.opacity))
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        opacity_row.addWidget(self._opacity_value_label)
        layout.addLayout(opacity_row)

        layout.addWidget(QLabel("Theme"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Dark (black background, white text)", "dark")
        self._theme_combo.addItem("Light (white background, black text)", "light")
        current_index = self._theme_combo.findData(self._overlay.theme)
        if current_index >= 0:
            self._theme_combo.setCurrentIndex(current_index)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        layout.addWidget(self._theme_combo)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def _on_opacity_changed(self, value: int):
        self._opacity_value_label.setText(str(value))
        self._overlay.set_opacity(value)

    def _on_theme_changed(self, index: int):
        theme = self._theme_combo.itemData(index)
        self._overlay.set_theme(theme)
