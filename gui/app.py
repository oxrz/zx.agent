#!/usr/bin/env python
"""
Agentic GUI - standalone transparent overlay display frontend.

Runs as its own process, completely independent from main.py (the agent
core). It only speaks the TCP+JSONL protocol defined in display_protocol.py,
so it does not import anything from audio/, ai/, or memory.py -- no matter
what language the agent core ends up being written in, this process does not
need to change.
"""

import argparse
import os
import signal
import sys
import threading

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from display_protocol import DEFAULT_HOST, DEFAULT_PORT
from gui.overlay import (
    OverlayWindow,
    _DEFAULT_HEIGHT,
    _DEFAULT_OPACITY,
    _DEFAULT_THEME,
    _DEFAULT_WIDTH,
    _THEMES,
)

# Must match the identical literal in main.py's _GUI_LAUNCH_TOKEN_ENV -- kept
# as a plain duplicated string rather than a shared import, since importing
# anything from main.py would pull in the whole agent core (audio/, ai/,
# STT client, etc.) into the GUI process, which is exactly what the
# decoupling in display_protocol.py is meant to avoid.
_GUI_LAUNCH_TOKEN_ENV = "AGENTIC_GUI_LAUNCH_TOKEN"

# How long to give Qt's normal shutdown path (app.quit() -> event loop
# returns -> process exits) before assuming it's stuck and force-exiting.
# See _quit() for why this exists.
_QUIT_WATCHDOG_SECONDS = 1.5


def _make_tray_icon() -> QIcon:
    """A small drawn dot instead of a bundled image file -- keeps the GUI
    free of any external asset dependency for something this minor. Just
    needs to be visually distinct enough to find in a crowded tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(255, 255, 255, 220))
    painter.setBrush(QColor(90, 170, 255, 230))
    painter.drawEllipse(2, 2, 28, 28)
    painter.end()
    return QIcon(pixmap)


def _quit(app: QApplication, tray: QSystemTrayIcon | None):
    """Actually terminate the process, robustly.

    Plain app.quit() from a tray icon's menu action is known to sometimes not
    fully tear down the process on Windows -- the native tray icon holds onto
    OS-level resources (a taskbar notification area slot), and if those don't
    release cleanly, Qt's event loop can return from exec() while some
    background thread or native handle keeps the interpreter alive, leaving a
    zombie process with no window and no visible tray icon.

    So: explicitly hide the tray icon first (releases its native resource
    deterministically before quitting, rather than leaving it to whatever
    order objects happen to get torn down in), call quit() as normal, and
    arm a plain Python-thread watchdog that force-exits via os._exit(0) if
    the process is somehow still alive shortly after. os._exit() bypasses
    Python/Qt cleanup entirely -- deliberate, since the whole point is "get
    out no matter what got stuck."
    """
    if tray is not None:
        tray.hide()
    app.quit()

    def _watchdog():
        os._exit(0)

    timer = threading.Timer(_QUIT_WATCHDOG_SECONDS, _watchdog)
    timer.daemon = True
    timer.start()


def _setup_tray_icon(window: OverlayWindow, app: QApplication) -> QSystemTrayIcon:
    """The overlay window is this app's only top-level window and is closed
    (hidden, see OverlayWindow.contextMenuEvent) far more often than a normal
    window -- with Qt's default quitOnLastWindowClose behavior, hiding it
    would silently kill the whole GUI process with no way to bring it back.
    The tray icon is what makes "hide" actually mean hide: it survives the
    window being closed and is the only place "show it again", "open
    Settings", and "actually quit" live."""
    tray = QSystemTrayIcon(_make_tray_icon(), app)
    tray.setToolTip("Agentic overlay")

    menu = QMenu()
    show_action = menu.addAction("Show Overlay")
    show_action.triggered.connect(lambda: (window.show(), window.raise_(), window.activateWindow()))
    settings_action = menu.addAction("Settings...")
    settings_action.triggered.connect(window.open_settings)
    menu.addSeparator()
    quit_action = menu.addAction("Quit")
    quit_action.triggered.connect(lambda: _quit(app, tray))
    tray.setContextMenu(menu)

    # Left-click (Trigger) or double-click toggles visibility, so the tray
    # icon itself acts as the show/hide button, not just a menu launcher.
    def on_activated(reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            if window.isVisible():
                window.hide()
            else:
                window.show()
                window.raise_()
                window.activateWindow()

    tray.activated.connect(on_activated)
    tray.show()
    return tray


def main():
    # main.py (--gui/-g) sets this env var when it spawns this process, so
    # gui.app is never runnable as a standalone command -- the agent and the
    # GUI are now a paired session (see main.py's _start_gui_monitor: if this
    # process exits, the agent shuts itself down too). Running this by hand
    # would silently produce an overlay with no paired agent to shut down,
    # and no way for the agent side of that pairing logic to ever apply.
    if os.environ.get(_GUI_LAUNCH_TOKEN_ENV) != "1":
        print(
            "gui.app is not meant to be run directly -- start it via "
            "`python main.py --gui` (or -g), which launches this "
            "automatically and keeps the two processes paired.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Agentic transparent overlay GUI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--width", type=int, default=_DEFAULT_WIDTH, help=f"window width in pixels (default: {_DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT, help=f"window height in pixels (default: {_DEFAULT_HEIGHT})")
    parser.add_argument("--opacity", type=int, default=_DEFAULT_OPACITY, help=f"background panel opacity, 0-255 (default: {_DEFAULT_OPACITY}); also adjustable live via the Settings window (right-click the overlay)")
    parser.add_argument("--theme", choices=sorted(_THEMES), default=_DEFAULT_THEME, help=f"color theme (default: {_DEFAULT_THEME}); also switchable live via the Settings window (right-click the overlay)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    # Without this, hiding the overlay window (Escape / right-click Hide) --
    # its normal, frequent, non-destructive use -- would be indistinguishable
    # from "the app's last window closed" and Qt would quit the whole process.
    # The system tray icon (see _setup_tray_icon) is what makes it possible to
    # bring the window back after that, and owns the real "Quit" action.
    app.setQuitOnLastWindowClosed(False)

    # Qt's C++ event loop doesn't yield back to the Python interpreter on its
    # own, so a bare Ctrl+C (SIGINT) previously wasn't delivered until the
    # loop was woken up some other way -- that's why Ctrl+C appeared to do
    # nothing (or needed mashing). A no-op QTimer firing periodically gives
    # Python's signal handler a chance to run between Qt events, so one
    # Ctrl+C now quits cleanly via closeEvent() instead of being force-killed.
    tray_ref = {"tray": None}
    signal.signal(signal.SIGINT, lambda *_: _quit(app, tray_ref["tray"]))
    keepalive_timer = QTimer()
    keepalive_timer.timeout.connect(lambda: None)
    keepalive_timer.start(200)

    window = OverlayWindow(
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        opacity=args.opacity,
        theme=args.theme,
    )
    window.show()

    if QSystemTrayIcon.isSystemTrayAvailable():
        tray_ref["tray"] = _setup_tray_icon(window, app)
    else:
        # No system tray available (some Linux desktop environments, some
        # minimal/headless setups) -- fall back to the old behavior so the
        # app is still usable, just without the "bring it back" affordance.
        app.setQuitOnLastWindowClosed(True)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
