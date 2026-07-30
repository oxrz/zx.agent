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
import signal
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from display_protocol import DEFAULT_HOST, DEFAULT_PORT
from gui.overlay import OverlayWindow, _DEFAULT_HEIGHT, _DEFAULT_WIDTH


def main():
    parser = argparse.ArgumentParser(description="Agentic transparent overlay GUI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--width", type=int, default=_DEFAULT_WIDTH, help=f"window width in pixels (default: {_DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT, help=f"window height in pixels (default: {_DEFAULT_HEIGHT})")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Qt's C++ event loop doesn't yield back to the Python interpreter on its
    # own, so a bare Ctrl+C (SIGINT) previously wasn't delivered until the
    # loop was woken up some other way -- that's why Ctrl+C appeared to do
    # nothing (or needed mashing). A no-op QTimer firing periodically gives
    # Python's signal handler a chance to run between Qt events, so one
    # Ctrl+C now quits cleanly via closeEvent() instead of being force-killed.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keepalive_timer = QTimer()
    keepalive_timer.timeout.connect(lambda: None)
    keepalive_timer.start(200)

    window = OverlayWindow(host=args.host, port=args.port, width=args.width, height=args.height)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
