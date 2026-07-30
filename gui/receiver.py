"""
DisplayReceiver -- the GUI-side TCP server. Listens for connections from the
agent core (gui/publisher.py) and turns each incoming JSONL message into
a Qt signal, so the rest of the GUI never has to know about sockets at all.

Runs its own accept-loop in a background thread (not the Qt main thread --
socket I/O must not block the UI event loop), and only touches the UI via
pyqtSignal, which Qt marshals onto the main thread automatically. This is the
only file that imports `socket`/`json` for the GUI process; overlay.py deals
purely with Qt widgets and plain Python values.

Accepts multiple concurrent connections (though in practice there is only one
agent core process at a time) so restarting main.py doesn't require also
restarting the GUI.
"""

from __future__ import annotations

import json
import socket
import threading

from PyQt6.QtCore import QObject, pyqtSignal

from display_protocol import DEFAULT_HOST, DEFAULT_PORT


class DisplayReceiver(QObject):
    transcript_received = pyqtSignal(str, str, bool)   # text, source, is_final
    answer_chunk_received = pyqtSignal(str, bool)       # text, done
    status_received = pyqtSignal(str, str)              # state, detail
    clear_received = pyqtSignal(str)                    # target

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._server_sock: socket.socket | None = None
        self._running = False
        self._accept_thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(5)
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="display-receiver-accept"
        )
        self._accept_thread.start()

    def stop(self):
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass

    def _accept_loop(self):
        while self._running:
            try:
                conn, _addr = self._server_sock.accept()
            except OSError:
                break  # socket closed -> stop() was called
            threading.Thread(
                target=self._client_loop, args=(conn,), daemon=True,
                name="display-receiver-client",
            ).start()

    def _client_loop(self, conn: socket.socket):
        buf = b""
        with conn:
            while self._running:
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._dispatch(line)

    def _dispatch(self, line: bytes):
        try:
            msg = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        msg_type = msg.get("type")
        if msg_type == "transcript":
            self.transcript_received.emit(
                msg.get("text", ""), msg.get("source", ""), bool(msg.get("is_final"))
            )
        elif msg_type == "answer_chunk":
            self.answer_chunk_received.emit(msg.get("text", ""), bool(msg.get("done")))
        elif msg_type == "status":
            self.status_received.emit(msg.get("state", ""), msg.get("detail", ""))
        elif msg_type == "clear":
            self.clear_received.emit(msg.get("target", "all"))
        # unknown message types are ignored -- forward-compatible with future
        # fields/message kinds a Go rewrite might add without breaking old GUIs
