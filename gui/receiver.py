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
        try:
            self._server_sock.bind((self._host, self._port))
        except OSError as e:
            # Windows only: WSAEACCES (10013) on bind almost never means "busy" --
            # a busy port gives WSAEADDRINUSE (10048). It means the port sits in a
            # range Hyper-V has reserved out of the TCP dynamic port range, which
            # WSL2 causes and which is re-picked at every boot. Nothing is
            # listening on a reserved port, so the agent core's pre-launch probe
            # sees it as free and starts this process anyway. Say so explicitly:
            # the bare OSError sends people hunting for a process to kill that
            # does not exist.
            self._server_sock.close()
            self._server_sock = None
            self._running = False
            if getattr(e, "winerror", None) == 10013:
                raise OSError(
                    f"Cannot bind {self._host}:{self._port} -- the port is reserved by "
                    f"Windows (WinError 10013), not in use by another program. Hyper-V "
                    f"(enabled by WSL2) reserves port blocks out of the dynamic range "
                    f"(usually 1024-15000) and re-picks them on every boot. List them with "
                    f"'netsh interface ipv4 show excludedportrange protocol=tcp' and set "
                    f"display.port in config/common.yaml to a port above 15000."
                ) from e
            raise
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
