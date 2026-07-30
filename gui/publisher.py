"""
DisplayPublisher -- the single exit point for the agent core to push display
events out.

main.py should only call the methods this module provides (transcript /
answer_chunk / status / clear). It must never import PyQt or any GUI-related
code directly -- that is the whole point of the decoupling: if the GUI
process is not running, crashes, or is later rewritten in another language,
the agent core's behavior is completely unaffected, because the core has no
idea what (or whether) a display frontend even exists.

Implementation: a lazily-connecting TCP client + background thread + bounded
queue.
  - enabled=False (default, GUI not explicitly turned on): every method is a
    pure no-op, zero overhead.
  - Sending never blocks the caller's thread (audio processing thread /
    asyncio thread) -- it just drops the message onto a queue and returns.
  - If disconnected / the GUI isn't running, messages are dropped silently;
    reconnection is retried periodically. No exceptions are raised, no retry
    storms.
  - If the queue is full (GUI stuck or not connected for a while), the oldest
    message is dropped so memory usage stays bounded.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any, Dict, Optional

from display_protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    answer_chunk_message,
    clear_message,
    encode_message,
    status_message,
    transcript_message,
)

_RECONNECT_INTERVAL = 3.0  # seconds between reconnect attempts
_QUEUE_MAXSIZE = 200


class DisplayPublisher:
    def __init__(
        self,
        enabled: bool = False,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        logger=None,
    ):
        self.enabled = enabled
        self._host = host
        self._port = port
        self._logger = logger
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._last_connect_attempt = 0.0

        if self.enabled:
            self._start()

    def _start(self):
        self._running = True
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="display-publisher"
        )
        self._worker.start()

    def _log(self, level: str, msg: str):
        if self._logger:
            getattr(self._logger, level, self._logger.info)(msg)

    def _run(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._ensure_connected()
            if self._sock is None:
                continue  # can't connect -- GUI not running is normal, drop and move on
            try:
                self._sock.sendall(encode_message(item))
            except OSError:
                self._close_socket()

    def _ensure_connected(self):
        if self._sock is not None:
            return
        now = time.time()
        if now - self._last_connect_attempt < _RECONNECT_INTERVAL:
            return
        self._last_connect_attempt = now
        try:
            sock = socket.create_connection((self._host, self._port), timeout=0.5)
            self._sock = sock
            self._log("info", f"Display frontend connected ({self._host}:{self._port})")
        except OSError:
            self._sock = None

    def _close_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _enqueue(self, msg: Dict[str, Any]):
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop the oldest to make room
                self._queue.put_nowait(msg)
            except queue.Empty:
                pass

    # ---- public API used by main.py ----
    def transcript(self, text: str, source: str, is_final: bool):
        self._enqueue(transcript_message(text, source, is_final))

    def answer_chunk(self, text: str, done: bool = False):
        self._enqueue(answer_chunk_message(text, done))

    def status(self, state: str, detail: str = ""):
        self._enqueue(status_message(state, detail))

    def clear(self, target: str = "all"):
        self._enqueue(clear_message(target))

    def close(self):
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=1)
        self._close_socket()
