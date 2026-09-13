"""Debug-only output duplication for terminal text and GUI render snapshots."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class OutputRecorder:
    """Append text or JSONL records to a UTF-8 file, safely across threads."""

    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open(mode, encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._file.write(text)
            self._file.flush()

    def event(self, message: dict) -> None:
        record = dict(message)
        record.setdefault("recorded_at", time.time())
        self.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()


class TeeTextIO:
    """Write to the original stream and a recorder at the same time."""

    def __init__(self, stream, recorder: OutputRecorder):
        self._stream = stream
        self._recorder = recorder

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        self._stream.flush()
        self._recorder.write(text)
        return written

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @property
    def encoding(self):
        return self._stream.encoding
