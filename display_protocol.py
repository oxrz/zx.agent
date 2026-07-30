"""
Agentic display protocol -- decoupling contract between the agent core logic and
external display frontends (GUI / future implementations in other languages).

Wire protocol: local TCP + newline-delimited JSON (JSON Lines / NDJSON), UTF-8
encoded, one JSON object per line, terminated with "\n". This minimal format
(instead of protobuf/gRPC) lets any language -- including a possible future Go
rewrite of the agent core -- implement a sender in a few dozen lines using only
the standard library, no codegen or extra dependencies required. This mirrors
the "TCP + JSONL" convention already used by the remote STT service in this
project (see audio/stt.py), so no new communication paradigm is introduced.

Roles:
  - The GUI process (gui/) is the TCP **server**: it only passively listens and
    renders. It has no knowledge of, and does not care, whether the data comes
    from the Python agent or a future Go agent.
  - The agent core logic (main.py, via display/publisher.py) is the TCP
    **client**: it connects to the GUI and pushes events. If the GUI isn't
    running, connection failures are swallowed silently -- the original
    CLI-only behavior (printing to the terminal) is unaffected, and no
    exception should ever block the main flow.

See PROTOCOL.md at the project root for the full message spec (for future
non-Python implementations -- no need to read this Python source to implement
a sender).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def encode_message(msg: Dict[str, Any]) -> bytes:
    """Encode as one line of JSON + newline, ready for TCP sendall()."""
    msg.setdefault("ts", time.time())
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def transcript_message(text: str, source: str, is_final: bool) -> Dict[str, Any]:
    """Real-time transcript text. source: "mic" | "loopback"; is_final=False means
    a streaming partial result -- the GUI should overwrite the previous unconfirmed
    line for the same source in place, not append to history."""
    return {"type": "transcript", "source": source, "text": text, "is_final": is_final}


def answer_chunk_message(text: str, done: bool = False) -> Dict[str, Any]:
    """One streaming increment of the assist-mode LLM answer. done=True marks the
    end of this answer turn (text is typically empty at that point, just a marker)."""
    return {"type": "answer_chunk", "text": text, "done": done}


def status_message(state: str, detail: str = "") -> Dict[str, Any]:
    """Optional status hint. state e.g. "listening" | "thinking" | "idle" | "error"."""
    return {"type": "status", "state": state, "detail": detail}


def clear_message(target: str = "all") -> Dict[str, Any]:
    """Clear the display. target: "transcript" | "answer" | "all"."""
    return {"type": "clear", "target": target}
