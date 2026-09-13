"""
Agentic display protocol -- decoupling contract between the agent core logic and
external display frontends (GUI / future implementations in other languages).

Wire protocol: local TCP + newline-delimited JSON (JSON Lines / NDJSON), UTF-8
encoded, one JSON object per line, terminated with "\n". This minimal format
(instead of protobuf/gRPC) lets any language -- including a possible future Go
rewrite of the agent core -- implement a sender in a few dozen lines using only
the standard library, no codegen or extra dependencies required. This is an
independent local display channel; the remote STT service uses WebSocket and
binary PCM frames (see audio/stt.py).

Roles:
  - The GUI process (gui/) is the TCP **server**: it only passively listens and
    renders. It has no knowledge of, and does not care, whether the data comes
    from the Python agent or a future Go agent.
  - The agent core logic (main.py, via gui/publisher.py) is the TCP
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
# Deliberately above 15000. On Windows, Hyper-V (which WSL2 enables) reserves
# blocks of ports out of the TCP dynamic port range -- by default 1024-15000 --
# and picks fresh blocks at boot. Binding inside a reserved block fails with
# WinError 10013 rather than the 10048 you would expect for a busy port, and
# nothing is listening there, so it reads as "free" to every liveness probe.
# The previous default of 8765 sat inside one such block after a reboot and
# broke the overlay intermittently. Anything above the dynamic range is immune.
# Check the current reservations with:
#   netsh interface ipv4 show excludedportrange protocol=tcp
DEFAULT_PORT = 18765


def encode_message(msg: Dict[str, Any]) -> bytes:
    """Encode as one line of JSON + newline, ready for TCP sendall()."""
    msg.setdefault("ts", time.time())
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def transcript_message(
    text: str, source: str, is_final: bool, pending_correction: bool = False,
    utterance_id: int | None = None, replace_utterance_id: int | None = None,
) -> Dict[str, Any]:
    """Real-time transcript text. source: "mic" | "loopback"; is_final=False means
    a streaming partial result -- the GUI should overwrite the previous unconfirmed
    line for the same source in place, not append to history.
    pending_correction=True: a Whisper final for an already-past utterance -- add to
    history but do NOT clear the current in-progress partial for this source."""
    msg: Dict[str, Any] = {"type": "transcript", "source": source, "text": text, "is_final": is_final}
    if pending_correction:
        msg["pending_correction"] = True
    if utterance_id is not None:
        msg["utterance_id"] = utterance_id
    if replace_utterance_id is not None:
        msg["replace_utterance_id"] = replace_utterance_id
    return msg


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
