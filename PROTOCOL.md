# Display Protocol (agent core <-> GUI)

This document specifies the wire protocol used to decouple the agent core
(speech recognition + Q&A logic) from the display frontend (currently a PyQt
transparent overlay in `gui/`). It exists so that if the agent core is later
rewritten in another language (e.g. Go), the new implementation only needs to
follow this document -- the GUI does not need to change at all, and vice
versa.

## Transport

- Local TCP socket, default `127.0.0.1:8765`.
- The **GUI is the TCP server** (binds and listens). The **agent core is the
  TCP client** (connects out).
- This direction was chosen deliberately: the GUI can be started or restarted
  independently of the agent core, and the agent core does not need to know
  or care whether a GUI is even running. If the connection fails, the agent
  core silently drops display events and continues operating exactly as it
  does today (CLI-only output) -- see `gui/publisher.py`.
- No authentication, no TLS. Only bind to `127.0.0.1` unless you understand
  the risk of exposing this on a wider network.

## Framing

- Newline-delimited JSON (JSON Lines / NDJSON).
- Each message is one JSON object, UTF-8 encoded, terminated by a single
  `\n` byte.
- The receiver must buffer partial reads and only parse once a full line
  (up to `\n`) has been received -- messages may be sent in multiple small
  TCP writes/reads.
- Unknown message `type` values, or unknown fields within a known type,
  MUST be ignored by the receiver (forward compatibility -- a newer sender
  may add fields/types an older GUI doesn't understand yet).

## Message Types

Every message includes a `ts` field (float, Unix timestamp in seconds),
added automatically by the sender if not already present.

### `transcript`

Real-time transcription of one audio source.

```json
{"type": "transcript", "source": "loopback", "text": "hello world", "is_final": false, "ts": 1234567890.123}
```

- `source`: `"mic"` or `"loopback"`.
- `text`: the current text for this source. For streaming/incremental
  updates (`is_final: false`), this is the latest recognized text for the
  *current, still-open* utterance and REPLACES (does not append to) the
  previous partial for the same source. When `is_final: true`, this is the
  final confirmed text for that utterance.
- `is_final`: whether this utterance has ended.

### `answer_chunk`

One increment of a streamed LLM answer (assist mode only).

```json
{"type": "answer_chunk", "text": "The direct answer is...", "done": false, "ts": 1234567890.123}
```

- `text`: text to append to the currently-displayed answer. Empty string is
  valid (e.g. paired with `done: true` as a pure end-of-stream marker).
- `done`: `true` marks the end of this answer turn. The GUI should treat the
  next `answer_chunk` (or an explicit `clear`) as the start of a new answer.

### `status`

Optional, informational only. Safe to ignore entirely.

```json
{"type": "status", "state": "thinking", "detail": "", "ts": 1234567890.123}
```

- `state`: free-form short string, e.g. `"listening"`, `"thinking"`, `"idle"`,
  `"error"`.
- `detail`: free-form human-readable text, may be empty.

### `clear`

Tells the GUI to clear part or all of its displayed content.

```json
{"type": "clear", "target": "answer", "ts": 1234567890.123}
```

- `target`: one of `"transcript"`, `"answer"`, `"all"`.

## Implementing a sender in another language

A minimal sender only needs to:
1. Open a TCP connection to the GUI's host:port.
2. Serialize one JSON object per event, append `\n`, and write the bytes.
3. Reconnect (with backoff) if the write fails -- the GUI may not be running
   yet, or may restart.
4. Never let a failed/absent connection block or crash the caller -- display
   is decorative, not part of the core recognition/Q&A pipeline.

No client library, schema compiler, or additional dependency is required.
`display_protocol.py` in this repo is the Python reference implementation of
message construction (not a required dependency for other languages -- it's
just documentation-as-code).
