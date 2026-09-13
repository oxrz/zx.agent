"""
Speech recognition module (WebSocket streaming client)

Connects to the ASR POC server over WebSocket, streams PCM16 audio as
binary frames, and receives partial (Zipformer CPU) and final (OpenVINO
Whisper GPU) recognition results.

Protocol (v1, see asr_server.py on the server):
  - Client sends a JSON text frame first:
    {"type":"start","session_id":"...","audio":{"encoding":"pcm_s16le","sample_rate":16000,"channels":1}}
  - Client sends binary frames: uint32 seq (LE) + uint64 timestamp_ms (LE) + int16 PCM payload
  - Server replies with JSON text frames:
      type "ready"   : server accepted the session and loaded models
      type "partial" : streaming partial from Zipformer (CPU)
      type "final"   : sentence-end correction from OpenVINO Whisper (GPU)
      type "closed"  : session ended
  - Client sends {"type":"stop"} to flush and close

Design:
  - Threading model matches the rest of zAgent: a reader thread receives
    messages, audio is sent from the capture thread via send_incremental.
  - Utterance tracking: the server assigns utterance_ids.  When the id
    advances in a partial, we emit a synthetic is_final for the previous
    utterance so main.py can commit it and start a new line.
  - Results carry both ``text`` (delta for terminal printing) and
    ``full_text`` (authoritative accumulated text for the GUI / LLM).
"""

from __future__ import annotations

import json
import re
import struct
import threading
import time
import uuid

import numpy as np

from utils.logger import logger


class RemoteSTTClient:
    """WebSocket streaming connection from one audio source to the ASR server."""

    def __init__(
        self,
        host: str,
        port: int,
        source: str,
        on_result=None,
        sample_rate: int = 16000,
        reconnect_delay: float = 2.0,
        connect_timeout: float = 5.0,
        keepalive_interval: float = 60.0,
        is_healthy=None,
    ):
        self.url = f"ws://{host}:{port}/v1/stream"
        self.source = source
        self.on_result = on_result
        self.sample_rate = sample_rate
        self.reconnect_delay = reconnect_delay
        self.connect_timeout = connect_timeout
        self.keepalive_interval = keepalive_interval
        self.is_healthy = is_healthy

        self._ws = None
        self._send_lock = threading.Lock()
        self._sent_samples = 0
        self._seq = 0
        self._last_send_ts = 0.0
        self._running = False
        self._reader_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._connected = threading.Event()

        self._current_uid: int | None = None
        self._pending_uids: set[int] = set()
        self._finalized_uids: set[int] = set()
        self._final_texts: dict[int, str] = {}
        self._pending_texts: dict[int, str] = {}
        self._partial_text = ""
        # audio_end_ms of the newest partial already shown for the current utterance;
        # -1 = nothing shown yet. Absolute session time, so it never runs backwards.
        self._display_end_ms = -1

    # -- lifecycle --------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._connection_loop, daemon=True, name=f"stt-ws-{self.source}"
        )
        self._reader_thread.start()
        if self.keepalive_interval:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True, name=f"stt-ka-{self.source}"
            )
            self._keepalive_thread.start()

    def stop(self):
        self._running = False
        self._connected.clear()
        with self._send_lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
        if self._reader_thread:
            self._reader_thread.join(timeout=3)
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=3)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    # -- connection -------------------------------------------------------

    def _connection_loop(self):
        import websocket as ws_lib

        while self._running:
            try:
                ws = ws_lib.WebSocket()
                ws.settimeout(self.connect_timeout)
                ws.connect(self.url)
                ws.settimeout(None)

                session_id = f"{self.source}-{uuid.uuid4().hex[:8]}"
                ws.send(json.dumps({
                    "type": "start",
                    "session_id": session_id,
                    "audio": {
                        "encoding": "pcm_s16le",
                        "sample_rate": self.sample_rate,
                        "channels": 1,
                    },
                }))

                with self._send_lock:
                    self._ws = ws
                    self._sent_samples = 0
                    self._seq = 0
                    self._last_send_ts = time.time()
                    self._current_uid = None
                    self._pending_uids = set()
                    self._finalized_uids = set()
                    self._pending_texts = {}
                    self._partial_text = ""
                    self._display_end_ms = -1
                self._connected.set()
                logger.info(
                    f"[{self.source}] Connected to ASR server {self.url} "
                    f"(session {session_id})"
                )
                self._read_loop(ws)
            except Exception as e:
                logger.warning(
                    f"[{self.source}] Connection failed: {e}, "
                    f"retrying in {self.reconnect_delay}s"
                )
            finally:
                self._connected.clear()
                with self._send_lock:
                    self._ws = None
            if self._running:
                time.sleep(self.reconnect_delay)

    def _read_loop(self, ws):
        import websocket as ws_lib

        try:
            while self._running:
                try:
                    data = ws.recv()
                except ws_lib.WebSocketTimeoutException:
                    continue
                if not data:
                    logger.info(f"[{self.source}] Server closed connection")
                    break
                if isinstance(data, str):
                    try:
                        self._handle_message(json.loads(data))
                    except json.JSONDecodeError:
                        logger.debug(
                            f"[{self.source}] Unparseable message: {data[:200]!r}"
                        )
        except (
            ws_lib.WebSocketConnectionClosedException,
            ConnectionResetError,
            OSError,
        ) as e:
            logger.warning(f"[{self.source}] Connection error: {e}")

    # -- message handling -------------------------------------------------

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "ready":
            logger.info(
                f"[{self.source}] Server ready, engines: {msg.get('engines', [])}"
            )
            return

        if msg_type == "partial":
            self._on_partial(msg)
            return

        if msg_type == "final":
            self._on_final(msg)
            return

        if msg_type == "closed":
            logger.info(f"[{self.source}] Session closed")
            return

        if msg_type in ("error", "warning", "finalizer_unavailable"):
            detail = msg.get("message") or msg.get("reason", "")
            logger.warning(f"[{self.source}] Server {msg_type}: {detail}")
            return

        logger.debug(f"[{self.source}] Unknown message type: {msg_type}")

    def _on_partial(self, msg: dict):
        uid = msg.get("utterance_id", 0)
        text = msg.get("text", "")
        end_ms = msg.get("audio_end_ms", -1)

        if uid in self._finalized_uids:
            return

        # Finalizer interim jobs are queued behind the live Zipformer stream and
        # can arrive after a newer utterance has already started.  Such a partial
        # belongs to an older utterance; treating it as a new current UID rolls the
        # client state backwards and the next final can clear the newer text from
        # the GUI.  Final messages for older UIDs are still accepted below as
        # pending corrections, so only stale partials are discarded here.
        if self._current_uid is not None and uid < self._current_uid:
            logger.debug(
                f"[{self.source}] Ignoring stale partial uid={uid}; "
                f"current uid={self._current_uid}"
            )
            return

        if self._current_uid is not None and uid != self._current_uid:
            old_uid = self._current_uid
            if self._partial_text:
                self._pending_texts[old_uid] = self._partial_text
            self._pending_uids.add(old_uid)
            self._partial_text = ""
            self._display_end_ms = -1
            # fallback: if uid=old-1 is still pending (Whisper never arrived), commit it now
            fallback_uid = old_uid - 1
            if fallback_uid in self._pending_uids and fallback_uid not in self._finalized_uids:
                fb_text = self._pending_texts.pop(fallback_uid, "")
                self._pending_uids.discard(fallback_uid)
                if fb_text:
                    # fallback_uid is old_uid - 1, so this is always a past
                    # utterance and a newer one is already streaming. Mark it as
                    # a correction: without the flag main.py/overlay take the
                    # normal final path and wipe the partial slot, which blanks
                    # the sentence currently on screen until its own final lands.
                    self._emit(
                        "", is_final=True, full_text=fb_text,
                        engine="zipformer-fallback", pending_correction=True,
                        utterance_id=fallback_uid, replace_utterance_id=fallback_uid,
                    )

        self._current_uid = uid

        # Drop a partial that covers LESS audio than what is already on screen for
        # this utterance. A Whisper interim is transcribed from a snapshot and
        # routinely loses the race with the faster Zipformer stream, so displaying
        # it would roll the text backwards. Compare spans, not string lengths: a
        # correction is often shorter than the raw text it replaces, and a length
        # test also mistakes the start of a new utterance -- short by definition --
        # for a regression, which suppresses the whole sentence until it outgrows
        # the previous one.
        if end_ms >= 0 and end_ms < self._display_end_ms:
            return

        prev = self._partial_text
        self._partial_text = text
        if end_ms >= 0:
            self._display_end_ms = end_ms

        if text.startswith(prev):
            delta = text[len(prev):]
        else:
            delta = text

        if delta or text != prev:
            self._emit(
                delta, is_final=False, full_text=text,
                engine=msg.get("engine", ""), utterance_id=uid,
            )

    def _on_final(self, msg: dict):
        uid = msg.get("utterance_id", 0)
        text = msg.get("text", "")
        latency = msg.get("latency_ms")
        raw_text = text

        is_current = (uid == self._current_uid)
        # For a delayed correction the live partial belongs to a newer UID;
        # compare against the partial saved when this older UID rolled over.
        reference_partial = self._partial_text if is_current else self._pending_texts.get(uid, "")

        # Whisper finalization is normally authoritative, but a delayed or
        # badly segmented final can occasionally be only a suffix of the live
        # Zipformer text (or a decoder hallucination such as Rrrrr...).  Never
        # let such a result erase a more complete sentence already shown.
        partial = reference_partial
        normalized_partial = self._normalize_for_compare(partial)
        normalized_final = self._normalize_for_compare(text)
        merged = self._merge_suffix_correction(partial, text)
        if merged is not None:
            # If the apparent overlap is only look-ahead inside the previous
            # final (rather than its boundary), discard the merge and keep the
            # finalizer text on its own; otherwise a phrase such as ``I do``
            # can be attached to the wrong sentence.
            trimmed = self._trim_final_overlap(uid, merged)
            if trimmed is None:
                merged = None
            else:
                merged = trimmed
        if merged is not None:
            logger.info(
                f"[{self.source}] Restoring omitted prefix for uid={uid} "
                f"from Zipformer partial"
            )
            text = merged
            normalized_final = self._normalize_for_compare(text)
        elif partial and self._final_is_suspicious(normalized_partial, normalized_final):
            logger.warning(
                f"[{self.source}] Keeping Zipformer partial for uid={uid}; "
                f"suspicious Whisper final={text!r} partial={partial!r}"
            )
            text = partial

        self._finalized_uids.add(uid)
        self._final_texts[uid] = text
        self._pending_uids.discard(uid)
        self._pending_texts.pop(uid, None)

        if len(self._finalized_uids) > 100:
            cutoff = max(self._finalized_uids) - 50
            self._finalized_uids = {u for u in self._finalized_uids if u >= cutoff}
            self._final_texts = {
                u: value for u, value in self._final_texts.items() if u >= cutoff
            }

        if is_current:
            self._current_uid = None
            self._partial_text = ""
            self._display_end_ms = -1

        self._emit(
            "", is_final=True, full_text=text,
            engine=msg.get("engine", ""),
            latency_ms=latency,
            pending_correction=(not is_current),
            utterance_id=uid,
            replace_utterance_id=msg.get("replace_utterance_id", uid),
            raw_final_text=raw_text if raw_text != text else None,
        )

    @staticmethod
    def _normalize_for_compare(text: str) -> str:
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        contractions = {
            "i'm": "i am", "i've": "i have", "i'd": "i would",
            "you're": "you are", "you'll": "you will", "you've": "you have",
            "it's": "it is", "they're": "they are", "they'll": "they will",
            "we're": "we are", "we'll": "we will", "don't": "do not",
            "can't": "can not", "won't": "will not", "didn't": "did not",
        }
        expanded = []
        for word in words:
            expanded.extend(contractions.get(word, word).split())
        return " ".join(expanded)

    @classmethod
    def _merge_suffix_correction(cls, partial: str, final: str) -> str | None:
        """Restore a Zipformer prefix when Whisper returned only a suffix.

        This happens when a finalizer snapshot starts after the beginning of a
        long utterance. Keep the prefix already heard, then use Whisper's text
        for the overlapping suffix (preserving its punctuation/casing).
        """
        p_words = cls._normalize_for_compare(partial).split()
        f_words = cls._normalize_for_compare(final).split()
        original_words = partial.strip().split()
        if len(p_words) < 8 or len(f_words) < 4:
            return None
        for start in range(1, len(p_words)):
            overlap = min(len(f_words), len(p_words) - start, 12)
            # A finalizer snapshot can differ by one insertion/deletion or a
            # small recognition error (for example Zipformer says ``ah in``
            # while Whisper says ``in``).  Compare a bounded overlap rather
            # than requiring byte-for-byte equality, but keep the threshold
            # high enough that unrelated repeated phrases are not joined.
            if overlap < 8:
                continue
            max_run = run = 0
            for left, right in zip(p_words[start:start + overlap], f_words[:overlap]):
                if left == right:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            if max_run >= 6:
                # Keep the Zipformer prefix's original casing and punctuation
                # in the displayed text; only the comparison uses normalized
                # words.  Contractions before the boundary are uncommon, but
                # clamp defensively if normalization expanded a token.
                prefix = " ".join(original_words[:min(start, len(original_words))])
                return f"{prefix} {final.strip()}".strip()
        return None

    def _trim_final_overlap(self, uid: int, text: str) -> str | None:
        """Avoid repeating words already committed by the preceding utterance.

        Zipformer can carry a few boundary words into the next utterance while
        a Whisper finalizer snapshot starts after them.  The suffix-restoration
        step intentionally brings those words back; remove only the part that
        is already present in the most recent lower-UID final.
        """
        previous = [
            (other_uid, value)
            for other_uid, value in self._final_texts.items()
            if other_uid < uid and value.strip()
        ]
        if not previous:
            return text
        _, prior = max(previous, key=lambda item: item[0])
        prior_words = re.findall(r"[a-z]+(?:'[a-z]+)?", prior.lower())
        text_words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        max_overlap = min(12, len(prior_words), len(text_words))
        overlap = 0
        match_end = 0
        for size in range(max_overlap, 1, -1):
            prefix = text_words[:size]
            # Usually this is the prior utterance's suffix (``...but this``),
            # but a finalizer may itself have included a little look-ahead
            # (``...I do very well`` before the next UID starts ``I do...``).
            # Search only the tail of the previous final so repeated phrases
            # elsewhere in a long transcript are not collapsed.
            tail = prior_words[-24:]
            for pos in range(len(tail) - size + 1):
                if tail[pos:pos + size] == prefix:
                    overlap = size
                    match_end = pos + size
                    break
            if overlap:
                break
        if overlap == 0:
            return text
        if match_end < len(tail):
            return None

        # The merged prefix is composed of ordinary whitespace-separated
        # tokens.  Drop the corresponding leading tokens while retaining the
        # finalizer's casing and punctuation for the rest of the sentence.
        tokens = text.strip().split()
        if overlap >= len(tokens):
            return ""
        return " ".join(tokens[overlap:]).strip()

    @staticmethod
    def _final_is_suspicious(partial: str, final: str) -> bool:
        if not partial or not final:
            return False
        # A repeated-character hallucination is never useful as a correction.
        if re.fullmatch(r"([a-z])\1{8,}", final.replace(" ", "")):
            return True
        partial_words = partial.split()
        final_words = final.split()
        if len(final_words) < 3:
            return False
        # If the final is wholly contained in the already displayed partial
        # and is materially shorter, it is a suffix/fragment, not a correction
        # for the complete utterance.
        if final in partial and len(final_words) * 1.35 < len(partial_words):
            return True
        # A final that starts deep inside the partial and has little shared
        # prefix would otherwise erase the beginning of a long sentence.
        common = 0
        for left, right in zip(partial_words, final_words):
            if left != right:
                break
            common += 1
        if len(partial_words) >= 12 and common <= 2 and len(final_words) * 1.35 < len(partial_words):
            return True
        return False

    def _emit(self, text: str, is_final: bool, full_text: str | None = None, **extra):
        if self.on_result is None:
            return
        result: dict = {"text": text, "is_final": is_final}
        if full_text is not None:
            result["full_text"] = full_text
        result.update(extra)
        try:
            self.on_result(self.source, result)
        except Exception as e:
            logger.error(f"[{self.source}] Callback error: {e}")

    # -- keepalive --------------------------------------------------------

    def _keepalive_loop(self):
        silence = np.zeros(int(0.1 * self.sample_rate), dtype=np.int16)
        while self._running:
            time.sleep(1.0)
            if not self._running:
                break
            if self.is_healthy is not None:
                try:
                    if not self.is_healthy():
                        continue
                except Exception:
                    continue
            with self._send_lock:
                ws = self._ws
                if ws is None or self._sent_samples > 0:
                    continue
                if time.time() - self._last_send_ts < self.keepalive_interval:
                    continue
                try:
                    self._send_binary(ws, silence)
                    logger.debug(f"[{self.source}] Sent keepalive silence")
                except Exception as e:
                    logger.warning(f"[{self.source}] Keepalive failed: {e}")
                    try:
                        ws.close()
                    except Exception:
                        pass
                    self._ws = None

    # -- audio sending ----------------------------------------------------

    def _send_binary(self, ws, samples_int16: np.ndarray):
        import websocket as ws_lib

        header = struct.pack(
            "<IQ", self._seq, int(self._sent_samples * 1000 / self.sample_rate)
        )
        ws.send(header + samples_int16.tobytes(), opcode=ws_lib.ABNF.OPCODE_BINARY)
        self._seq += 1
        self._last_send_ts = time.time()

    def send_incremental(self, full_buffer: np.ndarray, is_final: bool):
        with self._send_lock:
            ws = self._ws
            if ws is None:
                if is_final:
                    self._sent_samples = 0
                return

            new_samples = full_buffer[self._sent_samples:]
            if len(new_samples) > 0:
                pcm16 = (np.clip(new_samples, -1.0, 1.0) * 32767).astype(np.int16)
                try:
                    self._send_binary(ws, pcm16)
                except Exception as e:
                    logger.warning(f"[{self.source}] Send failed: {e}")
                    try:
                        ws.close()
                    except Exception:
                        pass
                    self._ws = None
                    return
                self._sent_samples = len(full_buffer)

            if is_final:
                self._sent_samples = 0


class SpeechRecognizer:
    """Manages WebSocket STT connections for multiple audio sources.

    WebSocket SpeechRecognizer used by the client; the public interface
    (start/stop/feed/is_connected) is unchanged.
    """

    def __init__(
        self,
        sources: dict[str, tuple[str, int]],
        on_result=None,
        sample_rate: int = 16000,
        is_healthy=None,
        **_kwargs,
    ):
        self.sample_rate = sample_rate
        self._clients: dict[str, RemoteSTTClient] = {
            source: RemoteSTTClient(
                host=host,
                port=port,
                source=source,
                on_result=on_result,
                sample_rate=sample_rate,
                is_healthy=is_healthy,
            )
            for source, (host, port) in sources.items()
        }

    def start(self):
        for client in self._clients.values():
            client.start()

    def stop(self):
        for client in self._clients.values():
            client.stop()

    def feed(self, source: str, full_buffer: np.ndarray, is_final: bool):
        client = self._clients.get(source)
        if client is None:
            return
        client.send_incremental(full_buffer, is_final)

    def is_connected(self, source: str) -> bool:
        client = self._clients.get(source)
        return client.connected if client else False
