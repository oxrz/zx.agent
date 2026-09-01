"""
Speech recognition module (remote streaming recognition client)

No longer runs a model locally. Instead, connects over TCP to a remote
SimulStreaming service (deployed on a dedicated GPU server, running Whisper
large-v3 + AlignAtt streaming decoding).

Protocol (see SimulStreaming's whisper_server.py):
  - Client -> server: raw PCM16 audio bytes (16kHz mono, little-endian). No framing
    or length prefix needed -- just keep sending; the server handles VAC (voice
    activity detection) and buffering internally.
  - Server -> client: one JSON object per line, with fields:
      start/end   : timestamps (seconds, estimated) this output corresponds to
      text        : the newly confirmed text increment for this update (not the
                    full text accumulated from the start)
      words       : word-level timestamps
      is_final    : whether this marks the end of an utterance (VAC-detected silence)
      emission_time: time (seconds) from connection establishment to this output, on the server side

Design notes:
  - The server maintains its own audio buffer state internally (it is not stateless,
    one-shot inference), so the client only needs to send the newly added audio -- it
    must not resend the whole buffer the way the old local approach did.
  - Each source (mic/loopback) uses one persistent TCP connection, sending
    incrementally via send_incremental(); internally it tracks "how far we've sent"
    per source, and resets on is_final.
  - A short burst of silence is sent periodically while nothing else is being sent,
    to keep the connection alive -- see _keepalive_loop for why this is required
    rather than merely nice to have.
  - Automatically reconnects on network/server errors, so a single hiccup doesn't
    kill the whole transcription session.
"""

import io
import json
import socket
import threading
import time

import numpy as np

from utils.logger import logger

# Magic prefix of the optional per-connection configuration header, which must match
# the server's HANDSHAKE_MAGIC in whisper_server.py. The server recognizes the header
# by this prefix alone and treats anything else as audio, which is what makes sending
# it safe against an older server: an older build has no header parsing at all, so
# these bytes would be decoded as a fraction of a second of noise and nothing worse.
SESSION_HEADER_MAGIC = b"SSCFG1 "


class RemoteSTTClient:
    """A persistent streaming connection from a single source (e.g. loopback) to the remote SimulStreaming service"""

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
        session_prompt: str | None = None,
        send_header: bool = True,
    ):
        """
        Args:
            host: remote STT service address
            port: remote STT service port
            source: audio source label ("mic" | "loopback"), used only for logging/callback distinction
            on_result: callback on_result(source: str, result: dict), invoked when a
                       line of JSON result arrives, called from the internal reader
                       thread -- the callback must handle its own thread safety
            sample_rate: sample rate, must match the server (default 16000)
            reconnect_delay: seconds to wait before reconnecting after a disconnect
            connect_timeout: timeout in seconds for a single connection attempt
            keepalive_interval: send a short burst of silence if nothing has been sent
                       for this long, to stop the server from closing an idle
                       connection (see _keepalive_loop). Must stay comfortably below
                       the server's --client-timeout, which defaults to 300s.
                       0 disables it.
            is_healthy: optional callable returning whether our own audio capture is
                       still working. Gates the keepalive -- see _keepalive_loop for
                       why sending it unconditionally would be actively harmful.
                       None means "assume healthy".
            session_prompt: optional text describing what this recording is about,
                       sent once per connection so the server conditions its decoder
                       on it. Only worth setting when the subject and the proper
                       nouns in it are known in advance -- a prompt that does not
                       match the audio measurably makes recognition worse, not just
                       no better. None or "" sends no header at all, leaving whatever
                       the server was launched with. See config/trans.yaml.
        """
        self.host = host
        self.port = port
        self.source = source
        self.on_result = on_result
        self.sample_rate = sample_rate
        self.reconnect_delay = reconnect_delay
        self.connect_timeout = connect_timeout
        self.keepalive_interval = keepalive_interval
        self.is_healthy = is_healthy
        self.session_prompt = session_prompt or None
        self.send_header = send_header

        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._sent_samples = 0  # sample position the current utterance has been sent up to (for incremental sending)
        self._last_send_ts = 0.0  # when we last put any bytes on the wire, for keepalive accounting
        self._silence = np.zeros(int(0.1 * sample_rate), dtype=np.int16).tobytes()
        self._running = False
        self._reader_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._connected = threading.Event()

    def start(self):
        """Start the background connect+read thread"""
        if self._running:
            return
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._connection_loop, daemon=True, name=f"stt-remote-{self.source}"
        )
        self._reader_thread.start()
        if self.keepalive_interval:
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, daemon=True, name=f"stt-keepalive-{self.source}"
            )
            self._keepalive_thread.start()

    def stop(self):
        self._running = False
        self._connected.clear()
        with self._send_lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
        if self._reader_thread:
            self._reader_thread.join(timeout=3)
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=3)

    def _connection_loop(self):
        """Keep trying to connect, reconnecting after disconnects, until stop() is called"""
        while self._running:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout
                )
                sock.settimeout(None)  # subsequent reads can block; we rely on the peer closing/erroring to exit
                with self._send_lock:
                    if self.send_header:
                        self._send_session_header(sock)
                    self._sock = sock
                    self._sent_samples = 0
                    # Count the keepalive window from the moment we connect, not from
                    # process start, so a fresh connection doesn't immediately emit one.
                    self._last_send_ts = time.time()
                self._connected.set()
                logger.info(f"[{self.source}] Connected to remote speech recognition service {self.host}:{self.port}")
                self._read_loop(sock)
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                logger.warning(f"[{self.source}] Failed to connect to remote speech recognition service: {e}, retrying in {self.reconnect_delay}s")
            finally:
                self._connected.clear()
                with self._send_lock:
                    self._sock = None
            if self._running:
                time.sleep(self.reconnect_delay)

    def _send_session_header(self, sock: socket.socket):
        """Send the per-connection configuration header, if a prompt is configured.

        Sent on every connection, including reconnects: the server rebuilds its
        decoder context per connection and resets any key it is not told about, so a
        reconnect mid-meeting would otherwise silently lose the prompt.

        Failures here are raised, not swallowed -- the caller's reconnect path
        handles them. Sending audio to a server that never got the header would work
        but transcribe under the wrong conditioning, which is harder to notice than
        a reconnect.
        """
        if not self.session_prompt:
            return
        header = (
            SESSION_HEADER_MAGIC
            + json.dumps({"static_init_prompt": self.session_prompt}).encode("utf-8")
            + b"\n"
        )
        sock.sendall(header)
        logger.info(
            f"[{self.source}] Sent session prompt to the recognition server "
            f"({len(self.session_prompt)} chars): {self.session_prompt!r}"
        )

    def _read_loop(self, sock: socket.socket):
        """Continuously read line-delimited JSON results from the server until the connection closes or errors"""
        buffer = b""
        try:
            while self._running:
                data = sock.recv(4096)
                if not data:
                    logger.info(f"[{self.source}] Remote speech recognition service connection closed")
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._handle_line(line)
        except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
            logger.warning(f"[{self.source}] Remote speech recognition connection error: {e}")

    def _handle_line(self, line: bytes):
        t_recv = time.time()
        try:
            result = json.loads(line.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            logger.debug(f"[{self.source}] Unparseable result line: {line[:200]!r}")
            return
        # emission_time is the server-side elapsed time from connection establishment
        # to producing this result (recorded in the server's process() loop). We also
        # stamp the local time we received it here, so it's easy to line up the
        # server log against emission_time and figure out how much of the delay is
        # server-side processing vs. network + client latency.
        logger.debug(
            f"[{self.source}] Received recognition result: is_final={result.get('is_final')} "
            f"emission_time={result.get('emission_time')} "
            f"text={result.get('text', '')!r} recv_ts={t_recv:.3f}"
        )
        if self.on_result:
            try:
                self.on_result(self.source, result)
            except Exception as e:
                logger.error(f"[{self.source}] Error in recognition result callback: {e}")

    def _keepalive_loop(self):
        """Send a short burst of silence whenever nothing else has been sent for
        keepalive_interval seconds.

        This is required, not cosmetic. The server closes any connection that has
        received no audio for --client-timeout seconds (300 by default) -- it serves
        one client at a time, so it cannot let a connected-but-mute client block
        every other one. But we only transmit while our own VAD reports speech, so a
        session where nobody happens to be talking sends literally zero bytes and
        looks exactly like a client whose capture died. Getting dropped for that is
        bad in a specific way: send_incremental discards audio while disconnected,
        so the reconnect_delay seconds after the drop are a hole, and the thing most
        likely to happen right after a long silence is someone starting to speak.

        Silence is safe to send: the client already transmits the trailing silent
        blocks of every utterance (that is what lets the server's VAC detect
        end-of-speech at all), so this is nothing the server does not routinely
        handle -- its VAD sees no speech and runs no ASR, costing no GPU work.

        But it must NOT be sent unconditionally, and this is the subtle part. The
        server's timeout exists to evict a client whose capture died while its socket
        stayed healthy -- otherwise that client blocks every other one, which is
        exactly the failure that once wedged the server for hours. From the server's
        side "quiet room" and "dead capture" are indistinguishable: both send
        silence, or nothing. So keeping the connection warm regardless of our own
        state would defeat the eviction entirely and reinstate that bug. We are the
        only side that can tell the two apart, so the keepalive is gated on
        is_healthy(): if our capture is gone, we deliberately stop keeping the
        connection alive and let the server reclaim it.

        Note the gate is only as good as its signal -- it catches a dead capture
        process, not a live one whose device thread has wedged while still being
        polled. That narrower case remains uncovered on both sides.
        """
        while self._running:
            time.sleep(1.0)
            if not self._running:
                break
            if self.is_healthy is not None:
                try:
                    healthy = self.is_healthy()
                except Exception as e:
                    # A failing health check is not evidence of health.
                    logger.debug(f"[{self.source}] Keepalive health check raised: {e}")
                    healthy = False
                if not healthy:
                    continue
            with self._send_lock:
                sock = self._sock
                if sock is None:
                    continue
                # _sent_samples > 0 means an utterance is in flight; never inject
                # silence into the middle of one. An utterance in progress is also
                # sending regularly, so the interval below would not have elapsed.
                if self._sent_samples > 0:
                    continue
                if time.time() - self._last_send_ts < self.keepalive_interval:
                    continue
                try:
                    sock.sendall(self._silence)
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    # Let the reader thread drive the reconnect; just drop the socket
                    # here, same as send_incremental does.
                    logger.warning(f"[{self.source}] Keepalive send failed: {e}")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    continue
                self._last_send_ts = time.time()
                logger.debug(
                    f"[{self.source}] Sent keepalive silence "
                    f"({len(self._silence) // 2 / self.sample_rate:.2f}s) after "
                    f"{self.keepalive_interval:.0f}s idle"
                )

    def send_incremental(self, full_buffer: np.ndarray, is_final: bool):
        """
        Send an audio increment.

        Args:
            full_buffer: the complete audio buffer for the current utterance so far
                         (float32, [-1, 1]), kept growing by capture_process.py's
                         existing logic; this method only slices off "everything after
                         the last sent position" and sends that, to avoid resending
                         the whole utterance.
            is_final: whether this utterance has ended (per capture_process's VAD
                       decision). The send position is reset afterward, ready for the
                       next utterance.
        """
        with self._send_lock:
            sock = self._sock
            if sock is None:
                # Silently drop while disconnected (after reconnecting, just start
                # sending from the next utterance -- we don't cache and replay,
                # to avoid audio queued up during the outage causing timestamp
                # confusion once the connection recovers)
                if is_final:
                    self._sent_samples = 0
                return

            new_samples = full_buffer[self._sent_samples:]
            if len(new_samples) > 0:
                pcm16 = (np.clip(new_samples, -1.0, 1.0) * 32767).astype(np.int16)
                chunk_seconds = len(new_samples) / self.sample_rate
                t_send_start = time.time()
                try:
                    sock.sendall(pcm16.tobytes())
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    logger.warning(f"[{self.source}] Failed to send audio: {e}")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    return
                send_elapsed = time.time() - t_send_start
                logger.debug(
                    f"[{self.source}] Sent audio {chunk_seconds:.3f}s ({len(pcm16)} samples), "
                    f"is_final={is_final}, sendall took={send_elapsed:.4f}s, "
                    f"total sent so far={len(full_buffer) / self.sample_rate:.3f}s"
                )
                self._sent_samples = len(full_buffer)
                self._last_send_ts = time.time()

            if is_final:
                logger.debug(f"[{self.source}] Utterance ended (is_final), resetting send position")
                self._sent_samples = 0

    @property
    def connected(self) -> bool:
        return self._connected.is_set()


class SpeechRecognizer:
    """
    Manages remote recognition connections for multiple sources, replacing the old
    local model wrapper.

    Each source (currently only loopback, mic support is planned) has its own
    independent remote connection (the SimulStreaming server handles one connection
    at a time, so multiple sources each need their own host:port).

    Usage kept similar to the old interface, so main.py needs minimal changes:
        stt = SpeechRecognizer(sources={"loopback": ("<remote recognition service address>", 45678)}, on_result=callback)
        stt.start()
        stt.feed("loopback", audio_buffer, is_final)
        ...
        stt.stop()
    """

    def __init__(self, sources: dict[str, tuple[str, int]], on_result=None, sample_rate: int = 16000,
                 is_healthy=None, session_prompt: str | None = None, send_header: bool = True):
        """
        Args:
            sources: {source_name: (host, port)}, e.g. {"loopback": ("<remote recognition service address>", 45678)}
            on_result: callback on_result(source, result_dict), invoked when a remote recognition result arrives
            sample_rate: sample rate, must match the server's config
            is_healthy: optional callable returning whether audio capture is still
                       working; forwarded to each connection to gate its keepalive
                       (see RemoteSTTClient._keepalive_loop)
            session_prompt: optional description of what is being recorded, forwarded
                       to every connection (see RemoteSTTClient)
            send_header: whether to send a per-connection configuration header on connect.
                       Set to False when connecting to servers that don't support the
                       SESSION_HEADER_MAGIC protocol (e.g. whisper.cpp stream_server).
        """
        self.sample_rate = sample_rate
        self._clients: dict[str, RemoteSTTClient] = {
            source: RemoteSTTClient(
                host=host, port=port, source=source, on_result=on_result, sample_rate=sample_rate,
                is_healthy=is_healthy, session_prompt=session_prompt,
                send_header=send_header,
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
        """Send the current full buffer for this utterance from the given source to its remote connection (incrementally)"""
        client = self._clients.get(source)
        if client is None:
            return
        client.send_incremental(full_buffer, is_final)

    def is_connected(self, source: str) -> bool:
        client = self._clients.get(source)
        return client.connected if client else False
