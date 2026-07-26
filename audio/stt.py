"""
zAgent speech recognition module (remote streaming recognition client)

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
        """
        self.host = host
        self.port = port
        self.source = source
        self.on_result = on_result
        self.sample_rate = sample_rate
        self.reconnect_delay = reconnect_delay
        self.connect_timeout = connect_timeout

        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._sent_samples = 0  # sample position the current utterance has been sent up to (for incremental sending)
        self._running = False
        self._reader_thread: threading.Thread | None = None
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

    def _connection_loop(self):
        """Keep trying to connect, reconnecting after disconnects, until stop() is called"""
        while self._running:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout
                )
                sock.settimeout(None)  # subsequent reads can block; we rely on the peer closing/erroring to exit
                with self._send_lock:
                    self._sock = sock
                    self._sent_samples = 0
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

    def __init__(self, sources: dict[str, tuple[str, int]], on_result=None, sample_rate: int = 16000):
        """
        Args:
            sources: {source_name: (host, port)}, e.g. {"loopback": ("<remote recognition service address>", 45678)}
            on_result: callback on_result(source, result_dict), invoked when a remote recognition result arrives
            sample_rate: sample rate, must match the server's config
        """
        self.sample_rate = sample_rate
        self._clients: dict[str, RemoteSTTClient] = {
            source: RemoteSTTClient(
                host=host, port=port, source=source, on_result=on_result, sample_rate=sample_rate
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
