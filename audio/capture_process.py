"""
Audio capture subprocess

Runs as an independent process, responsible only for:
  - Microphone capture + VAD segmentation
  - WASAPI loopback capture + VAD segmentation

Fully segmented speech chunks are passed to the main process via an inter-process queue.

Design goal: isolate latency-sensitive audio capture from the main process's GPU
inference (whisper/LLM), so that GIL/scheduling jitter from heavy computation in the
main process doesn't cause the WASAPI loopback buffer to overflow and drop frames.
"""

import time
import queue
import threading
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CaptureConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_duration: float = 0.5
    silence_threshold: float = 0.02       # baseline threshold for system-output (loopback) detection (multiplied by 0.5 internally)
    mic_silence_threshold: float = 0.05   # separate mic threshold, raised to filter ambient-noise false triggers
    silence_duration: float = 0.6     # English speech is faster-paced, so pause detection is more sensitive (original 1.5s)
    min_record_duration: float = 0.5
    max_record_duration: float = 7    # sliding-window cap (mirrors whisper.cpp stream's length_ms idea, original 25s)
    input_device: int = None
    loopback_device: str = None
    loopback_enabled: bool = False
    mix_mode: str = "auto"
    streaming_enabled: bool = True    # streaming mode: periodically re-recognize the growing buffer while speech is ongoing
    streaming_interval: float = 1.0   # streaming re-recognition interval (seconds, mirrors whisper.cpp stream's step_ms idea, original 2.0s)


class _VadState:
    """Independent VAD state for a single audio source"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.record_start = None
        self.last_stream_emit = None


def _capture_worker(config_dict: dict, audio_out: mp.Queue, log_out: mp.Queue, stop_event):
    """
    Subprocess entry point: captures audio, does VAD segmentation, and puts fully
    segmented chunks into the audio_out queue.

    audio_out carries tuples: (source, samples_int16_bytes, num_samples)
      - transmitted as int16 bytes to avoid the pickle overhead of large float32 arrays
    log_out carries log strings: (level, message)
    """
    import signal
    # The subprocess ignores Ctrl+C -- exiting is entirely controlled by the main
    # process via stop_event, to avoid a KeyboardInterrupt interrupting a blocking
    # queue.get and raising an ugly traceback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import numpy as np
    import sounddevice as sd

    cfg = CaptureConfig(**config_dict)

    def log(level, msg):
        try:
            log_out.put_nowait((level, msg))
        except Exception:
            pass

    audio_queue = queue.Queue()
    states = {"mic": _VadState(), "loopback": _VadState()}

    def emit(buf, source, is_final):
        """Package up the current buffer and send it to the main process. is_final=False
        means this is a mid-utterance chunk (rolling re-recognition in streaming mode);
        is_final=True means this utterance has ended (pause or forced cut)."""
        if not buf:
            return
        audio_data = np.concatenate(buf, axis=0).flatten()
        duration = len(audio_data) / cfg.sample_rate
        if duration < cfg.min_record_duration:
            return
        if is_final:
            src_name = "system" if source == "loopback" else "microphone"
            log("info", f"🎙️ Recording finished: {duration:.1f}s [{src_name}]")
        pcm16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            audio_out.put((source, pcm16.tobytes(), len(pcm16), is_final))
        except Exception as e:
            log("error", f"Failed to transfer audio chunk: {e}")

    def finalize(buf, source):
        emit(buf, source, is_final=True)

    def process_block(source, block, state):
        volume = np.sqrt(np.mean(block**2))
        if source == "loopback":
            threshold = cfg.silence_threshold * 0.5
        else:
            threshold = cfg.mic_silence_threshold
        # Streaming re-recognition is now enabled for both mic and loopback:
        # transcribe mode only uses loopback; assist mode needs early feedback from
        # both mic (questions) and loopback (statement context)
        streaming = cfg.streaming_enabled

        if volume > threshold:
            if not state.is_speaking:
                state.is_speaking = True
                state.record_start = time.time()
                state.last_stream_emit = time.time()
                state.buffer = [block]
            else:
                state.buffer.append(block)
            state.silence_start = None
        else:
            if state.is_speaking:
                state.buffer.append(block)
                if state.silence_start is None:
                    state.silence_start = time.time()
                if time.time() - state.silence_start >= cfg.silence_duration:
                    log("debug", f"🔇 Pause detected (silence {cfg.silence_duration}s), natural segmentation")
                    finalize(state.buffer, source)
                    state.reset()
                    return

        if state.is_speaking:
            # Streaming mode: while speech is ongoing, repackage and re-send the full
            # current buffer (from the start of this utterance to now) every
            # streaming_interval seconds, so upstream code can do rolling
            # re-recognition + confirm the newly-stable prefix
            if streaming and (time.time() - state.last_stream_emit) >= cfg.streaming_interval:
                emit(state.buffer, source, is_final=False)
                state.last_stream_emit = time.time()

            if (time.time() - state.record_start) >= cfg.max_record_duration:
                log("info", "⏱️ Max recording duration reached, forcing a segment cut")
                finalize(state.buffer, source)
                state.reset()

    # Loopback capture thread (a thread inside this subprocess, fully isolated from the main process)
    def loopback_loop():
        import warnings
        try:
            import soundcard as sc
            from soundcard.mediafoundation import SoundcardRuntimeWarning
        except ImportError:
            log("error", "soundcard is not installed, loopback capture unavailable")
            return

        chunk = int(cfg.sample_rate * cfg.block_duration)

        def get_loopback_mic():
            """Get the loopback device that should currently be used (the one configured,
            or otherwise whatever tracks the system default output)"""
            if cfg.loopback_device:
                return sc.get_microphone(cfg.loopback_device, include_loopback=True)
            return sc.get_microphone(sc.default_speaker().name, include_loopback=True)

        current_device_name = None
        DEVICE_CHECK_INTERVAL = 5.0   # actually check every 5 seconds, to avoid calling the WASAPI enumeration API too often
        PENDING_CONFIRM_NEEDED = 2    # only switch after seeing the same new device 2 times in a row, to filter out transient false positives

        while not stop_event.is_set():
            try:
                mic = get_loopback_mic()
                if mic.name != current_device_name:
                    if current_device_name is not None:
                        log("info", f"System default output switched: {current_device_name} -> {mic.name}")
                    else:
                        log("info", f"System output loopback enabled: {mic.name}")
                    current_device_name = mic.name

                discontinuity = 0
                last_check = time.time()
                pending_device_name = None
                pending_count = 0
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", SoundcardRuntimeWarning)
                    with mic.recorder(samplerate=cfg.sample_rate, channels=1, blocksize=chunk * 4) as rec:
                        while not stop_event.is_set():
                            now = time.time()
                            if now - last_check >= DEVICE_CHECK_INTERVAL:
                                last_check = now
                                try:
                                    new_mic = get_loopback_mic()
                                    if new_mic.name != current_device_name:
                                        if new_mic.name == pending_device_name:
                                            pending_count += 1
                                        else:
                                            pending_device_name = new_mic.name
                                            pending_count = 1
                                        if pending_count >= PENDING_CONFIRM_NEEDED:
                                            log("info", "Output device switch detected, reconnecting...")
                                            break  # exit the inner loop and reconnect to the new device
                                    else:
                                        pending_device_name = None
                                        pending_count = 0
                                except Exception:
                                    pass

                            caught.clear()
                            data = rec.record(numframes=chunk)
                            if caught:
                                discontinuity += 1
                                if discontinuity % 50 == 1:
                                    log("debug", f"Loopback audio frame drop (total {discontinuity}), block discarded")
                                continue
                            if data.ndim == 1:
                                data = data.reshape(-1, 1)
                            audio_queue.put(("loopback", data.astype(np.float32)))

            except Exception as e:
                log("warning", f"Loopback capture error, retrying in 2s: {e}")
                time.sleep(2)

    def mic_callback(indata, frames, timestamp, status):
        if status:
            log("warning", f"Microphone status warning: {status}")
        audio_queue.put(("mic", indata.copy()))

    loopback_thread = None
    if cfg.loopback_enabled and cfg.mix_mode != "mic":
        loopback_thread = threading.Thread(target=loopback_loop, daemon=True)
        loopback_thread.start()
        log("info", "🎤 Listening on microphone + system output (meeting/video mode)")
    else:
        log("info", "🎤 Listening on microphone")

    mic_stream = None
    try:
        mic_stream = sd.InputStream(
            device=cfg.input_device,
            channels=cfg.channels,
            samplerate=cfg.sample_rate,
            blocksize=int(cfg.sample_rate * cfg.block_duration),
            callback=mic_callback,
        )
        mic_stream.start()

        while not stop_event.is_set():
            try:
                source, block = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if cfg.mix_mode == "loopback" and source == "mic":
                continue
            if cfg.mix_mode == "mic" and source == "loopback":
                continue
            process_block(source, block, states[source])

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log("error", f"Audio capture error: {e}")
    finally:
        if mic_stream:
            try:
                mic_stream.stop()
                mic_stream.close()
            except Exception:
                pass
        log("info", "🛑 Capture process stopped")


class AudioCaptureProcess:
    """
    Audio capture process manager (runs in the main process).

    Runs the capture logic in an independent process, receiving segmented speech
    chunks via a queue.
    Usage:
        cap = AudioCaptureProcess(config, on_speech_end=callback)
        cap.start()
        ...
        cap.stop()
    """

    def __init__(self, config: CaptureConfig = None, on_speech_end=None, logger=None):
        self.config = config or CaptureConfig()
        self.on_speech_end = on_speech_end
        self.logger = logger
        self._ctx = mp.get_context("spawn")  # must use spawn on Windows
        self._audio_out = self._ctx.Queue(maxsize=32)
        self._log_out = self._ctx.Queue(maxsize=256)
        self._stop_event = self._ctx.Event()
        self._process = None
        self._consumer_thread = None
        self._log_thread = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        self._process = self._ctx.Process(
            target=_capture_worker,
            args=(asdict(self.config), self._audio_out, self._log_out, self._stop_event),
            name="agent-audio-capture",
            daemon=True,
        )
        self._process.start()

        # One unbounded local queue + one persistent worker thread per source
        # (mic/loopback), processed sequentially and never dropped. When processing
        # is slow the queue just grows (higher latency), but content is never lost
        # the way a "lock + drop" approach would lose utterances.
        self._local_queues = {"mic": queue.Queue(), "loopback": queue.Queue()}
        self._worker_threads = {}
        for src, q in self._local_queues.items():
            t = threading.Thread(target=self._worker_loop, args=(src, q), daemon=True, name=f"audio-worker-{src}")
            t.start()
            self._worker_threads[src] = t

        # Main process side: quickly move items from the cross-process queue to the
        # local queue (just moving data, no heavy processing, to avoid slowing down
        # the subprocess's capture loop)
        self._consumer_thread = threading.Thread(target=self._consume_audio, daemon=True, name="audio-consumer")
        self._consumer_thread.start()

        # Main process side: forward subprocess logs
        self._log_thread = threading.Thread(target=self._consume_logs, daemon=True, name="audio-log-forward")
        self._log_thread.start()

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None

    def _consume_audio(self):
        """Quickly move items from the cross-process queue to the local queue, without
        any heavy processing, to guarantee the subprocess's capture loop is never blocked"""
        while self._running:
            try:
                source, pcm16_bytes, num_samples, is_final = self._audio_out.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            except Exception:
                continue
            samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            local_q = self._local_queues.get(source)
            if local_q is not None:
                if not is_final:
                    # Streaming intermediate chunk: if the queue head still has an
                    # unprocessed intermediate chunk waiting, drop it right away
                    # (the new chunk is a more complete rolling buffer for the same
                    # utterance, so the old one is already stale) to avoid backlog
                    # slowing down the latest result. Use the queue's own lock to
                    # avoid a race with the worker thread's get(); stop as soon as we
                    # hit a final chunk or the queue is empty -- a final chunk is
                    # never dropped.
                    with local_q.mutex:
                        while local_q.queue and not local_q.queue[0][1]:
                            local_q.queue.popleft()
                local_q.put((samples, is_final))

    def _worker_loop(self, source, local_q):
        """Persistent per-source worker: sequentially pops chunks and calls the
        callback, never drops a final chunk; if processing is slow it just queues up"""
        while self._running:
            try:
                samples, is_final = local_q.get(timeout=0.2)
            except queue.Empty:
                continue
            backlog = local_q.qsize()
            if backlog > 0 and self.logger:
                self.logger.debug(f"[{source}] {backlog} chunk(s) backlogged, processing")
            if self.on_speech_end:
                try:
                    self.on_speech_end(source, samples, is_final)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Speech callback error: {e}")

    def _consume_logs(self):
        while self._running:
            try:
                level, msg = self._log_out.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            except Exception:
                continue
            if self.logger:
                getattr(self.logger, level, self.logger.info)(msg)
