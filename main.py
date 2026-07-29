#!/usr/bin/env python3
"""
Voice assistant (real-time English streaming recognition + Q&A assistance)

Usage:
    python main.py                            # start the voice assistant (always-on, default config/trans.yaml)
    python main.py --once                     # one-shot Q&A mode
    python main.py --list-devices             # list microphone devices
    python main.py --config my.yaml           # use a custom config

Modes (each started with its own independent config file; no runtime switching.
If you need both modes at the same time, start two separate processes with
different --config files, e.g. config/trans.yaml / config/assist.yaml):
    transcribe : pure real-time transcription of English audio, text output only, no Q&A
    assist     : real-time English transcription + answers only questions, with a
                 bilingual (Chinese/English) explanation; statements are buffered into
                 a rolling context (20 minutes by default) as background for answering questions
"""

import os
import sys
import time
import asyncio
import argparse
import signal
import threading
import platform
from pathlib import Path
import yaml

from utils.logger import logger, setup_logger
from audio.capture_process import AudioCaptureProcess, CaptureConfig
from audio.stt import SpeechRecognizer
from audio.tts import TextToSpeech
from ai.llm import LLMClient, LLMConfig
from memory import ContextBuffer


def set_high_performance():
    """Set the process to high-performance mode (Windows)"""
    if platform.system() != "Windows":
        return
    try:
        # Raise process priority to high
        import psutil
        proc = psutil.Process()
        proc.nice(psutil.HIGH_PRIORITY_CLASS)
        logger.info("Process priority set to HIGH")
    except ImportError:
        # Fall back to ctypes if psutil is not available
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST
        except Exception:
            pass
    except Exception:
        pass


def load_env_file(env_path=None):
    """Load a .env file into environment variables, without overwriting variables that already exist.

    Defaults to loading .env from the project root (next to .env.example) rather than
    some global path, to make sure that values filled in after `cp .env.example .env`
    actually get picked up.
    """
    if env_path is None:
        env_file = Path(__file__).resolve().parent / ".env"
    else:
        env_file = Path(env_path).expanduser()
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


class ZxAgent:
    def __init__(self, config_path="config/trans.yaml", log_level_override=None):
        self.config = self._load_config(config_path)
        self._running = False

        # The CLI's --log-level/-v takes priority over the config file's logging.level,
        # for quick ad-hoc debugging (e.g. to see how often partials get skipped /
        # inference time, without editing the yaml -- just run once with -v DEBUG)
        log_level = log_level_override or self.config.get("logging", {}).get("level", "INFO")
        log_file = self.config.get("logging", {}).get("file")
        setup_logger(level=log_level, log_file=log_file)

        # Run mode (English-only speech recognition; the two modes each use their own
        # independent config file, no runtime switching; if you need both modes at
        # the same time, just start two processes with different --config files):
        #   transcribe : pure real-time transcription of English audio, no Q&A
        #   assist     : real-time English transcription + answers only questions,
        #                with a bilingual explanation; statements are buffered for reference
        self.mode = self.config.get("mode", "transcribe")
        if self.mode not in ("transcribe", "assist"):
            logger.warning(f"Unknown mode '{self.mode}', falling back to transcribe")
            self.mode = "transcribe"

        # A persistent asyncio event loop (its own dedicated thread).
        # All LLM requests are submitted to this one loop, so the httpx client bound
        # to it can properly reuse its connection pool; this avoids the problem of
        # "creating/closing a new loop per request leaves the httpx client holding a
        # reference to a dead event loop, causing a permanent deadlock".
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="asyncio-loop"
        )
        self._loop_thread.start()

        # Remote recognition results are now confirmed incrementally natively by the
        # server (SimulStreaming + AlignAtt), so the client no longer needs to
        # maintain its own local-agreement state or serialize local inference calls
        # with a lock.
        # Tracks, per source, whether this utterance has already triggered the LLM
        # early because an incremental result ended in a question mark, so we don't
        # trigger it again once the utterance ends (is_final).
        self._question_fired = {"mic": False, "loopback": False}

        self._init_stt()
        self._init_llm()
        self._init_tts()
        self._init_context()
        self._init_listener()

        # Warm up the STT model
        logger.info("Warming up the speech recognition model...")
        self._warmup_stt()

        logger.info("agent initialization complete")

    def _run_event_loop(self):
        """Entry point for the persistent event loop thread"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _warmup_stt(self):
        """Start the remote recognition client's background connection thread
        (non-blocking; the actual connection keeps retrying in the background thread)"""
        self.stt.start()
        logger.info("Remote speech recognition client started (connecting in the background)")

    def _load_config(self, config_path):
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using default config")
            return {}
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Config file loaded: {config_path}")
        return config

    def _resolve_env_var(self, value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1])
        return value

    def _init_stt(self):
        """Remote speech recognition client: connects to the SimulStreaming service on
        a dedicated GPU server, keeps streaming audio, and receives incremental
        recognition results via the _on_remote_result callback.
        No longer loads any model locally, and no longer needs _stt_lock /
        local-agreement to simulate streaming.

        This version only supports registering a single audio source at startup; it
        does not support connecting both mic and loopback to the remote service in
        the same process (the server currently handles one connection at a time
        sequentially, see whisper_server.py). Which source gets registered is decided
        by audio.mix_mode:
          mix_mode: "mic"      -> registers "mic" (capture_process only captures the microphone)
          mix_mode: "loopback" -> registers "loopback" (capture_process only captures system audio)
        If you need to transcribe meeting audio and recognize the microphone at the
        same time, run two separate client processes with different --config files.
        """
        stt_config = self.config.get("stt", {})
        server_host = self._resolve_env_var(stt_config.get("server_host", "127.0.0.1")) or "127.0.0.1"
        server_port = self._resolve_env_var(stt_config.get("server_port", 45678)) or 45678
        mix_mode = self.config.get("audio", {}).get("mix_mode", "loopback")
        self._audio_source = "mic" if mix_mode == "mic" else "loopback"
        sources = {
            self._audio_source: (server_host, int(server_port))
        }
        self.stt = SpeechRecognizer(
            sources=sources,
            on_result=self._on_remote_result,
            sample_rate=self.config.get("audio", {}).get("sample_rate", 16000),
        )

    def _init_llm(self):
        ai_config = self.config.get("ai", {})
        api_key = self._resolve_env_var(ai_config.get("api_key", ""))
        provider = self._resolve_env_var(ai_config.get("provider"))
        model = self._resolve_env_var(ai_config.get("model"))
        api_base = self._resolve_env_var(ai_config.get("api_base"))
        if not all([provider, model, api_base]):
            raise ValueError(
                "ai.provider/model/api_base is not configured or the corresponding "
                "environment variable is missing -- check that .env contains "
                "AI_PROVIDER/AI_MODEL/AI_API_BASE"
            )
        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_base=api_base,
            api_key=api_key or None,
            max_tokens=ai_config.get("max_tokens", 2048),
            temperature=ai_config.get("temperature", 0.7),
        )
        self.llm = LLMClient(llm_config)

    def _init_tts(self):
        tts_config = self.config.get("tts", {})
        if tts_config.get("enabled", False):
            self.tts = TextToSpeech(
                provider=tts_config.get("provider", "edge-tts"),
                voice=tts_config.get("voice", "zh-CN-XiaoxiaoNeural"),
                speed=tts_config.get("speed", 1.0),
            )
        else:
            self.tts = None

    def _init_context(self):
        """Rolling statement context cache for assist mode (20-minute window by
        default, trimmed by time, not by turn count)"""
        context_config = self.config.get("context", {})
        window_minutes = context_config.get("window_minutes", 20)
        self.context = ContextBuffer(window_seconds=window_minutes * 60)

    def _init_listener(self):
        audio_config = self.config.get("audio", {})
        capture_config = CaptureConfig(
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            block_duration=audio_config.get("block_duration", 0.5),
            silence_threshold=audio_config.get("silence_threshold", 0.02),
            mic_silence_threshold=audio_config.get("mic_silence_threshold", 0.05),
            silence_duration=audio_config.get("silence_duration", 0.6),
            min_record_duration=audio_config.get("min_record_duration", 0.5),
            max_record_duration=audio_config.get("max_record_duration", 7),
            input_device=audio_config.get("input_device"),
            loopback_device=audio_config.get("loopback_device"),
            loopback_enabled=audio_config.get("loopback_enabled", False),
            mix_mode=audio_config.get("mix_mode", "auto"),
            streaming_enabled=audio_config.get("streaming_enabled", True),
            streaming_interval=audio_config.get("streaming_interval", 1.0),
        )
        # Audio capture runs in an independent process, to avoid scheduling jitter
        # from the main process's GPU inference causing dropped loopback frames
        self.listener = AudioCaptureProcess(
            config=capture_config,
            on_speech_end=self._on_speech_end,
            logger=logger,
        )

    def _on_speech_end(self, source, audio_data, is_final=True):
        """source: "mic" (microphone capture) | "loopback" (system output, meeting/video audio)
        is_final: True = this utterance has ended (pause/forced cut); False = a
                  mid-utterance chunk in streaming mode (the buffer is still growing).

        No longer calls model inference locally -- just forwards the current
        (incremental) audio to the remote recognition service; the actual recognition
        result arrives asynchronously via the _on_remote_result callback (the server's
        AlignAtt natively confirms increments, so the client doesn't need to simulate
        local-agreement itself).

        Only forwards the one source registered at startup (self._audio_source,
        decided by audio.mix_mode); the other source is ignored (under normal
        conditions capture_process already only captures the configured source, this
        is a second layer of protection).
        """
        if source != self._audio_source:
            return
        try:
            self.stt.feed(source, audio_data, is_final)
        except Exception as e:
            logger.error(f"Error forwarding audio to the remote recognition service: {e}")

    def _on_remote_result(self, source, result: dict):
        """Callback for incremental results from the remote recognition service
        (invoked on the STT client's reader thread).

        result fields (per SimulStreaming's whisper_server.py protocol):
          text     : the newly confirmed text increment for this update (not the
                     full text accumulated from the start)
          is_final : whether this marks the end of an utterance (server-side VAC-detected silence)

        transcribe mode: incremental text is printed directly, for a live-subtitle
        effect; a newline is printed on is_final.
        assist mode: the mic source is checked for questions; the loopback source is
        buffered into the rolling context.
        """
        text = result.get("text", "")
        is_final = result.get("is_final", False)

        if self.mode == "transcribe":
            if text:
                print(text, end="", flush=True)
            if is_final:
                print(flush=True)
            return

        # assist mode
        if text:
            if source == "mic":
                if not self._question_fired[source] and text.strip().endswith("?"):
                    self._question_fired[source] = True
                    logger.info(f"Question detected: {text}")
                    self._ask_assist(text)
            elif source == "loopback":
                self.context.add(text, source="loopback")

        if is_final:
            self._question_fired[source] = False

    def _ask_assist(self, question):
        """Ask the LLM: include the rolling context as background, output a bilingual (Chinese/English) answer"""
        context_text = self.context.get_context_text()
        user_content = question
        if context_text:
            user_content = (
                f"[Recent context, for background only]\n{context_text}\n\n"
                f"[Question]\n{question}"
            )
        messages = [
            {"role": "system", "content": self._ASSIST_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        logger.info("AI analyzing...")
        print("\n💡 ", end="", flush=True)
        self._run_stream(self.llm.chat(messages, stream=True))

    _ASSIST_SYSTEM_PROMPT = (
        "You are a real-time assistive thinking helper. The user is in a meeting/watching a video/"
        "communicating in English. You will receive recent contextual statements (for background only) "
        "and a question extracted from the speech. Answer the question concisely but with insight: "
        "lead with the direct answer, then briefly add the key reasoning, points, or possible directions, "
        "using the context if relevant. Avoid vague filler. Do not use Markdown formatting.\n"
        "Output format requirement: respond in English only, with no extra title or explanation."
    )

    def _run_stream(self, async_gen):
        """Submit a streaming generator to the persistent event loop for consumption,
        printing as chunks arrive, and return the full text.

        Blocks on the calling thread (the audio-processing thread) waiting for the
        result, but the coroutine itself runs on the dedicated loop, so the httpx
        client always stays bound to the same loop and its connection pool can be
        reused -- no deadlock from switching loops.
        """
        chunks = []

        async def consume():
            async for chunk in async_gen:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            print()

        future = asyncio.run_coroutine_threadsafe(consume(), self._loop)
        try:
            # Give it a generous timeout, to avoid blocking forever on a network hiccup
            future.result(timeout=self.llm.config.timeout + 30)
        except Exception as e:
            logger.error(f"AI request error: {e}")
            future.cancel()
        return "".join(chunks)

    def run(self):
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        logger.info("=" * 50)
        logger.info("agent voice assistant started")
        logger.info("=" * 50)
        names = {"transcribe": "transcribe only", "assist": "Q&A assist"}
        source_names = {"mic": "microphone", "loopback": "system audio"}
        source_label = source_names.get(self._audio_source, self._audio_source)
        print(f"\nListening ({source_label})... current mode: {self.mode} ({names.get(self.mode, self.mode)})")
        print("Press Ctrl+C to exit\n")
        self.listener.start()
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_once(self):
        logger.info("One-shot Q&A mode, please speak...")
        self.listener.start()
        try:
            # The model is already warmed up, 30 seconds is enough
            time.sleep(30)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("Shutting down agent...")
        self._running = False
        self.listener.stop()
        # Close the httpx client (submitted to the persistent loop), then stop the loop
        try:
            if self._loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self.llm.close(), self._loop)
                fut.result(timeout=5)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=3)
        except Exception:
            pass
        logger.info("agent has exited")

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        self._running = False


def list_audio_devices():
    import sounddevice as sd
    print("\nAvailable microphone devices:")
    print("-" * 60)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}")
            print(f"      channels: {dev['max_input_channels']}, default sample rate: {dev['default_samplerate']}")
    print()


def main():
    # Enable high-performance mode
    set_high_performance()

    # Auto-load .env from the project root on startup (see load_env_file's default path)
    load_env_file()

    parser = argparse.ArgumentParser(description="Agentic voice assistant")
    parser.add_argument("--config", "-c", default="config/trans.yaml", help="Path to the config file")
    parser.add_argument("--once", "-1", action="store_true", help="One-shot Q&A mode")
    parser.add_argument("--list-devices", "-l", action="store_true", help="List microphone devices")
    parser.add_argument(
        "--log-level", "-v",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override the config file's logging.level, for quick ad-hoc debugging "
             "(e.g. -v DEBUG to see partial-skip/inference timing)",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    agent = ZxAgent(config_path=args.config, log_level_override=args.log_level)
    if args.once:
        agent.run_once()
    else:
        agent.run()


if __name__ == "__main__":
    main()
