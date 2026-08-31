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
import socket
import asyncio
import argparse
import signal
import subprocess
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
from gui.publisher import DisplayPublisher

# Env var set on the GUI subprocess we spawn (see _maybe_launch_gui) so
# gui/app.py can tell "launched by this agent" apart from "run directly by
# hand" and refuse the latter -- --gui/-g is meant to be the only entry
# point, since agent and GUI are now a paired session (see the GUI-process
# monitor in run()/run_once()/shutdown()). Must match the identical literal
# in gui/app.py -- kept as a plain duplicated string rather than a shared
# import, since importing anything from gui/app.py would pull in PyQt6 at
# module load time, which the agent core must never depend on, even
# indirectly.
_GUI_LAUNCH_TOKEN_ENV = "AGENTIC_GUI_LAUNCH_TOKEN"


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
    def __init__(self, config_path="config/trans.yaml", log_level_override=None,
                 gui_override=None, gui_auto_launch=False):
        self.config = self._load_config(config_path)
        self._running = False
        # The CLI's --gui/-g takes priority over the config file's display.enabled,
        # same override pattern as --log-level -- lets you turn the overlay on/off
        # per-run without editing the yaml. None = defer to the config file.
        self._gui_override = gui_override
        # When True (only set by --gui/-g), also spawn `python -m gui.app` as a
        # child process if nothing is already listening, so a single command
        # starts both the agent and the overlay window.
        self._gui_auto_launch = gui_auto_launch

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
        # Per-source running text for the utterance currently in progress. The
        # remote recognition service sends `text` as an incremental delta on
        # every update (see audio/stt.py's docstring), not the full sentence
        # accumulated so far -- printing deltas back-to-back to the terminal
        # happens to reconstruct the sentence visually, but anything that needs
        # the *whole* current sentence (the GUI display, the LLM question text,
        # the rolling context) needs it concatenated here first. Reset to ""
        # whenever is_final=True closes out an utterance.
        self._transcript_accum = {"mic": "", "loopback": ""}
        # transcribe mode only: how much of the case-normalized full_text has
        # already been printed to the terminal for each source, so only the
        # newly-added tail gets printed on each update (see _on_remote_result).
        self._transcript_printed_len = {"mic": 0, "loopback": 0}

        self._init_stt()
        self._init_llm()
        self._init_tts()
        self._init_context()
        self._init_display()
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
        """Load the mode-specific config file, deep-merged on top of
        config/common.yaml (fields shared across all modes -- audio capture
        basics, ai provider/model/api_base/api_key, logging.level, the
        display block). The mode file always wins on any key conflict; only
        add a field to common.yaml if it's genuinely identical across every
        mode config, otherwise leave it in the mode file where it belongs.

        common.yaml is optional -- if it's missing (e.g. a stripped-down
        deployment, or someone deleted it), the mode file is used as-is with
        a warning, so this isn't a hard dependency."""
        common_file = Path(config_path).resolve().parent / "common.yaml"
        common_config = {}
        if common_file.exists():
            with open(common_file, "r", encoding="utf-8") as f:
                common_config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"{common_file} not found, skipping common config merge")

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using default config")
            return common_config
        with open(config_file, "r", encoding="utf-8") as f:
            mode_config = yaml.safe_load(f) or {}
        logger.info(f"Config file loaded: {config_path} (merged with {common_file.name})")
        return self._deep_merge(common_config, mode_config)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge `override` into `base`, returning a new dict.
        Nested dicts are merged key-by-key; any other value type (including
        lists) is replaced outright by `override`'s value -- lists aren't
        concatenated, since silently merging list contents from two files
        would be surprising and hard to reason about."""
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ZxAgent._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

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
            # Per-recording decoder conditioning. Empty by default and meant to stay
            # that way for everyday use: it only helps when it describes the audio at
            # hand. Set it before a call whose subject and names are known. See the
            # stt.session_prompt notes in config/trans.yaml.
            session_prompt=self._resolve_env_var(stt_config.get("session_prompt", "")) or None,
            # Late-bound on purpose: _init_listener() runs after this, so
            # self.listener does not exist yet. The callback is only ever invoked
            # from the STT client's keepalive thread, long after both exist.
            # It reports False until listener.start(), which is correct -- there is
            # nothing to keep a connection alive for before capture is running.
            is_healthy=self._capture_is_healthy,
        )

    def _capture_is_healthy(self) -> bool:
        """Whether audio capture is currently working. Gates the STT keepalive so a
        client whose capture died stops holding the (one-client-at-a-time) recognition
        server open -- see RemoteSTTClient._keepalive_loop."""
        listener = getattr(self, "listener", None)
        if listener is None:
            return False
        return listener.is_capture_alive()

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

    def _init_display(self):
        """Optional display frontend publisher (e.g. the PyQt transparent overlay
        in gui/). Disabled by default -- costs nothing and changes no behavior
        unless display.enabled: true is set in the config. The agent core never
        imports PyQt or anything GUI-related directly; it only talks to
        DisplayPublisher's tiny API (transcript/answer_chunk/status/clear), which
        is a silent no-op whenever the display frontend isn't running. This keeps
        the agent core swappable for a rewrite in another language (e.g. Go)
        without touching the GUI at all -- only display_protocol.py's wire format
        needs to be respected."""
        display_config = self.config.get("display", {})
        enabled = display_config.get("enabled", False)
        if self._gui_override is not None:
            enabled = self._gui_override
        host = display_config.get("host", "127.0.0.1")
        port = display_config.get("port", 18765)

        self._gui_process = None
        if enabled and self._gui_auto_launch:
            width = display_config.get("width")
            height = display_config.get("height")
            opacity = display_config.get("opacity")
            theme = display_config.get("theme")
            self._gui_process = self._maybe_launch_gui(host, port, width, height, opacity, theme)

        self.display = DisplayPublisher(
            enabled=enabled,
            host=host,
            port=port,
            logger=logger,
        )

    def _maybe_launch_gui(self, host, port, width=None, height=None, opacity=None, theme=None):
        """Launch `python -m gui.app` as a child process, unless something is
        already listening on host:port (e.g. the user started the GUI manually
        in another terminal -- don't spawn a second, redundant overlay window).
        Only called when --gui/-g requested auto-launch; the agent core still
        never imports anything from gui/ or PyQt directly -- it just shells out
        to a separate `python -m gui.app` process, same as running it by hand.

        width/height/opacity/theme (optional, from config.display.*) are
        forwarded as the matching --flag so the auto-launched window starts
        with whatever was configured, instead of always falling back to
        gui.app's built-in defaults. All four are also adjustable live from
        the GUI's own Settings window (right-click the overlay) regardless of
        what they started as -- these config values only set the initial
        state at launch."""
        probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
        try:
            with socket.create_connection((probe_host, port), timeout=0.3):
                logger.info(f"GUI already listening on {probe_host}:{port}, not launching a new one")
                return None
        except OSError:
            pass  # nothing listening yet -- go ahead and launch it

        cmd = [sys.executable, "-m", "gui.app", "--host", str(host), "--port", str(port)]
        if width:
            cmd += ["--width", str(width)]
        if height:
            cmd += ["--height", str(height)]
        if opacity is not None:
            cmd += ["--opacity", str(opacity)]
        if theme:
            cmd += ["--theme", str(theme)]

        env = dict(os.environ)
        env[_GUI_LAUNCH_TOKEN_ENV] = "1"

        try:
            proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent), env=env)
            logger.info(f"Launched GUI overlay (pid={proc.pid})")
            self._start_gui_monitor(proc)
            return proc
        except Exception as e:
            logger.warning(f"Failed to auto-launch GUI overlay: {e}")
            return None

    def _start_gui_monitor(self, proc):
        """Since --gui/-g explicitly asked to pair the agent with a GUI
        overlay it launched itself, treat that pairing as a real session:
        if the GUI process exits for any reason (Quit from its tray menu,
        the window being killed, a crash), the agent shuts itself down too,
        rather than continuing to run headless with no display and no way
        to bring one back short of restarting the agent.

        Polling in a plain daemon thread rather than a signal/callback,
        since subprocess.Popen doesn't offer an exit notification and this
        only needs to react within a second or so, not instantly."""
        def _watch():
            proc.wait()  # blocks until the GUI process exits, however it exits
            if self._running:
                logger.info(
                    f"GUI overlay (pid={proc.pid}) exited -- shutting down the "
                    f"agent too, since it was launched paired via --gui/-g"
                )
                self._running = False

        thread = threading.Thread(target=_watch, daemon=True, name="gui-monitor")
        thread.start()

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

    # Minimum number of CONSECUTIVE all-caps words required before treating a
    # stretch of text as "Whisper's emphatic/agitated-speech all-caps quirk"
    # and normalizing it. Kept intentionally low-risk: a single all-caps word
    # (an acronym like "NASA"/"TV"/"OK", or a name written in caps) is left
    # completely untouched -- only genuinely sentence-length runs get
    # degraded, so normal mixed-case speech is never altered.
    _CAPS_RUN_THRESHOLD = 3

    @staticmethod
    def _looks_capsy(word: str) -> bool:
        """Whether `word` looks like an all-caps token: at least 1 alphabetic
        character and every one of them uppercase. Used for BOTH detecting
        contiguous run boundaries and counting run length -- deliberately
        includes single-letter words ("I", "A") so they don't break up an
        otherwise-continuous run just because they're short (e.g. "...AT
        ALL? IT CAUSED A LOT OF..." is one 9-word run, not two runs split
        around "A"). Whether a word actually gets degraded once it's inside
        a long-enough run is decided separately -- see _degrade_all_caps_runs."""
        letters = [c for c in word if c.isalpha()]
        return len(letters) >= 1 and all(c.isupper() for c in letters)

    @classmethod
    def _degrade_all_caps_runs(cls, text: str) -> str:
        """Lowercase words that are part of a run of _CAPS_RUN_THRESHOLD or
        more CONSECUTIVE all-caps words, leaving isolated all-caps words
        (acronyms, etc.) exactly as they came from the model.

        Whisper has a known quirk: for emphatic/agitated speech (raised
        voice, arguing, etc.) it sometimes renders a whole stretch of text in
        all-caps, mirroring a subtitle convention from its training data.
        That is not reliable signal in practice -- it is an all-or-nothing
        side effect tied to how a segment happens to get decoded, not a
        deliberate word-by-word emphasis marker -- so long runs of it are
        normalized away. A single capitalized word on its own (run length 1)
        is far more likely to be a genuine acronym/proper noun than "the
        model is shouting", so those are left alone; only a real run (a
        whole clause/sentence, run length >= _CAPS_RUN_THRESHOLD) is treated
        as the quirk and degraded.

        The pronoun "I" is never degraded even when it falls inside a
        confirmed long run, since it is always capitalized in standard
        English regardless of tone/emphasis -- but it still counts toward
        the run's length and does not break run continuity, so a run like
        "YOU KNOW I MUST GO NOW" still gets treated as one continuous run.

        Operates on the full accumulated utterance text (not a single
        streaming delta) specifically so a run split across two separate
        incremental updates -- e.g. delta 1 ends "...text YOU", delta 2
        starts "DON'T BUY..." -- is still correctly detected as one 3-word
        run, instead of two sub-threshold fragments that would each be
        wrongly left untouched."""
        words = text.split(" ")
        capsy = [cls._looks_capsy(w) for w in words]
        i, n = 0, len(words)
        while i < n:
            if not capsy[i]:
                i += 1
                continue
            j = i
            while j < n and capsy[j]:
                j += 1
            if j - i >= cls._CAPS_RUN_THRESHOLD:
                for k in range(i, j):
                    if words[k].upper() != "I":
                        words[k] = words[k].lower()
            i = j
        return " ".join(words)

    @staticmethod
    def _capitalize_first_letter(text: str) -> str:
        """Capitalize the first alphabetic character in `text`, leaving
        everything else untouched. Used after _degrade_all_caps_runs to fix
        up the sentence-initial word when a run starting at the very
        beginning of the utterance got degraded -- e.g. "YOU DON'T BUY..."
        degrades to "you don't buy...", which needs to become "You don't
        buy..." for a normal sentence start."""
        for i, c in enumerate(text):
            if c.isalpha():
                return text[:i] + c.upper() + text[i + 1:]
        return text

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
        delta = result.get("text", "")
        is_final = result.get("is_final", False)

        # `delta` is only the newly confirmed increment for this update (see
        # audio/stt.py) -- accumulate the RAW (un-normalized) delta here, then
        # recompute the case-normalized full text from scratch every time (see
        # _degrade_all_caps_runs) rather than normalizing delta-by-delta. This
        # matters because a genuine all-caps run can span a delta boundary
        # (see that method's docstring) -- normalizing only the current delta
        # in isolation could see fewer than _CAPS_RUN_THRESHOLD caps words and
        # wrongly leave a real run untouched.
        if delta:
            self._transcript_accum[source] += delta
        full_text = self._degrade_all_caps_runs(self._transcript_accum[source])
        full_text = self._capitalize_first_letter(full_text)

        if self.mode == "transcribe":
            if delta:
                # Only print the portion of the normalized text that is new
                # since the last update, to preserve the incremental
                # live-subtitle printing effect. Case-folding never changes
                # string length, so this offset stays valid even if an
                # earlier (already-printed) word's case is retroactively
                # corrected by a run that only became long enough on this
                # update -- the terminal just keeps whatever case it already
                # printed for that word (a minor, rare, cosmetic-only gap;
                # the GUI/LLM/context below always see the fully corrected
                # text, since they're sent the complete current string each
                # time rather than a diff).
                printed_len = self._transcript_printed_len[source]
                print(full_text[printed_len:], end="", flush=True)
                self._transcript_printed_len[source] = len(full_text)
            # `delta or is_final`, not just `delta`: the remote service's
            # final update for an utterance often carries no new text at all
            # (everything was already confirmed in earlier partial updates --
            # this is-final message is purely "this utterance is now over").
            # Gating the GUI push on `delta` alone meant the GUI never learned
            # is_final=True for those utterances -- it stayed sitting in
            # _transcript_partial (see OverlayWindow._on_transcript) and got
            # silently overwritten the moment the NEXT utterance's first
            # partial arrived, i.e. the whole utterance visibly vanished from
            # the GUI even though the terminal (which prints incrementally as
            # deltas arrive, independent of this call) showed it correctly.
            if delta or is_final:
                self.display.transcript(full_text, source, is_final)
            if is_final:
                print(flush=True)
                self._transcript_accum[source] = ""
                self._transcript_printed_len[source] = 0
            return

        # assist mode
        # Same `delta or is_final` fix as transcribe mode above -- otherwise
        # an utterance whose final update carries no new text never reaches
        # the GUI as a completed line.
        if delta or is_final:
            self.display.transcript(full_text, source, is_final)
            if source == "mic" and delta:
                if not self._question_fired[source] and full_text.strip().endswith("?"):
                    self._question_fired[source] = True
                    logger.info(f"Question detected: {full_text}")
                    self._ask_assist(full_text)

        if is_final:
            if source == "loopback" and full_text:
                self.context.add(full_text, source="loopback")
            self._question_fired[source] = False
            self._transcript_accum[source] = ""

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
        self.display.clear("answer")
        self.display.status("thinking")
        self._run_stream(self.llm.chat(messages, stream=True))
        self.display.status("listening")

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
                self.display.answer_chunk(chunk)
            print()
            self.display.answer_chunk("", done=True)

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
        self._running = True
        self.listener.start()
        try:
            # The model is already warmed up, 30 seconds is enough. Polling
            # self._running in small increments (rather than one flat sleep(30))
            # so a paired GUI overlay exiting (see _start_gui_monitor) can end
            # this early too, same as it does for run().
            deadline = time.time() + 30
            while self._running and time.time() < deadline:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("Shutting down agent...")
        self._running = False
        self.listener.stop()
        self.display.close()
        if self._gui_process is not None:
            try:
                self._gui_process.terminate()
                self._gui_process.wait(timeout=3)
            except Exception:
                pass
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
        "--gui", "-g",
        action="store_true",
        help="Enable the transparent overlay GUI display (overrides display.enabled "
             "in the config file). Also auto-launches `python -m gui.app` as a "
             "child process if nothing is already listening on the configured "
             "host:port, so a single command starts both. The two are paired for "
             "the rest of the session: closing the GUI (Quit from its tray icon, "
             "or the window being killed) shuts this agent process down too, and "
             "vice versa. gui.app also refuses to run standalone by hand -- --gui/-g "
             "on this command is the only supported way to start it.",
    )
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

    agent = ZxAgent(
        config_path=args.config,
        log_level_override=args.log_level,
        gui_override=(True if args.gui else None),
        gui_auto_launch=args.gui,
    )
    if args.once:
        agent.run_once()
    else:
        agent.run()


if __name__ == "__main__":
    main()
