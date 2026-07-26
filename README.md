# zAgent - Real-time English Speech Recognition + Q&A Assistant

> **Always-on microphone/system audio capture -> remote streaming speech recognition (English only) -> real-time text output / Q&A assistance**

A voice assistant for meeting/video scenarios: transcribes English speech in real time, or detects questions and answers them with a bilingual (Chinese/English) explanation using recent context. Speech recognition does not run locally -- the client connects to a remote SimulStreaming service (a dedicated GPU server running Whisper large-v3 + AlignAtt streaming decoding).

---

## Two Modes

The program only supports two modes, **each started with its own independent config file; there is no runtime hotkey to switch between them**. If you need both modes at the same time, start two separate processes with different `--config` files:

```bash
python main.py --config config/trans.yaml     # transcribe mode (default config)
python main.py --config config/assist.yaml    # assist mode
```

| Mode | Description |
|------|-------------|
| `transcribe` | Only listens to system audio (loopback), streams it to the remote recognition service, and prints the recognized text in real time. No Q&A, no translation - optimized for the lowest possible latency. |
| `assist` | Listens to both the microphone (your questions) and system audio (statements from the meeting/video). System audio goes through remote recognition and confirmed sentences are buffered as rolling context (20-minute window by default). Local recognition for microphone questions is still under design (see "Known Limitations" below). |

Both modes **only recognize English** (`stt.language: "en"`) - no Chinese recognition or translation of the transcribed audio itself.

---

## Project Structure

```
zAgent/
├── main.py                    # Entry point - CLI argument parsing, module wiring, dispatching remote recognition results
├── config/
│   ├── trans.yaml             # transcribe mode config
│   └── assist.yaml            # assist mode config
├── requirements.txt           # Dependency list
├── .env.example                # Environment variable template (STT server address, AI provider/model/api_base/api_key)
├── memory.py                  # ContextBuffer (rolling statement context, used by assist mode)
├── audio/
│   ├── __init__.py
│   ├── capture_process.py     # Independent subprocess: mic + WASAPI loopback capture, streaming sliding-window segmentation
│   ├── stt.py                 # Remote speech recognition client (TCP connection to the SimulStreaming service, sends/receives incremental results)
│   └── tts.py                 # Text-to-speech (edge-tts / gTTS, optional)
├── ai/
│   ├── __init__.py
│   └── llm.py                 # LLM client (any OpenAI-compatible API, streaming output)
└── utils/
    ├── __init__.py
    └── logger.py               # Colored logging utility
```

---

## Architecture

### Core Flow

```
+------------------------------+         +---------------------------------+
|        zAgent Client          |         |      Remote GPU Server           |
|                                |  TCP    |                                   |
| +-------------+  +----------+ | audio-> | +-------------------------------+ |
| | Audio capture|->| Remote STT| +--------+->| SimulStreaming                | |
| | mic+loopback |  | client    | | <-incr | | Whisper large-v3 + AlignAtt   | |
| | own process  |  |(stt.py)   | | JSONL  | | automatic VAC segmentation    | |
| +-------------+  +----+-----+ |         | +-------------------------------+ |
|                        |       |         +-----------------------------------+
|              +---------v--------+
|              |  Mode dispatch    |
|              | transcribe: print |
|              | assist: Q detect  |
|              +--------+---------+
|                        |
|              +---------v--------+
|              |  LLM streaming    |
|              |  answer (bilingual,|
|              |  with context)    |
|              +------------------+
+---------------------------------+
```

### Remote Streaming Recognition

The client no longer loads any model locally. The dedicated audio capture subprocess only captures, segments, and forwards audio incrementally:

- `audio/capture_process.py` keeps growing the buffer for the current utterance using its existing sliding-window logic. When `streaming_enabled` is on, it fires a callback with the current buffer every `streaming_interval` seconds (default 1s), with `is_final=False`.
- `audio/stt.py`'s `RemoteSTTClient` receives the buffer and only sends the PCM16 audio that is new since the last send - it never resends the whole buffer (the server keeps its own audio state; it's not stateless inference).
- The remote SimulStreaming service uses VAC (voice activity detection) + the AlignAtt strategy to do true streaming incremental decoding. Results are streamed back as line-delimited JSON (JSONL): the `text` field is the newly confirmed text increment, and `is_final` marks whether this utterance has ended.
- `max_record_duration` (default 60s) is only a client-side safety net, to keep the buffer from growing unbounded if the server misbehaves or disconnects. Under normal conditions, segmentation is entirely driven by the server's VAC and this value is never hit.

In `transcribe` mode, incoming incremental text is printed directly as it streams in (live-subtitle effect). In `assist` mode, confirmed text from the loopback source is stored in the rolling context for use when answering later questions.

### Context Mechanism in Assist Mode

- **ContextBuffer**: a rolling time window (20 minutes by default) that stores the raw text of recently heard statements (currently only loopback/meeting audio, see "Known Limitations" below), used as background when answering questions. Content older than the window is dropped automatically; it is not trimmed by turn count.

### Known Limitations

- **Microphone (mic) question recognition is not implemented yet**: by design, mic audio is meant to be processed locally (it does not go through the remote recognition service, which is dedicated to meeting/other-party audio), but the local recognition approach is still being designed. In the current code, mic audio is captured but never forwarded to any recognition path. The mic-related recognition settings in `config/assist.yaml` are kept in their old format as placeholders until the local recognition approach is finalized.

---

## Technical Design

### Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Audio capture** | `sounddevice` + `soundcard` | Microphone + WASAPI loopback (system audio), isolated in its own subprocess |
| **Speech recognition (STT)** | Remote SimulStreaming service (TCP client) | Whisper large-v3 + AlignAtt streaming decoding, deployed on a dedicated GPU server; the client only captures/forwards/displays, with no local inference dependency |
| **AI / LLM** | `httpx` | Compatible with any OpenAI-style `/chat/completions` endpoint (DeepSeek/OpenAI/Moonshot/etc.); `provider` is only a logging label, not tied to a specific vendor |
| **Text-to-speech (TTS)** | `edge-tts` / `gTTS` | Optional |
| **Configuration** | `PyYAML` + `.env` | Structured config lives in YAML; sensitive/environment-specific values (server address, API keys) come from environment variables |
| **Logging** | `logging` + `RotatingFileHandler` | Colored console output + rotating log files |

### Supported AI Backends

Not tied to any specific vendor - any OpenAI-compatible `/chat/completions` endpoint works out of the box. `ai.provider` is only a logging label and does not affect request dispatch; which service is actually called is entirely determined by `ai.api_base` (together with `ai.model`/`ai.api_key`). Configure `AI_PROVIDER`/`AI_MODEL`/`AI_API_BASE`/`AI_API_KEY` in `.env` to pick your backend (see `.env.example`).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
#   STT_SERVER_HOST / STT_SERVER_PORT - remote SimulStreaming service address
#   AI_PROVIDER / AI_MODEL / AI_API_BASE / AI_API_KEY - LLM service config (needed for assist mode)
```

### 3. Run

```bash
# transcribe mode (default config: config/trans.yaml)
python main.py

# assist mode
python main.py --config config/assist.yaml

# one-shot Q&A mode
python main.py --once

# list available microphone devices
python main.py --list-devices

# temporarily override the log level (no need to edit the yaml)
python main.py -v DEBUG
```

### 4. Exit

Press `Ctrl+C` to shut down cleanly.

---

## Configuration Reference

- **stt**: `server_host`/`server_port` (remote recognition service address, recommended to use `${STT_SERVER_HOST}`/`${STT_SERVER_PORT}` from `.env`), `language` (fixed to `en`, for documentation only - the actual value in effect is whatever the server was started with)
- **audio**: sample rate, VAD silence thresholds, streaming segmentation parameters (`max_record_duration`/`streaming_interval`/`silence_duration`), mic/loopback mix mode
- **ai**: `provider`/`model`/`api_base`/`api_key` (recommended to use `${AI_PROVIDER}` etc. from `.env`), temperature
- **context**: assist mode's rolling context window (`window_minutes`, 20 minutes by default)
- **tts**: enable/disable spoken output

---

## Background

zAgent targets English listening assistance for meetings/videos on Windows. Audio capture runs in its own subprocess to avoid the main process's response handling / GPU scheduling jitter causing dropped frames in the system audio loopback. After moving speech recognition from local inference to a remote GPU server, per-segment recognition time dropped from 5s+ (local DirectML inference) to the 100-300ms range, bringing end-to-end latency down to 1-2s - only then did the streaming experience become genuinely usable.
