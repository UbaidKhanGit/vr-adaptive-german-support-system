# AI Server

FastAPI backend for the VR-based adaptive German learning system. It receives spoken audio
from a client, transcribes it (German speech-to-text), runs it through a simple rule-based
"doctor" dialogue engine (`DoctorBrain`), and returns the doctor's German reply plus an
English translation.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

On startup the server installs the Argos Translate English→German package (if not already
installed) and loads a `faster-whisper` "small" model on CPU. This can take a while on first
run.

## API

### `POST /translate-audio`

- **Accepts**: raw 16-bit PCM audio bytes (mono, 16 kHz) as the request body — no JSON
  wrapper, no WAV header.
- **Returns**: JSON with:
  - `user_transcript` — the German text transcribed from the audio (via faster-whisper)
  - `doctor_reply_de` — the doctor's next line, in German (from `DoctorBrain`)
  - `translated_text` — the English translation/subtitle of the doctor's reply

This is currently the server's only endpoint. It is a REST endpoint that expects one full
audio clip per request; a WebSocket endpoint for streaming/real-time interaction is planned
but not yet implemented.

## Folders

- `prototype/` — a standalone terminal prototype (`terminal_prototype.py`) of the full
  conversation flow: two answer modes (speak German directly, or answer in English and get
  pronunciation coaching), real German TTS via `edge-tts`/`pygame`, and the same
  `DoctorBrain` dialogue logic used by the server. Runs independently of FastAPI/Unity, useful
  for testing dialogue and pronunciation-check logic from the command line.
- `tools/` — small standalone diagnostic scripts used during development:
  - `mic_volume_check.py` — prints live microphone input volume, for tuning silence-detection
    thresholds.
  - `stt_silence_test.py` — records from the microphone until silence is detected, then
    transcribes the clip with faster-whisper (English), for testing the record/silence-cutoff
    logic in isolation.
