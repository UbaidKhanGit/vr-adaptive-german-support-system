# Architecture

The system has two main parts: a Unity VR client and a Python AI server. They communicate
over HTTP — the client records the user's spoken audio and sends it to the server; the server
replies with the transcript, the doctor's German reply, and an English translation.

## Components

- **Unity VR client** (`vr-client/`) — not yet implemented in this repository. It is
  responsible for capturing microphone audio in a VR doctor's-office scenario, sending it to
  the AI server, and displaying/speaking the response to the user.
- **AI server** (`ai-server/`) — a FastAPI application (`ai-server/main.py`) that, on startup,
  loads a `faster-whisper` "small" speech-to-text model and ensures both the Argos Translate
  German→English and English→German language packages are installed. It exposes one REST
  endpoint, `POST /translate-audio` (see [`docs/api/`](../api/)).
- **DoctorBrain** — a rule-based dialogue engine inside `ai-server/main.py`. It matches the
  transcribed German text against keyword sets to walk multi-step "symptom flow" trees (e.g.
  fever, headache, cough), recognize simpler one-off symptoms, and recognize general
  conversational intents (asking for a prescription, expressing worry, etc.), falling back to
  a rotating set of generic replies otherwise. It keeps per-process state (current flow,
  step, and which flows are already completed) across requests.
- **Argos Translate** — loaded at startup for German↔English translation (both directions).
  In the request path, it is invoked to translate the German transcription
  (`user_transcript`) into English (`user_transcript_en`) for subtitle display; the doctor's
  reply text itself is still written in German and English directly in `DoctorBrain`'s data,
  not machine-translated. The English→German direction is used for translation-assisted
  answers in the standalone terminal prototype (`ai-server/prototype/terminal_prototype.py`).

## Data flow (per request)

```
Unity VR client
      │  raw 16-bit PCM audio (mono, 16kHz)
      ▼
FastAPI server: POST /translate-audio
      │
      ▼
faster-whisper (German STT)
      │  german_transcript
      ├─────────────────────────────┐
      ▼                             ▼
DoctorBrain.reply(german_transcript)   Argos Translate (de → en)
      │  { de reply, en reply }        │  user_transcript_en
      ▼                             ◄──┘
JSON response
      │  { user_transcript, user_transcript_en,
      │    doctor_reply_de, doctor_reply_en, translated_text }
      ▼
Unity VR client
```

## Current limitations

- The Unity client itself does not exist yet in this repository (`vr-client/` is a
  placeholder).
- Only a single-shot REST endpoint exists; there is no WebSocket/streaming endpoint yet
  (planned).
- `DoctorBrain` state is a single global instance on the server, not session-scoped.
- Text-to-speech (turning the doctor's German reply into audio for the client to play) is
  demonstrated in the terminal prototype (`edge-tts` + `pygame`) but is not wired into the
  FastAPI server endpoint.
