# API

This documents the current, implemented API of the AI server (`ai-server/main.py`). There is
one endpoint. A WebSocket API for streaming audio is planned but does not exist yet.

## `POST /translate-audio`

### Request

- Method: `POST`
- Body: raw binary audio data — **16-bit signed PCM, mono, 16 kHz sample rate** — sent as the
  full request body (`Content-Type` is not inspected by the server). This is not a WAV file
  (no header) and not JSON/multipart; it is the bare PCM sample bytes.
- If the body is empty, the server responds `400 Bad Request` with `{"detail": "No audio data received"}`.
- If processing fails for any other reason, the server responds `500 Internal Server Error`
  with `{"detail": "<exception message>"}`.

### Response

`200 OK` with a JSON object:

| Field              | Type   | Meaning                                                                 |
|--------------------|--------|--------------------------------------------------------------------------|
| `user_transcript`  | string | German text transcribed from the submitted audio via faster-whisper. Empty string if nothing was recognized. |
| `doctor_reply_de`  | string | The doctor dialogue engine's (`DoctorBrain`) next line, in German.       |
| `translated_text`  | string | English translation/subtitle of `doctor_reply_de`.                       |

If the transcript is empty (nothing understood), `doctor_reply_de`/`translated_text` are a
fixed "I couldn't hear you clearly, could you repeat that?" message instead of a dialogue
response.

### Notes

- Dialogue state (`DoctorBrain`) is held in a single global instance on the server process —
  it is not currently scoped per client/session, so concurrent users would share one
  conversation state.
- There is no authentication on this endpoint.
