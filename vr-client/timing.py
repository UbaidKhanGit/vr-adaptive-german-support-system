import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 50      # raise if it stops too early, lower if it never stops
SILENCE_DURATION = 5.0      # seconds of silence before stopping
CHUNK = 0.25                # seconds per chunk

print("Press Enter, then start speaking...")
input()
print("Listening... (will stop after 5s of silence)")

recorded = []
silent_chunks = 0
chunks_needed = int(SILENCE_DURATION / CHUNK)
has_spoken = False

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
    while True:
        block, _ = stream.read(int(CHUNK * SAMPLE_RATE))
        recorded.append(block.copy())

        volume = np.abs(block).mean()

        if volume > SILENCE_THRESHOLD:
            has_spoken = True
            silent_chunks = 0
        elif has_spoken:
            silent_chunks += 1

        if has_spoken and silent_chunks >= chunks_needed:
            break

print("Silence detected. Transcribing...")

audio = np.concatenate(recorded, axis=0)
wav.write("recording.wav", SAMPLE_RATE, audio)
segments, _ = model.transcribe("recording.wav", language="en")
transcript = " ".join(s.text for s in segments)
print(f"Transcript: {transcript}")