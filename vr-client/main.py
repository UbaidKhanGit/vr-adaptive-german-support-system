import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
DURATION = 10

print("Loading done. Press Enter to start recording...")
input()

print(f"Recording for {DURATION} seconds...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("Done recording. Transcribing...")

wav.write("recording.wav", SAMPLE_RATE, audio)
segments, _ = model.transcribe("recording.wav")
transcript = " ".join(s.text for s in segments)

print(f"Transcript: {transcript}")
