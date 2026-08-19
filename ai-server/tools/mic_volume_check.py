import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHUNK = 0.25

print("Speak, then go silent. Watch the numbers. Ctrl+C to stop.")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
    while True:
        block, _ = stream.read(int(CHUNK * SAMPLE_RATE))
        volume = np.abs(block).mean()
        print(f"Volume: {volume:.0f}")