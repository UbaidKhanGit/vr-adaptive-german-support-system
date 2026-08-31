import sounddevice as sd, requests, time

URL = "http://localhost:8000/translate-audio"
SECONDS = 5

input("Press Enter, then speak German...")
rec = sd.rec(int(SECONDS * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()
print("sending...")

t0 = time.perf_counter()
r = requests.post(URL, data=rec.tobytes(),
                  headers={"Content-Type": "application/octet-stream"})
rt = time.perf_counter() - t0

print(f"round-trip: {rt:.2f}s")
print(r.json())