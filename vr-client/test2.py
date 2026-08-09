
import string
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate

# download and install the English->German model once (first run only)
_installed = argostranslate.translate.get_installed_languages()
_en = next((l for l in _installed if l.code == "en"), None)
_de = next((l for l in _installed if l.code == "de"), None)
if not (_en and _de and _en.get_translation(_de)):
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(p for p in available if p.from_code == "en" and p.to_code == "de")
    argostranslate.package.install_from_path(pkg.download())


# Load the STT model once at startup

print("Loading speech model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded.\n")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 50     # tuned to your mic (silence ~10-20, speech 100+)
SILENCE_DURATION = 2.0     # seconds of silence before recording stops
CHUNK = 0.25               # seconds per audio block



# DOCTOR logic :premeditated lines (English subtitle pre-translated)

GREETING = {"de": "Guten Tag, was kann ich für Sie tun?",
            "en": "Hello, what can I do for you?"}
CLOSING  = {"de": "Alles klar. Ruhen Sie sich aus und trinken Sie viel. Gute Besserung!",
            "en": "All right. Get some rest and drink plenty of fluids. Get well soon!"}
GENERIC  = {"de": "Ich verstehe. Können Sie mir mehr darüber erzählen?",
            "en": "I see. Can you tell me more about that?"}

SYMPTOM_REPLIES = {
    "fieber":        {"de": "Seit wann haben Sie Fieber?",
                      "en": "Since when have you had a fever?"},
    "kopfschmerzen": {"de": "Nehmen Sie schon Schmerzmittel?",
                      "en": "Are you already taking painkillers?"},
    "husten":        {"de": "Ist der Husten trocken oder mit Schleim?",
                      "en": "Is the cough dry or with mucus?"},
    "schmerzen":     {"de": "Wo genau haben Sie Schmerzen?",
                      "en": "Where exactly do you have pain?"},
}

MAX_TURNS = 5
MAX_PRON_ATTEMPTS = 2



# STT — record from mic until 5s of silence, then transcribe German

def listen_german():
    print("   🎤 Listening... (speak, stops after 5s of silence)")
    recorded = []
    silent_chunks = 0
    chunks_needed = int(SILENCE_DURATION / CHUNK)
    has_spoken = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
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

    audio = np.concatenate(recorded, axis=0)
    wav.write("recording.wav", SAMPLE_RATE, audio)
    segments, _ = model.transcribe("recording.wav", language="de")
    transcript = " ".join(s.text for s in segments).strip()
    print(f"   Heard: \"{transcript}\"")
    return transcript



# [SWAP] TRANSLATION, English -> German(demo dictionary for now)

def translate_to_german(english_text):
    if not english_text.strip():
        return ""
    return argostranslate.translate.translate(english_text, "en", "de")




# [SWAP] TTS: speak a German word aloud (placeholder)

def speak_word(word):
    print(f"   [AI voice] *pronounces clearly*: {word}")


# Pronunciation comparison
def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def check_pronunciation(target, spoken):
    target_words = normalize(target)
    spoken_words = normalize(spoken)
    first_wrong = None
    for i, word in enumerate(target_words):
        if not (i < len(spoken_words) and spoken_words[i] == word):
            first_wrong = word
            break
    return (first_wrong is None), first_wrong


def doctor_reply(user_german):
    text = user_german.lower()
    for keyword, reply in SYMPTOM_REPLIES.items():
        if keyword in text:
            return reply
    return GENERIC


def pronunciation_round(target_german):
    print(f'   Say this in German: "{target_german}"')
    attempts = 0
    while True:
        attempts += 1
        spoken = listen_german()
        passed, wrong_word = check_pronunciation(target_german, spoken)
        if passed:
            print("   ✅ Pronunciation OK — moving on.\n")
            return
        print(f"   ❌ Problem with the word: '{wrong_word}'")
        if attempts >= MAX_PRON_ATTEMPTS:
            speak_word(wrong_word)
            print("   Moving on.\n")
            return
        else:
            print("   Try again.")


# MAIN LOOP
def run():
    print("=" * 60)
    print("  VR GERMAN DOCTOR SIMULATION — prototype (live mic)")
    print("=" * 60)

    print(f'\n👨‍⚕️ Doctor: {GREETING["de"]}')
    print(f'   Subtitle: {GREETING["en"]}')

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- Turn {turn} of {MAX_TURNS} ---")

        user_english = input("🗣️  Your answer (type in English): ")
        target_german = translate_to_german(user_english)
        print(f'   → In German: "{target_german}"')

        pronunciation_round(target_german)

        reply = doctor_reply(target_german)
        print(f'👨‍⚕️ Doctor: {reply["de"]}')
        print(f'   Subtitle: {reply["en"]}')

    print("\n--- Consultation closing ---")
    print(f'👨‍⚕️ Doctor: {CLOSING["de"]}')
    print(f'   Subtitle: {CLOSING["en"]}')
    print("\n[Session ended]")


if __name__ == "__main__":
    run()