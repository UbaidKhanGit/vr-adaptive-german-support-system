import string
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate
import asyncio
import os
import edge_tts
import pygame

# ---------------------------------------------------------------------------
# Install English->German model once (first run only)
# ---------------------------------------------------------------------------
_installed = argostranslate.translate.get_installed_languages()
_en = next((l for l in _installed if l.code == "en"), None)
_de = next((l for l in _installed if l.code == "de"), None)
if not (_en and _de and _en.get_translation(_de)):
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(p for p in available if p.from_code == "en" and p.to_code == "de")
    argostranslate.package.install_from_path(pkg.download())

# ---------------------------------------------------------------------------
# Load STT model once
# ---------------------------------------------------------------------------
print("Loading speech model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded.\n")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 50
SILENCE_DURATION = 1.2
CHUNK = 0.25

MAX_TURNS = 8
MAX_PRON_ATTEMPTS = 5
MIN_WORDS_FOR_PRONUNCIATION = 3

# ---------------------------------------------------------------------------
# Text helpers (needed by the brain)
# ---------------------------------------------------------------------------
DIGIT_WORDS = {
    "0": "null", "1": "eins", "2": "zwei", "3": "drei", "4": "vier",
    "5": "fünf", "6": "sechs", "7": "sieben", "8": "acht",
    "9": "neun", "10": "zehn", "11": "elf", "12": "zwölf",
}

def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [DIGIT_WORDS.get(w, w) for w in text.split()]

# ===========================================================================
#  DOCTOR BRAIN — per-symptom dialogue trees
# ===========================================================================
#  Structure of a flow:
#    Each symptom has a list of STEPS. A step is a question the doctor asks.
#    Each step lists BRANCHES: (set_of_keywords, reaction_de, reaction_en).
#    The user's answer is matched against the branches; the doctor speaks the
#    matched reaction and moves to the next step. The special keyword "*"
#    catches anything that didn't match, so EVERY reply has an answer.
#    When the steps run out, the flow ends and the floor is open again.
# ===========================================================================

YES = {"ja", "genau", "richtig", "stimmt", "natürlich", "klar", "doch", "schon"}
NO = {"nein", "nicht", "nie", "kein", "keine", "nichts", "noch"}
SHORT_TIME = {"gestern", "heute", "stunden", "morgen", "tag", "kurzem"}
LONG_TIME = {"woche", "wochen", "monat", "monaten", "jahr", "jahre", "jahren",
             "tagen", "tage", "lange", "länger", "ewig"}
DONT_KNOW = {"weiß", "keine", "ahnung", "vielleicht", "sicher", "genau"}
HIGH = {"hoch", "39", "40", "41", "stark", "sehr", "schlimm", "viel"}
LOW = {"niedrig", "37", "38", "leicht", "bisschen", "wenig"}

SYMPTOM_FLOWS = {
    # ----------------------------------------------------------- FIEBER ----
    "fieber": {
        "aliases": {"fieber"},
        "steps": [
            {
                "q": {"de": "Seit wann haben Sie Fieber?",
                      "en": "Since when have you had a fever?"},
                "branches": [
                    (SHORT_TIME,
                     "Das ist noch frisch, gut dass Sie gleich gekommen sind.",
                     "That's still recent — good that you came in right away."),
                    (LONG_TIME,
                     "Das ist schon eine Weile, das sollten wir ernst nehmen.",
                     "That's been a while — we should take that seriously."),
                    (DONT_KNOW,
                     "Kein Problem, das passiert. Wir messen gleich nach.",
                     "No problem, that happens. We'll measure it in a moment."),
                    ("*",
                     "In Ordnung, das notiere ich mir.",
                     "All right, I'll make a note of that."),
                ],
            },
            {
                "q": {"de": "Haben Sie Ihre Temperatur gemessen? Wie hoch war das Fieber?",
                      "en": "Did you measure your temperature? How high was the fever?"},
                "branches": [
                    (HIGH,
                     "Das ist ziemlich hoch. Dann sollten Sie sich wirklich schonen.",
                     "That's quite high. You should really take it easy then."),
                    (LOW,
                     "Das ist noch moderat, aber wir behalten es im Auge.",
                     "That's still moderate, but we'll keep an eye on it."),
                    (NO,
                     "Macht nichts, das machen wir jetzt hier.",
                     "No problem, we'll do that here now."),
                    (DONT_KNOW,
                     "Kein Problem, wir messen gleich hier nach.",
                     "No problem, we'll measure it right here."),
                    ("*",
                     "Gut, danke für die Information.",
                     "Good, thank you for the information."),
                ],
            },
            {
                "q": {"de": "Haben Sie auch Schüttelfrost oder Gliederschmerzen?",
                      "en": "Do you also have chills or body aches?"},
                "branches": [
                    (YES,
                     "Das klingt nach einem grippalen Infekt.",
                     "That sounds like a flu-like infection."),
                    (NO,
                     "Gut, dann ist es wahrscheinlich nichts Ernstes.",
                     "Good, then it's probably nothing serious."),
                    ("*",
                     "Verstehe, das behalten wir im Blick.",
                     "I see, we'll keep an eye on that."),
                ],
            },
        ],
    },
    # ----------------------------------------------- KOPFSCHMERZEN ---------
    "kopfschmerzen": {
        "aliases": {"kopfschmerzen", "kopfweh", "migräne"},
        "steps": [
            {
                "q": {"de": "Nehmen Sie schon Schmerzmittel dagegen?",
                      "en": "Are you already taking painkillers for it?"},
                "branches": [
                    (YES,
                     "Und helfen die Ihnen?",
                     "And are they helping you?"),
                    (NO,
                     "In Ordnung, manchmal ist das auch besser so.",
                     "All right, sometimes that's better anyway."),
                    ("*",
                     "Verstehe. Das schauen wir uns genauer an.",
                     "I see. We'll take a closer look at that."),
                ],
            },
            {
                "q": {"de": "Wo genau sitzt der Schmerz? Vorne, hinten oder an den Seiten?",
                      "en": "Where exactly is the pain? Front, back, or on the sides?"},
                "branches": [
                    ({"vorne", "stirn"},
                     "Stirnschmerzen können von den Nebenhöhlen kommen.",
                     "Frontal pain can come from the sinuses."),
                    ({"hinten", "nacken"},
                     "Das kann vom Nacken kommen, oft durch Verspannung.",
                     "That can come from the neck, often from tension."),
                    ({"seite", "seiten", "schläfe", "schläfen"},
                     "Einseitige Schmerzen können auf Migräne hindeuten.",
                     "One-sided pain can indicate a migraine."),
                    ({"überall", "ganz", "alles"},
                     "Ein diffuser Schmerz — das kann viele Ursachen haben.",
                     "A diffuse pain — that can have many causes."),
                    ("*",
                     "Verstehe. Das hilft mir bei der Einschätzung.",
                     "I see. That helps me assess it."),
                ],
            },
            {
                "q": {"de": "Trinken Sie genug Wasser und schlafen Sie ausreichend?",
                      "en": "Are you drinking enough water and sleeping enough?"},
                "branches": [
                    (YES,
                     "Gut, dann können wir das als Ursache ausschließen.",
                     "Good, then we can rule that out as a cause."),
                    (NO,
                     "Das könnte ein Grund sein. Achten Sie bitte darauf.",
                     "That could be a reason. Please pay attention to that."),
                    ("*",
                     "In Ordnung, denken Sie daran, das ist oft die Ursache.",
                     "All right — keep it in mind, it's often the cause."),
                ],
            },
        ],
    },
    # --------------------------------------------------------- HUSTEN ------
    "husten": {
        "aliases": {"husten", "huste"},
        "steps": [
            {
                "q": {"de": "Ist der Husten trocken oder mit Schleim?",
                      "en": "Is the cough dry or with mucus?"},
                "branches": [
                    ({"trocken"},
                     "Ein trockener Husten, verstehe. Der ist oft hartnäckig.",
                     "A dry cough, I see. Those are often stubborn."),
                    ({"schleim", "produktiv", "auswurf"},
                     "Mit Schleim — dann arbeitet der Körper schon dagegen an.",
                     "With mucus — then the body is already fighting it."),
                    (DONT_KNOW,
                     "Kein Problem, ich höre gleich Ihre Lunge ab.",
                     "No problem, I'll listen to your lungs in a moment."),
                    ("*",
                     "Verstehe. Ich höre gleich mal Ihre Lunge ab.",
                     "I see. I'll listen to your lungs in a moment."),
                ],
            },
            {
                "q": {"de": "Husten Sie eher tagsüber oder nachts?",
                      "en": "Do you cough more during the day or at night?"},
                "branches": [
                    ({"nachts", "nacht", "abend", "abends"},
                     "Nächtlicher Husten stört den Schlaf — das schwächt zusätzlich.",
                     "Coughing at night disturbs sleep — that weakens you further."),
                    ({"tagsüber", "tag", "morgens", "morgen"},
                     "Verstehe, tagsüber. Das ist meistens harmloser.",
                     "I see, during the day. That's usually more harmless."),
                    ({"beides", "immer", "ganze"},
                     "Durchgehend also — das schauen wir uns genau an.",
                     "Constant, then — we'll look at that closely."),
                    ("*",
                     "In Ordnung, danke.",
                     "All right, thank you."),
                ],
            },
            {
                "q": {"de": "Rauchen Sie?",
                      "en": "Do you smoke?"},
                "branches": [
                    (YES,
                     "Das verschlimmert den Husten deutlich. Versuchen Sie zu reduzieren.",
                     "That makes the cough significantly worse. Try to cut down."),
                    (NO,
                     "Sehr gut, das schließt eine häufige Ursache aus.",
                     "Very good, that rules out a common cause."),
                    ("*",
                     "In Ordnung.",
                     "All right."),
                ],
            },
        ],
    },
    # ------------------------------------------------ BAUCHSCHMERZEN -------
    "bauchschmerzen": {
        "aliases": {"bauchschmerzen", "bauch", "magen", "magenschmerzen"},
        "steps": [
            {
                "q": {"de": "Haben Sie etwas Ungewöhnliches gegessen?",
                      "en": "Have you eaten anything unusual?"},
                "branches": [
                    (YES,
                     "Das könnte die Ursache sein. Was war es denn?",
                     "That could be the cause. What was it?"),
                    (NO,
                     "Verstehe, dann müssen wir weiter suchen.",
                     "I see, then we have to keep looking."),
                    (DONT_KNOW,
                     "Denken Sie in Ruhe nach, das kann wichtig sein.",
                     "Take your time to think — it can be important."),
                    ("*",
                     "In Ordnung, das notiere ich.",
                     "All right, I'll note that down."),
                ],
            },
            {
                "q": {"de": "Ist der Schmerz eher krampfartig oder dauerhaft?",
                      "en": "Is the pain more cramp-like or constant?"},
                "branches": [
                    ({"krampf", "krampfartig", "krämpfe", "wellen"},
                     "Krampfartig — das spricht eher für den Verdauungstrakt.",
                     "Cramp-like — that points more to the digestive tract."),
                    ({"dauerhaft", "ständig", "immer", "konstant"},
                     "Ein Dauerschmerz — den sollten wir genauer untersuchen.",
                     "A constant pain — we should examine that more closely."),
                    ("*",
                     "Verstehe. Ich taste gleich Ihren Bauch ab.",
                     "I see. I'll examine your abdomen in a moment."),
                ],
            },
            {
                "q": {"de": "Haben Sie auch Übelkeit oder Durchfall?",
                      "en": "Do you also have nausea or diarrhea?"},
                "branches": [
                    (YES,
                     "Dann klingt das nach einem Magen-Darm-Infekt.",
                     "Then it sounds like a gastrointestinal infection."),
                    (NO,
                     "Gut, das grenzt die Möglichkeiten ein.",
                     "Good, that narrows down the possibilities."),
                    ("*",
                     "In Ordnung, danke für die Auskunft.",
                     "All right, thanks for the information."),
                ],
            },
        ],
    },
    # ---------------------------------------------- HALSSCHMERZEN ----------
    "halsschmerzen": {
        "aliases": {"halsschmerzen", "hals"},
        "steps": [
            {
                "q": {"de": "Haben Sie Schwierigkeiten beim Schlucken?",
                      "en": "Do you have difficulty swallowing?"},
                "branches": [
                    (YES,
                     "Dann schauen wir uns den Rachen gleich an.",
                     "Then we'll take a look at your throat right away."),
                    (NO,
                     "Gut, das ist ein gutes Zeichen.",
                     "Good, that's a good sign."),
                    ("*",
                     "Verstehe. Öffnen Sie gleich bitte einmal den Mund.",
                     "I see. Please open your mouth in a moment."),
                ],
            },
            {
                "q": {"de": "Ist Ihre Stimme auch heiser?",
                      "en": "Is your voice also hoarse?"},
                "branches": [
                    (YES,
                     "Dann sind die Stimmbänder mitbetroffen. Schonen Sie Ihre Stimme.",
                     "Then the vocal cords are affected too. Rest your voice."),
                    (NO,
                     "Gut, dann bleibt es wohl beim Rachen.",
                     "Good, then it's probably limited to the throat."),
                    ("*",
                     "In Ordnung.",
                     "All right."),
                ],
            },
        ],
    },
    # ------------------------------------------------------ ERKÄLTUNG ------
    "erkältung": {
        "aliases": {"erkältet", "erkältung", "schnupfen", "grippe"},
        "steps": [
            {
                "q": {"de": "Haben Sie auch Schnupfen oder Halsschmerzen dazu?",
                      "en": "Do you also have a runny nose or sore throat with it?"},
                "branches": [
                    (YES,
                     "Die klassische Kombination — das klingt nach einer Erkältung.",
                     "The classic combination — that sounds like a cold."),
                    (NO,
                     "Interessant, dann ist es vielleicht etwas anderes.",
                     "Interesting, then it might be something else."),
                    ("*",
                     "Verstehe.",
                     "I see."),
                ],
            },
            {
                "q": {"de": "Seit wann fühlen Sie sich so?",
                      "en": "Since when have you been feeling this way?"},
                "branches": [
                    (SHORT_TIME,
                     "Ganz frisch also. Ruhe hilft jetzt am meisten.",
                     "Very recent, then. Rest helps the most right now."),
                    (LONG_TIME,
                     "Das zieht sich schon — Ihr Körper braucht mehr Erholung.",
                     "It's been dragging on — your body needs more rest."),
                    ("*",
                     "In Ordnung, das notiere ich mir.",
                     "All right, I'll make a note of that."),
                ],
            },
            {
                "q": {"de": "Waren Sie in letzter Zeit viel unter Menschen?",
                      "en": "Have you been around many people lately?"},
                "branches": [
                    (YES,
                     "Da hat man sich schnell etwas eingefangen.",
                     "It's easy to catch something that way."),
                    (NO,
                     "Dann war es wohl einfach Pech.",
                     "Then it was probably just bad luck."),
                    ("*",
                     "Verstehe.",
                     "I see."),
                ],
            },
        ],
    },
}

# --- symptoms without a deep tree yet: single follow-up (extend the same way)
SIMPLE_SYMPTOMS = {
    "schwindelig": {"de": "Trinken Sie genug Wasser und schlafen Sie gut?",
                    "en": "Are you drinking enough water and sleeping well?"},
    "müde": {"de": "Wie viele Stunden schlafen Sie pro Nacht?",
             "en": "How many hours do you sleep per night?"},
    "rücken": {"de": "Arbeiten Sie viel im Sitzen?",
               "en": "Do you work sitting down a lot?"},
    "übelkeit": {"de": "Mussten Sie sich auch übergeben?",
                 "en": "Did you also have to vomit?"},
    "schmerzen": {"de": "Wo genau haben Sie Schmerzen?",
                  "en": "Where exactly do you have pain?"},
}

# openers for vague "I feel sick" answers
VAGUE_WORDS = {"krank", "schlecht", "unwohl", "gut"}
VAGUE_REPLY = {"de": "Was für Beschwerden haben Sie genau?",
               "en": "What exactly are your symptoms?"}

GENERICS = [
    {"de": "Ich verstehe. Können Sie mir mehr darüber erzählen?",
     "en": "I see. Can you tell me more about that?"},
    {"de": "Interessant. Erzählen Sie bitte weiter.",
     "en": "Interesting. Please go on."},
    {"de": "Hm, das sollten wir uns genauer ansehen.",
     "en": "Hm, we should look at that more closely."},
    {"de": "Gut, dass Sie gekommen sind. Beschreiben Sie das bitte genauer.",
     "en": "It's good that you came in. Please describe that in more detail."},
]


# --- conversational intents: things patients SAY that aren't symptoms ------
INTENTS = [
    # patient asks for a prescription / medication
    ({"rezept", "verschreiben", "medikament", "medikamente", "tabletten", "antibiotika"},
     {"de": "Ein Rezept bekommen Sie am Ende des Termins von mir, keine Sorge. Erst untersuche ich Sie kurz.",
      "en": "You'll get a prescription from me at the end of the appointment, don't worry. First let me examine you briefly."}),

    # patient asks for a sick note
    ({"krankschreibung", "krankgeschrieben", "attest", "arbeitsunfähig"},
     {"de": "Eine Krankschreibung kann ich Ihnen ausstellen. Für wie viele Tage brauchen Sie sie?",
      "en": "I can write you a sick note. For how many days do you need it?"}),

    # sleep problems
    ({"schlafen", "schlaflos", "wach", "einschlafen", "durchschlafen"},
     {"de": "Schlafmangel verschlimmert alles. Versuchen Sie, vor dem Schlafen keine Bildschirme zu benutzen.",
      "en": "Lack of sleep makes everything worse. Try to avoid screens before going to bed."}),

    # patient describes it as bothersome / affecting daily life
    ({"lästig", "stört", "nervt", "arbeiten", "konzentrieren", "alltag"},
     {"de": "Ich verstehe, dass das Ihren Alltag beeinträchtigt. Wir kümmern uns darum.",
      "en": "I understand that it's affecting your daily life. We'll take care of it."}),

    # patient asks a direct question to the doctor
    ({"kannst", "können", "würden", "dürfte", "darf", "sollte", "soll"},
     {"de": "Gute Frage. Lassen Sie mich erst die Untersuchung abschließen, dann beantworte ich alles.",
      "en": "Good question. Let me finish the examination first, then I'll answer everything."}),

    # patient is worried / scared
    ({"angst", "sorge", "sorgen", "ernst", "schlimm", "gefährlich"},
     {"de": "Machen Sie sich keine allzu großen Sorgen. Nach der Untersuchung wissen wir mehr.",
      "en": "Don't worry too much. We'll know more after the examination."}),

    # gratitude / politeness
    ({"danke", "dankeschön", "nett", "freundlich"},
     {"de": "Gern geschehen, dafür bin ich da. Gibt es noch etwas?",
      "en": "You're welcome, that's what I'm here for. Is there anything else?"}),

    # patient says nothing else is wrong -> steer toward closing
    ({"sonst", "alles", "fertig", "das wars", "mehr"},
     {"de": "Gut, dann habe ich alles, was ich brauche. Ich untersuche Sie jetzt kurz.",
      "en": "Good, then I have everything I need. I'll examine you briefly now."}),
]

class DoctorBrain:
    """Walks per-symptom dialogue trees. Inside a flow, every user reply is
    matched to a prepared branch; '*' guarantees full coverage."""

    def __init__(self):
        self.flow = None        # active symptom flow (key into SYMPTOM_FLOWS)
        self.step_i = 0         # which step of the flow we're on
        self.done_flows = set()
        self.generic_i = 0

    # ---- helpers ----------------------------------------------------------
    def _find_new_symptom(self, words, text):
        for key, flow in SYMPTOM_FLOWS.items():
            if key in self.done_flows:
                continue
            if words & flow["aliases"]:
                return key
        return None

    def _match_branch(self, branches, words):
        for keys, de, en in branches:
            if keys == "*":
                continue
            if words & keys:
                return de, en
        # guaranteed catch-all
        for keys, de, en in branches:
            if keys == "*":
                return de, en
        return None, None

    def _generic(self):
        r = GENERICS[self.generic_i % len(GENERICS)]
        self.generic_i += 1
        return dict(r)

    # ---- main entry --------------------------------------------------------
    def reply(self, user_german):
        words = set(normalize(user_german))
        text = user_german.lower()

        # 1. Are we inside a symptom flow? React to the branch, then ask next.
        if self.flow is not None:
            flow = SYMPTOM_FLOWS[self.flow]
            step = flow["steps"][self.step_i]
            react_de, react_en = self._match_branch(step["branches"], words)
            self.step_i += 1

            if self.step_i < len(flow["steps"]):
                nxt = flow["steps"][self.step_i]["q"]
                return {"de": f"{react_de} {nxt['de']}",
                        "en": f"{react_en} {nxt['en']}"}
            else:
                # flow finished
                self.done_flows.add(self.flow)
                self.flow = None
                self.step_i = 0
                return {"de": f"{react_de} Gibt es sonst noch etwas?",
                        "en": f"{react_en} Is there anything else?"}

        # 2. New deep-flow symptom mentioned? Enter its tree.
        new = self._find_new_symptom(words, text)
        if new:
            self.flow = new
            self.step_i = 0
            return dict(SYMPTOM_FLOWS[new]["steps"][0]["q"])

        # 3. Simple symptoms (single follow-up, extendable to full trees)
        for key, r in SIMPLE_SYMPTOMS.items():
            if key in text and key not in self.done_flows:
                self.done_flows.add(key)
                return dict(r)

        # 4. Vague "I feel sick" openers
        if words & VAGUE_WORDS:
            return dict(VAGUE_REPLY)

        # 4.5 Conversational intents (requests, complaints, questions...)
        for keys, response in INTENTS:
            if words & keys:
               return dict(response)

        # 5. Varied generic fallback
        return self._generic()


# ---------------------------------------------------------------------------
# STT — record until silence, transcribe in the requested language
# ---------------------------------------------------------------------------
def listen(language):
    print(f"   🎤 Listening ({language})... (take your time, stops after silence once you've spoken)")
    recorded = []
    silent_chunks = 0
    loud_chunks = 0
    chunks_needed = int(SILENCE_DURATION / CHUNK)
    MIN_SPEECH_CHUNKS = 3      # ~0.75s of sustained sound = real speech
    has_spoken = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            block, _ = stream.read(int(CHUNK * SAMPLE_RATE))
            recorded.append(block.copy())
            volume = np.abs(block).mean()

            if volume > SILENCE_THRESHOLD:
                loud_chunks += 1
                silent_chunks = 0
                if loud_chunks >= MIN_SPEECH_CHUNKS:
                    has_spoken = True          # only now does speech count
            else:
                if not has_spoken:
                    loud_chunks = 0            # a brief spike resets — not speech
                else:
                    silent_chunks += 1

            if has_spoken and silent_chunks >= chunks_needed:
                break

    audio = np.concatenate(recorded, axis=0)
    wav.write("recording.wav", SAMPLE_RATE, audio)
    segments, _ = model.transcribe("recording.wav", language=language,
                                   beam_size=1, vad_filter=True)
    transcript = " ".join(s.text for s in segments).strip()
    print(f"   Heard: \"{transcript}\"")
    return transcript

# ---------------------------------------------------------------------------
# Translation (English -> German)
# ---------------------------------------------------------------------------
def translate_to_german(english_text):
    if not english_text.strip():
        return ""
    return argostranslate.translate.translate(english_text, "en", "de")

# ---------------------------------------------------------------------------
# TTS placeholder
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TTS — Layer 1: GENERATE audio bytes (this layer goes unchanged to Unity)
# ---------------------------------------------------------------------------
TTS_VOICE = "de-DE-KatjaNeural"     # natural female German voice
                                    # alternative: "de-DE-ConradNeural" (male)
TTS_CACHE = "tts_cache"
os.makedirs(TTS_CACHE, exist_ok=True)

async def _generate_tts(text, path):
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate="-20%")
    await communicate.save(path)

def generate_word_audio(text):
    """Generate German TTS audio for a word/sentence. Returns the mp3 filepath.
    Cached: each word is only generated once, then reused (works offline after)."""
    safe = "".join(c for c in text.lower() if c.isalnum()) or "audio"
    path = os.path.join(TTS_CACHE, f"{safe}.mp3")
    if not os.path.exists(path):
        asyncio.run(_generate_tts(text, path))
    return path

# ---------------------------------------------------------------------------
# TTS — Layer 2: DELIVER the audio.
# Terminal prototype: play through speakers.
# [SWAP for Unity]: instead of playing, read the file bytes and send them
# over the WebSocket as a "pronunciation_audio" event; Unity plays them.
# ---------------------------------------------------------------------------
pygame.mixer.init()

def speak_word(word):
    print(f"   🔊 [AI voice] pronouncing: {word}")
    try:
        path = generate_word_audio(word)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"   (TTS unavailable: {e})")

# ---------------------------------------------------------------------------
# Pronunciation comparison
# ---------------------------------------------------------------------------
def check_pronunciation(target, spoken):
    target_words = normalize(target)
    spoken_words = normalize(spoken)
    first_wrong = None
    for i, word in enumerate(target_words):
        if not (i < len(spoken_words) and spoken_words[i] == word):
            first_wrong = word
            break
    return (first_wrong is None), first_wrong

def pronunciation_round(target_german):
    print(f'   Say this in German: "{target_german}"')
    attempts = 0
    while True:
        attempts += 1
        spoken = listen("de")
        passed, wrong_word = check_pronunciation(target_german, spoken)
        if passed:
            print("   ✅ Pronunciation OK — moving on.\n")
            return
        print(f"   ❌ Trouble with: '{wrong_word}'")
        if attempts >= MAX_PRON_ATTEMPTS:
            print("   Let's move on for now — you'll get it next time.\n")
            return
        if attempts >= 2:
            speak_word(wrong_word)
        print("   Say the WHOLE sentence again, focusing on that word.")

# ---------------------------------------------------------------------------
# Two ways to answer a turn
# ---------------------------------------------------------------------------
def turn_direct():
    print("   Mode: speak German directly.")
    return listen("de")

def turn_assisted():
    print("   Mode: answer in English, we'll help you say it in German.")
    user_english = listen("en")
    target_german = translate_to_german(user_english)
    print(f'   → In German: "{target_german}"')
    if len(normalize(target_german)) < MIN_WORDS_FOR_PRONUNCIATION:
        print("   (too short for pronunciation practice — passing to the doctor)\n")
    else:
        pronunciation_round(target_german)
    return target_german

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run():
    print("=" * 60)
    print("  VR GERMAN DOCTOR SIMULATION — prototype")
    print("=" * 60)

    brain = DoctorBrain()

    print('\n👨‍⚕️ Doctor: Guten Tag, was kann ich für Sie tun?')
    print('   Subtitle: Hello, what can I do for you?')

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- Turn {turn} of {MAX_TURNS} ---")

        choice = ""
        while choice not in ("1", "2"):
            choice = input("Choose:  [1] Speak German directly   [2] Answer in English (with help) : ").strip()

        user_german = turn_direct() if choice == "1" else turn_assisted()

        reply = brain.reply(user_german)
        print(f'👨‍⚕️ Doctor: {reply["de"]}')
        print(f'   Subtitle: {reply["en"]}')

    print("\n--- Consultation closing ---")
    print('👨‍⚕️ Doctor: Alles klar. Ich verschreibe Ihnen etwas. Ruhen Sie sich aus und trinken Sie viel. Gute Besserung!')
    print("   Subtitle: All right. I'll prescribe you something. Get some rest, drink plenty of fluids. Get well soon!")
    print("\n[Session ended]")


if __name__ == "__main__":
    run()