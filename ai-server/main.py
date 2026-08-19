import string
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
import numpy as np
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate


# ===========================================================================
# 1. DIALOGUE DATA & INTENTS FROM TEST3.PY
# ===========================================================================

DIGIT_WORDS = {
    "0": "null", "1": "eins", "2": "zwei", "3": "drei", "4": "vier",
    "5": "fünf", "6": "sechs", "7": "sieben", "8": "acht",
    "9": "neun", "10": "zehn", "11": "elf", "12": "zwölf",
}

def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [DIGIT_WORDS.get(w, w) for w in text.split()]

YES = {"ja", "genau", "richtig", "stimmt", "natürlich", "klar", "doch", "schon"}
NO = {"nein", "nicht", "nie", "kein", "keine", "nichts", "noch"}
SHORT_TIME = {"gestern", "heute", "stunden", "morgen", "tag", "kurzem"}
LONG_TIME = {"woche", "wochen", "monat", "monaten", "jahr", "jahre", "jahren", "tagen", "tage", "lange", "länger", "ewig"}
DONT_KNOW = {"weiß", "keine", "ahnung", "vielleicht", "sicher", "genau"}
HIGH = {"hoch", "39", "40", "41", "stark", "sehr", "schlimm", "viel"}
LOW = {"niedrig", "37", "38", "leicht", "bisschen", "wenig"}

SYMPTOM_FLOWS = {
    "fieber": {
        "aliases": {"fieber"},
        "steps": [
            {
                "q": {"de": "Seit wann haben Sie Fieber?", "en": "Since when have you had a fever?"},
                "branches": [
                    (SHORT_TIME, "Das ist noch frisch, gut dass Sie gleich gekommen sind.", "That's still recent — good that you came in right away."),
                    (LONG_TIME, "Das ist schon eine Weile, das sollten wir ernst nehmen.", "That's been a while — we should take that seriously."),
                    (DONT_KNOW, "Kein Problem, das passiert. Wir messen gleich nach.", "No problem, that happens. We'll measure it in a moment."),
                    ("*", "In Ordnung, das notiere ich mir.", "All right, I'll make a note of that."),
                ],
            },
            {
                "q": {"de": "Haben Sie Ihre Temperatur gemessen? Wie hoch war das Fieber?", "en": "Did you measure your temperature? How high was the fever?"},
                "branches": [
                    (HIGH, "Das ist ziemlich hoch. Dann sollten Sie sich wirklich schonen.", "That's quite high. You should really take it easy then."),
                    (LOW, "Das ist noch moderat, aber wir behalten es im Auge.", "That's still moderate, but we'll keep an eye on it."),
                    (NO, "Macht nichts, das machen wir jetzt hier.", "No problem, we'll do that here now."),
                    (DONT_KNOW, "Kein Problem, wir messen gleich hier nach.", "No problem, we'll measure it right here."),
                    ("*", "Gut, danke für die Information.", "Good, thank you for the information."),
                ],
            },
            {
                "q": {"de": "Haben Sie auch Schüttelfrost oder Gliederschmerzen?", "en": "Do you also have chills or body aches?"},
                "branches": [
                    (YES, "Das klingt nach einem grippalen Infekt.", "That sounds like a flu-like infection."),
                    (NO, "Gut, dann ist es wahrscheinlich nichts Ernstes.", "Good, then it's probably nothing serious."),
                    ("*", "Verstehe, das behalten wir im Blick.", "I see, we'll keep an eye on that."),
                ],
            },
        ],
    },
    "kopfschmerzen": {
        "aliases": {"kopfschmerzen", "kopfweh", "migräne"},
        "steps": [
            {
                "q": {"de": "Nehmen Sie schon Schmerzmittel dagegen?", "en": "Are you already taking painkillers for it?"},
                "branches": [
                    (YES, "Und helfen die Ihnen?", "And are they helping you?"),
                    (NO, "In Ordnung, manchmal ist das auch besser so.", "All right, sometimes that's better anyway."),
                    ("*", "Verstehe. Das schauen wir uns genauer an.", "I see. We'll take a closer look at that."),
                ],
            },
            {
                "q": {"de": "Wo genau sitzt der Schmerz? Vorne, hinten oder an den Seiten?", "en": "Where exactly is the pain? Front, back, or on the sides?"},
                "branches": [
                    ({"vorne", "stirn"}, "Stirnschmerzen können von den Nebenhöhlen kommen.", "Frontal pain can come from the sinuses."),
                    ({"hinten", "nacken"}, "Das kann vom Nacken kommen, oft durch Verspannung.", "That can come from the neck, often from tension."),
                    ({"seite", "seiten", "schläfe", "schläfen"}, "Einseitige Schmerzen können auf Migräne hindeuten.", "One-sided pain can indicate a migraine."),
                    ({"überall", "ganz", "alles"}, "Ein diffuser Schmerz — das kann viele Ursachen haben.", "A diffuse pain — that can have many causes."),
                    ("*", "Verstehe. Das hilft mir bei der Einschätzung.", "I see. That helps me assess it."),
                ],
            },
            {
                "q": {"de": "Trinken Sie genug Wasser und schlafen Sie ausreichend?", "en": "Are you drinking enough water and sleeping enough?"},
                "branches": [
                    (YES, "Gut, dann können wir das als Ursache ausschließen.", "Good, then we can rule that out as a cause."),
                    (NO, "Das könnte ein Grund sein. Achten Sie bitte darauf.", "That could be a reason. Please pay attention to that."),
                    ("*", "In Ordnung, denken Sie daran, das ist oft die Ursache.", "All right — keep it in mind, it's often the cause."),
                ],
            },
        ],
    },
    "husten": {
        "aliases": {"husten", "huste"},
        "steps": [
            {
                "q": {"de": "Ist der Husten trocken oder mit Schleim?", "en": "Is the cough dry or with mucus?"},
                "branches": [
                    ({"trocken"}, "Ein trockener Husten, verstehe. Der ist oft hartnäckig.", "A dry cough, I see. Those are often stubborn."),
                    ({"schleim", "produktiv", "auswurf"}, "Mit Schleim — dann arbeitet der Körper schon dagegen an.", "With mucus — then the body is already fighting it."),
                    (DONT_KNOW, "Kein Problem, ich höre gleich Ihre Lunge ab.", "No problem, I'll listen to your lungs in a moment."),
                    ("*", "Verstehe. Ich höre gleich mal Ihre Lunge ab.", "I see. I'll listen to your lungs in a moment."),
                ],
            },
            {
                "q": {"de": "Husten Sie eher tagsüber oder nachts?", "en": "Do you cough more during the day or at night?"},
                "branches": [
                    ({"nachts", "nacht", "abend", "abends"}, "Nächtlicher Husten stört den Schlaf — das schwächt zusätzlich.", "Coughing at night disturbs sleep — that weakens you further."),
                    ({"tagsüber", "tag", "morgens", "morgen"}, "Verstehe, tagsüber. Das ist meistens harmloser.", "I see, during the day. That's usually more harmless."),
                    ({"beides", "immer", "ganze"}, "Durchgehend also — das schauen wir uns genau an.", "Constant, then — we'll look at that closely."),
                    ("*", "In Ordnung, danke.", "All right, thank you."),
                ],
            },
            {
                "q": {"de": "Rauchen Sie?", "en": "Do you smoke?"},
                "branches": [
                    (YES, "Das verschlimmert den Husten deutlich. Versuchen Sie zu reduzieren.", "That makes the cough significantly worse. Try to cut down."),
                    (NO, "Sehr gut, das schließt eine häufige Ursache aus.", "Very good, that rules out a common cause."),
                    ("*", "In Ordnung.", "All right."),
                ],
            },
        ],
    },
}

SIMPLE_SYMPTOMS = {
    "schwindelig": {"de": "Trinken Sie genug Wasser und schlafen Sie gut?", "en": "Are you drinking enough water and sleeping well?"},
    "müde": {"de": "Wie viele Stunden schlafen Sie pro Nacht?", "en": "How many hours do you sleep per night?"},
    "rücken": {"de": "Arbeiten Sie viel im Sitzen?", "en": "Do you work sitting down a lot?"},
    "übelkeit": {"de": "Mussten Sie sich auch übergeben?", "en": "Did you also have to vomit?"},
    "schmerzen": {"de": "Wo genau haben Sie Schmerzen?", "en": "Where exactly do you have pain?"},
}

VAGUE_WORDS = {"krank", "schlecht", "unwohl", "gut"}
VAGUE_REPLY = {"de": "Was für Beschwerden haben Sie genau?", "en": "What exactly are your symptoms?"}

GENERICS = [
    {"de": "Ich verstehe. Können Sie mir mehr darüber erzählen?", "en": "I see. Can you tell me more about that?"},
    {"de": "Interessant. Erzählen Sie bitte weiter.", "en": "Interesting. Please go on."},
    {"de": "Hm, das sollten wir uns genauer ansehen.", "en": "Hm, we should look at that more closely."},
    {"de": "Gut, dass Sie gekommen sind. Beschreiben Sie das bitte genauer.", "en": "It's good that you came in. Please describe that in more detail."},
]

INTENTS = [
    ({"rezept", "verschreiben", "medikament", "medikamente", "tabletten", "antibiotika"},
     {"de": "Ein Rezept bekommen Sie am Ende des Termins von mir, keine Sorge. Erst untersuche ich Sie kurz.", "en": "You'll get a prescription from me at the end of the appointment, don't worry. First let me examine you briefly."}),
    ({"krankschreibung", "krankgeschrieben", "attest", "arbeitsunfähig"},
     {"de": "Eine Krankschreibung kann ich Ihnen ausstellen. Für wie viele Tage brauchen Sie sie?", "en": "I can write you a sick note. For how many days do you need it?"}),
    ({"schlafen", "schlaflos", "wach", "einschlafen", "durchschlafen"},
     {"de": "Schlafmangel verschlimmert alles. Versuchen Sie, vor dem Schlafen keine Bildschirme zu benutzen.", "en": "Lack of sleep makes everything worse. Try to avoid screens before going to bed."}),
    ({"lästig", "stört", "nervt", "arbeiten", "konzentrieren", "alltag"},
     {"de": "Ich verstehe, dass das Ihren Alltag beeinträchtigt. Wir kümmern uns darum.", "en": "I understand that it's affecting your daily life. We'll take care of it."}),
    ({"kannst", "können", "würden", "dürfte", "darf", "sollte", "soll"},
     {"de": "Gute Frage. Lassen Sie mich erst die Untersuchung abschließen, dann beantworte ich alles.", "en": "Good question. Let me finish the examination first, then I'll answer everything."}),
    ({"angst", "sorge", "sorgen", "ernst", "schlimm", "gefährlich"},
     {"de": "Machen Sie sich keine allzu großen Sorgen. Nach der Untersuchung wissen wir mehr.", "en": "Don't worry too much. We'll know more after the examination."}),
    ({"danke", "dankeschön", "nett", "freundlich"},
     {"de": "Gern geschehen, dafür bin ich da. Gibt es noch etwas?", "en": "You're welcome, that's what I'm here for. Is there anything else?"}),
    ({"sonst", "alles", "fertig", "das wars", "mehr"},
     {"de": "Gut, dann habe ich alles, was ich brauche. Ich untersuche Sie jetzt kurz.", "en": "Good, then I have everything I need. I'll examine you briefly now."}),
]


# ===========================================================================
# 2. DOCTOR BRAIN LOGIC
# ===========================================================================

class DoctorBrain:
    def __init__(self):
        self.flow = None
        self.step_i = 0
        self.done_flows = set()
        self.generic_i = 0

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
        for keys, de, en in branches:
            if keys == "*":
                return de, en
        return None, None

    def _generic(self):
        r = GENERICS[self.generic_i % len(GENERICS)]
        self.generic_i += 1
        return dict(r)

    def reply(self, user_german):
        words = set(normalize(user_german))
        text = user_german.lower()

        if self.flow is not None:
            flow = SYMPTOM_FLOWS[self.flow]
            step = flow["steps"][self.step_i]
            react_de, react_en = self._match_branch(step["branches"], words)
            self.step_i += 1

            if self.step_i < len(flow["steps"]):
                nxt = flow["steps"][self.step_i]["q"]
                return {"de": f"{react_de} {nxt['de']}", "en": f"{react_en} {nxt['en']}"}
            else:
                self.done_flows.add(self.flow)
                self.flow = None
                self.step_i = 0
                return {"de": f"{react_de} Gibt es sonst noch etwas?", "en": f"{react_en} Is there anything else?"}

        new = self._find_new_symptom(words, text)
        if new:
            self.flow = new
            self.step_i = 0
            return dict(SYMPTOM_FLOWS[new]["steps"][0]["q"])

        for key, r in SIMPLE_SYMPTOMS.items():
            if key in text and key not in self.done_flows:
                self.done_flows.add(key)
                return dict(r)

        if words & VAGUE_WORDS:
            return dict(VAGUE_REPLY)

        for keys, response in INTENTS:
            if words & keys:
                return dict(response)

        return self._generic()


# Global DoctorBrain instance for session persistence
doctor_brain = DoctorBrain()


def translate_de_to_en(german_text):
    """Translate German text to English via Argos Translate, never raising on failure."""
    if not german_text or not german_text.strip():
        return ""
    try:
        return argostranslate.translate.translate(german_text, "de", "en")
    except Exception:
        return "[translation unavailable]"


# ===========================================================================
# 3. FASTAPI SERVER LIFESPAN & ENDPOINTS
# ===========================================================================

def _ensure_argos_package(from_code, to_code):
    installed = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in installed if l.code == from_code), None)
    to_lang = next((l for l in installed if l.code == to_code), None)
    if from_lang and to_lang and from_lang.get_translation(to_lang):
        return
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
    if pkg:
        argostranslate.package.install_from_path(pkg.download())


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🤖 Initializing AI Models & Translation Packages...")

    # 1. Setup Argos Translate (German <-> English, both directions)
    _ensure_argos_package("de", "en")
    _ensure_argos_package("en", "de")

    # 2. Initialize Faster-Whisper Model once
    stt_model = WhisperModel("small", device="cpu", compute_type="int8")
    
    app.state.models = {
        "stt_model": stt_model
    }
    
    print("🚀 All models loaded successfully!")
    yield
    print("🛑 Shutting down server...")
    app.state.models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/translate-audio")
async def translate_audio(request: Request):
    try:
        # 1. Capture raw PCM audio bytes from request body
        raw_pcm_bytes = await request.body()
        
        if not raw_pcm_bytes:
            raise HTTPException(status_code=400, detail="No audio data received")
            
        # 2. Convert 16-bit PCM binary to float32 NumPy array expected by Whisper
        audio_np = np.frombuffer(raw_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 3. Access pre-loaded Whisper model
        stt = request.app.state.models["stt_model"]
        
        # 4. Transcribe audio array directly (No temp WAV files needed!)
        segments, _ = stt.transcribe(audio_np, language="de", beam_size=5)
        german_transcript = " ".join(s.text for s in segments).strip()
        
        # 5. Calculate doctor response using DoctorBrain logic
        if german_transcript:
            doctor_response = doctor_brain.reply(german_transcript)
        else:
            doctor_response = {
                "de": "Ich habe Sie nicht verstanden. Können Sie das bitte wiederholen?",
                "en": "I couldn't hear you clearly. Could you please repeat that?"
            }

        # 6. Translate the German transcription into English for the subtitle pipeline
        user_transcript_en = translate_de_to_en(german_transcript)

        # 7. Return JSON payload required by the VR team
        return {
            "user_transcript": german_transcript,
            "user_transcript_en": user_transcript_en,
            "doctor_reply_de": doctor_response["de"],
            "doctor_reply_en": doctor_response["en"],
            "translated_text": doctor_response["en"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))