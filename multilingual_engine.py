"""
MULTILINGUAL LOCALIZATION & DICTIONARY ENGINE v6.0
Supports French (FR), English (EN), and Arabic (AR) for UI, Recognition, and Assistant.

Includes:
- Trilingual UI strings (Toasts, HUD, Action labels)
- Multi-dialect vocabulary (Standard French, English, and Arabic/Tunisian transliterations)
- Offline bilingual translation table for session notes synthesis
"""

UI_STRINGS = {
    "FR": {
        "TITLE": "AIR WRITER v6.0 PRO",
        "PEN_UP": "PLUME LEVEE",
        "DRAWING": "DESSIN EN COURS",
        "HOVER": "SURVOL CURSEUR",
        "SNAPPED": "FORME GEOMETRIQUE",
        "COMMITTED": "MOT VALIDE",
        "SPACE": "ESPACE",
        "ERASER": "EFFACEUR PAUME",
        "UNDO": "ANNULER DERNIER TRAIT",
        "CLEAR": "TABLEAU EFFACE",
        "REPLAY_START": "LECTURE REPLAY ANIMEE",
        "REPLAY_STOP": "FIN DU REPLAY",
        "REPLAY_PAUSE": "REPLAY EN PAUSE",
        "REPLAY_RESUME": "REPLAY EN COURS",
        "VCAM_ON": "CAM VIRTUELLE ACTIVE (OBS)",
        "VCAM_OFF": "CAM VIRTUELLE STOPPEE",
        "NOTES_SAVED": "NOTES EXPORTEES (.MD)",
        "MODE": "Mode",
        "WORDS": "Mots",
        "BRUSH": "Pinceau",
        "BOARD": "Fond",
        "LANG_SWITCH": "Langue : FRANCAIS"
    },
    "EN": {
        "TITLE": "AIR WRITER v6.0 PRO",
        "PEN_UP": "PEN UP",
        "DRAWING": "DRAWING",
        "HOVER": "HOVER CURSOR",
        "SNAPPED": "SHAPE SNAPPED",
        "COMMITTED": "COMMITTED",
        "SPACE": "SPACE",
        "ERASER": "SPONGE ERASER",
        "UNDO": "UNDO",
        "CLEAR": "CANVAS & TEXT CLEARED",
        "REPLAY_START": "ANIMATED REPLAY STARTED",
        "REPLAY_STOP": "REPLAY FINISHED",
        "REPLAY_PAUSE": "REPLAY PAUSED",
        "REPLAY_RESUME": "REPLAY RESUMED",
        "VCAM_ON": "V-CAM ACTIVE (OBS)",
        "VCAM_OFF": "V-CAM STOPPED",
        "NOTES_SAVED": "NOTES EXPORTED (.MD)",
        "MODE": "Mode",
        "WORDS": "Words",
        "BRUSH": "Brush",
        "BOARD": "Board",
        "LANG_SWITCH": "Language: ENGLISH"
    },
    "AR": {
        "TITLE": "AIR WRITER v6.0 PRO [AR]",
        "PEN_UP": "RAF' AL-QALAM",
        "DRAWING": "KITABA HAWAIYA",
        "HOVER": "TAHREEK",
        "SNAPPED": "CHAKL HANDASI",
        "COMMITTED": "TAM AL-HIFDH",
        "SPACE": "MASAFA",
        "ERASER": "MIMHAT AL-YAD",
        "UNDO": "TARAJOU3",
        "CLEAR": "MAS-H AL-LAWHA",
        "REPLAY_START": "I'ADAT AL-3ARD",
        "REPLAY_STOP": "IKHTITAM AL-3ARD",
        "REPLAY_PAUSE": "TAWAQQOF",
        "REPLAY_RESUME": "MOUTAWAASIL",
        "VCAM_ON": "KAMIRA IFTIRADHIYA (ON)",
        "VCAM_OFF": "KAMIRA IFTIRADHIYA (OFF)",
        "NOTES_SAVED": "TASDIR AL-MOULAHADHAT",
        "MODE": "Wadha3",
        "WORDS": "Kalimat",
        "BRUSH": "Risha",
        "BOARD": "Lawha",
        "LANG_SWITCH": "Al-Lougha: ARABIC"
    }
}

# Offline trilingual word meanings for the AI Session Assistant
MULTILINGUAL_GLOSSARY = {
    # French base
    "BONJOUR": {"FR": "Salutation du matin", "EN": "Good morning / Hello", "AR": "صباح الخير / مرحباً"},
    "SALUT": {"FR": "Salutation informelle", "EN": "Hi / Hello", "AR": "أهلاً وسهلاً"},
    "MERCI": {"FR": "Remerciement", "EN": "Thank you", "AR": "شكراً جزيلاً"},
    "OUI": {"FR": "Affirmation", "EN": "Yes", "AR": "نعم"},
    "NON": {"FR": "Négation", "EN": "No", "AR": "لا"},
    "BIEN": {"FR": "État positif / Bon", "EN": "Good / Well", "AR": "جيد / حسناً"},
    "JOUE": {"FR": "Verbe jouer (il/elle/je joue)", "EN": "Plays / Playing", "AR": "يلعب / يمارس لعبة"},
    "JOUER": {"FR": "Action de jouer", "EN": "To play", "AR": "اللعب"},
    "PARLE": {"FR": "Verbe parler", "EN": "Speaks / Speaking", "AR": "يتكلم / يتحدث"},
    "MANGE": {"FR": "Verbe manger", "EN": "Eats / Eating", "AR": "يأكل"},
    "DESSIN": {"FR": "Illustration / Croquis", "EN": "Drawing / Sketch", "AR": "رسم / مخطط"},
    "AIR": {"FR": "Espace aérien / Atmosphère", "EN": "Air / Spatial", "AR": "الهواء / الفضاء"},
    "TABLEAU": {"FR": "Surface d'écriture", "EN": "Whiteboard / Canvas", "AR": "سبورة / لوحة"},
    "CLAVIER": {"FR": "Dispositif de saisie", "EN": "Keyboard", "AR": "لوحة مفاتيح"},
    "CAMERA": {"FR": "Capteur vidéo", "EN": "Camera / Webcam", "AR": "كاميرا"},
    "PROJET": {"FR": "Travail de développement", "EN": "Project", "AR": "مشروع"},
    "AZIZ": {"FR": "Prénom de l'auteur (Mohamed Aziz)", "EN": "Aziz (Author name)", "AR": "عزيز (اسم المؤلف)"},
    "MEDAZ": {"FR": "Identifiant auteur (MedAz)", "EN": "MedAz (Author handle)", "AR": "معرّف عزيز"},
    # Arabic transliterated words
    "MARHABA": {"FR": "Bienvenue / Bonjour", "EN": "Welcome / Hello", "AR": "مرحباً"},
    "SALAM": {"FR": "Paix / Salutation", "EN": "Peace / Greeting", "AR": "السلام عليكم"},
    "CHOUKRAN": {"FR": "Merci", "EN": "Thanks / Thank you", "AR": "شكراً"},
    "LABES": {"FR": "Ça va / Tout va bien (Tunisien)", "EN": "Everything fine / OK", "AR": "لا بأس / الحمد لله"},
    "BEHI": {"FR": "D'accord / Très bien (Tunisien)", "EN": "Alright / Great", "AR": "حسناً / باهي"},
    # English base
    "HELLO": {"FR": "Bonjour", "EN": "Hello / Greeting", "AR": "مرحباً"},
    "WORLD": {"FR": "Monde", "EN": "World / Earth", "AR": "العالم"},
    "THANKS": {"FR": "Merci", "EN": "Thanks", "AR": "شكراً"},
    "YES": {"FR": "Oui", "EN": "Yes", "AR": "نعم"},
    "NO": {"FR": "Non", "EN": "No", "AR": "لا"},
    "CODE": {"FR": "Code source / Programme", "EN": "Source Code / Software", "AR": "برمجة / شفرة مصدرية"},
}


class MultilingualManager:
    """Manages active UI language and localized strings."""
    LANGS = ["FR", "EN", "AR"]

    def __init__(self, default_lang="FR"):
        self.current_idx = self.LANGS.index(default_lang) if default_lang in self.LANGS else 0
        self.lang = self.LANGS[self.current_idx]

    def cycle_language(self):
        """Switches to the next language (FR -> EN -> AR -> FR)."""
        self.current_idx = (self.current_idx + 1) % len(self.LANGS)
        self.lang = self.LANGS[self.current_idx]
        return self.lang, self.get("LANG_SWITCH")

    def set_language(self, lang_code):
        if lang_code.upper() in self.LANGS:
            self.current_idx = self.LANGS.index(lang_code.upper())
            self.lang = self.LANGS[self.current_idx]

    def get(self, key, default=""):
        """Returns the localized string for key."""
        table = UI_STRINGS.get(self.lang, UI_STRINGS["EN"])
        return table.get(key, default or key)

    @staticmethod
    def get_glossary_entry(word):
        """Returns translation mapping for recognized word if known."""
        w = word.upper().strip()
        return MULTILINGUAL_GLOSSARY.get(w, None)
