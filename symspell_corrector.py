"""
SYMSPELL & BIGRAM LANGUAGE CORRECTOR PRO v6.0
Ultra-fast Symmetric Delete Spelling Correction (O(1) lookup) + N-gram Context.

Includes:
- SymSpell Engine: Precomputed symmetric deletion hash index (distance <= 2)
- Bigram Contextual Language Model (P(w_i | w_{i-1}))
- Dynamic Learning: Real-time insertion into index upon word commit
- Alphanumeric & Proper Noun Preservation (AZIZ, GAYETH, ANAS, HANA, 2025, A4)
"""

import os
import re
import itertools
from collections import defaultdict, Counter


CONFUSION_LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'A': '4'
}
CONFUSION_DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'Z', '8': 'B'
}
DIGIT_TO_LETTERS_MAP = {
    '0': ['O', 'I', '0'],
    '1': ['I', 'L', '1'],
    '2': ['Z', '2'],
    '4': ['A', '4'],
    '5': ['S', '5'],
    '6': ['G', '6'],
    '7': ['Z', 'T', '7'],
    '8': ['B', '8']
}
ALPHANUM_PATTERNS = {
    "A4", "A3", "A5", "MP3", "MP4", "3D", "2D", "4K", "HD", "V1", "V2", "V3", "V4", "V5", "V6",
    "F1", "H2O", "CO2", "PS5", "PS4", "M1", "M2", "M3", "I7", "I5", "I9", "B1", "B2", "C1", "C2"
}

BUILTIN_LEXICON = [
    # Common French words
    "BONJOUR", "SALUT", "MERCI", "OUI", "NON", "BIEN", "FAIRE", "AVEC", "POUR", "DANS", "SUR",
    "TOUT", "PLUS", "TEMPS", "JOUR", "HEURE", "HOMME", "FEMME", "AMI", "VIE", "MONDE", "VOIR",
    "SAVOIR", "ALLER", "VENIR", "NOUVEAU", "GRAND", "PETIT", "MAISON", "TRAVAIL", "PAGE", "MOT",
    "LETTRE", "TEXTE", "ECRIRE", "DESSIN", "MAIN", "DOIGT", "AIR", "TABLEAU", "BLANC", "NOIR",
    "COULEUR", "CAMERA", "CLAVIER", "VISION", "INTELLIGENT", "PROJET", "CODE", "TEST", "FRANCE",
    "PARIS", "ECOLE", "UNIVERSITE", "ETUDIANT", "PROFESSEUR", "LIVRE", "CAHIER", "STYLOS", "GOMME",
    "ESPACE", "RETOUR", "AIDE", "OPTION", "FICHIER", "IMAGE", "VIDEO", "EXPORT", "SAUVER", "QUITTER",
    "UN", "DEUX", "TROIS", "QUATRE", "CINQ", "SIX", "SEPT", "HUIT", "NEUF", "DIX", "CENT", "MILLE",
    # Pronouns & Short French Words
    "JE", "TU", "IL", "ELLE", "NOUS", "VOUS", "ILS", "ELLES", "ON", "CE", "CET", "CETTE",
    "MON", "TON", "SON", "NOTRE", "VOTRE", "LEUR", "LE", "LA", "LES", "UNE", "DES",
    "EST", "SONT", "SUIS", "ES", "AI", "AS", "AVONS", "AVEZ", "ONT", "FAIS", "FAIT", "VA", "VAS", "VONT",
    "BONSOIR", "COUCOU", "CA", "VA", "BIENVENUE", "AZIZ", "ANAS", "HANA", "GAYETH", "MEDAZ",
    # Verbes et actions fréquents
    "JOUE", "JOUER", "JEU", "PARLE", "PARLER", "REGARDE", "REGARDER", "ECOUTE", "ECOUTER",
    "MANGE", "MANGER", "DORS", "DORMIR", "AIME", "AIMER", "VOIS", "VOIT", "SAIS", "SAIT", "PEUX", "PEUT", "VEUX", "VEUT",
    # Common English words
    "HELLO", "WORLD", "THANKS", "YES", "GOOD", "FINE", "MAKE", "WITH", "FROM", "HAVE", "TIME",
    "YEAR", "PEOPLE", "WAY", "DAY", "MAN", "WOMAN", "CHILD", "LIFE", "WORK", "FIRST", "NEW"
]

COMMON_BIGRAMS = {
    ("JE", "JOUE"): 30.0,
    ("IL", "JOUE"): 30.0,
    ("AZIZ", "JOUE"): 30.0,
    ("BONJOUR", "AZIZ"): 25.0,
    ("SALUT", "AZIZ"): 25.0,
    ("BONJOUR", "TOUT"): 20.0,
    ("TOUT", "LE"): 25.0,
    ("LE", "MONDE"): 25.0,
    ("CA", "VA"): 30.0,
    ("COMMENT", "CA"): 25.0,
    ("MERCI", "BEAUCOUP"): 25.0,
    ("JE", "SUIS"): 25.0,
    ("IL", "EST"): 25.0,
    ("C'EST", "BIEN"): 20.0,
    ("BONJOUR", "A"): 18.0,
    ("AU", "REVOIR"): 25.0,
    ("UN", "PROJET"): 18.0,
}


class SymSpellCorrector:
    """
    Symmetric Delete Spelling Correction engine.
    Guarantees sub-millisecond lookup time regardless of lexicon size.
    """
    def __init__(self, project_dir=None, max_edit_distance=2):
        self.max_edit_distance = max_edit_distance
        self.deletes = defaultdict(list)
        self.words = {}  # word -> frequency count
        self.user_words = set()
        self.bigrams = dict(COMMON_BIGRAMS)

        self.project_dir = project_dir or os.getcwd()
        self.user_words_file = os.path.join(self.project_dir, "user_words.txt")

        self._build_index()

    def _get_deletes(self, word, distance):
        """Generates all substrings of word by deleting up to distance characters."""
        res = set()
        queue = {word}
        for d in range(distance):
            next_q = set()
            for w in queue:
                if len(w) > 1:
                    for i in range(len(w)):
                        cand = w[:i] + w[i+1:]
                        res.add(cand)
                        next_q.add(cand)
            queue = next_q
        return res

    def _add_word(self, word, count=1):
        word = word.upper().strip()
        if not word:
            return
        self.words[word] = self.words.get(word, 0) + count

        # Index deletes
        dels = self._get_deletes(word, self.max_edit_distance)
        for d in dels:
            if word not in self.deletes[d]:
                self.deletes[d].append(word)

    def _build_index(self):
        # 1. Add built-in lexicon
        for w in BUILTIN_LEXICON:
            self._add_word(w, count=10)

        # 2. Add user words file if present
        if os.path.exists(self.user_words_file):
            try:
                with open(self.user_words_file, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().upper()
                        if w:
                            self.user_words.add(w)
                            self._add_word(w, count=100)  # High priority
            except Exception:
                pass

        print(f"[SymSpell] Index ready: {len(self.words)} words, {len(self.deletes)} deletion hashes.")

    def learn(self, word, prev_word=None):
        """Learns a word dynamically and updates index & user_words.txt."""
        w = word.upper().strip()
        if not w:
            return
        self.user_words.add(w)
        self._add_word(w, count=100)

        # Learn bigram if previous word exists
        if prev_word:
            pw = prev_word.upper().strip()
            self.bigrams[(pw, w)] = self.bigrams.get((pw, w), 5.0) + 15.0

        # Persist to disk
        try:
            with open(self.user_words_file, "a", encoding="utf-8") as f:
                f.write(w + "\n")
        except Exception:
            pass

    def lookup(self, word, max_dist=None):
        """
        Fast lookup via symmetric deletions. Returns [(cand_word, edit_dist, freq), ...].
        """
        if max_dist is None:
            max_dist = self.max_edit_distance

        word = word.upper().strip()
        if not word:
            return []

        # Exact match
        if word in self.words:
            return [(word, 0, self.words[word])]

        if max_dist == 0:
            return []

        candidates = {}
        # Generate deletions for query word
        query_dels = self._get_deletes(word, max_dist)
        query_dels.add(word)

        for d in query_dels:
            if d in self.deletes:
                for match_w in self.deletes[d]:
                    if match_w not in candidates:
                        # Approximate edit distance
                        ed = abs(len(word) - len(match_w)) + (0 if d in match_w else 1)
                        if ed <= max_dist:
                            candidates[match_w] = (match_w, ed, self.words.get(match_w, 1))

        # Sort by distance asc, frequency desc
        res = list(candidates.values())
        res.sort(key=lambda x: (x[1], -x[2]))
        return res

    def correct(self, char_candidates, prev_word=None, mode="auto"):
        """
        Contextual sequence correction with SymSpell and Bigram scoring.
        Accepts:
        - raw string: "SALDT"
        - list of chars: ['S', 'A', 'L', 'D', 'T']
        - list of dicts or tuples: [{'S': 0.9, '5': 0.1}, ...]
        """
        if not char_candidates:
            return "", []

        # Handle raw string input
        if isinstance(char_candidates, str):
            char_candidates = [[(c, 1.0)] for c in char_candidates]
        elif isinstance(char_candidates, (list, tuple)) and char_candidates:
            if isinstance(char_candidates[0], str):
                char_candidates = [[(c, 1.0)] for c in char_candidates]

        # Normalize in case a list of dicts was passed
        normalized_candidates = []
        for cands in char_candidates:
            if isinstance(cands, dict):
                normalized_candidates.append(sorted(cands.items(), key=lambda x: x[1], reverse=True))
            elif isinstance(cands, list):
                normalized_candidates.append(cands)
            else:
                normalized_candidates.append([(str(cands), 1.0)])
        char_candidates = normalized_candidates

        raw_chars = [cands[0][0] if cands else '?' for cands in char_candidates]
        raw_word = "".join(raw_chars)
        as_letters = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)

        # Mode: DIGITS ONLY
        if mode == "digits":
            as_letters = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)
            if as_letters in self.user_words or as_letters in self.words:
                alts = [raw_word] if raw_word != as_letters else []
                return as_letters, alts
            digit_chars = [CONFUSION_LETTER_TO_DIGIT.get(ch, ch) for ch in raw_chars]
            best_dig = "".join(digit_chars)
            alts = [raw_word] if raw_word != best_dig else []
            return best_dig, alts

        # Mode: LETTERS ONLY
        if mode == "letters":
            letter_chars = [CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars]
            best_let = "".join(letter_chars)
            alts = [raw_word] if raw_word != best_let else []
            return best_let, alts

        # -------------------------------------------------------------
        # Mode: AUTO
        # -------------------------------------------------------------
        # 1. Alphanumeric pattern match (A4, PS5, etc.)
        if raw_word in ALPHANUM_PATTERNS:
            return raw_word, []

        # 2. Multi-candidate digit confusion match against user words & lexicon (e.g. 4707 -> AZIZ)
        options = [DIGIT_TO_LETTERS_MAP.get(ch, [CONFUSION_DIGIT_TO_LETTER.get(ch, ch)]) for ch in raw_chars]
        if any(c.isdigit() for c in raw_chars):
            for combo in itertools.product(*options):
                cand = "".join(combo)
                if cand in self.user_words or cand in self.words:
                    alts = [raw_word] if raw_word != cand else []
                    return cand, alts
        else:
            as_letters = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)
            if as_letters in self.user_words or as_letters in self.words:
                alts = [raw_word] if raw_word != as_letters else []
                return as_letters, alts

        # 3. Direct digit confusion match (e.g. 2O25 -> 2025, 1Z3 -> 123)
        as_digits = "".join(CONFUSION_LETTER_TO_DIGIT.get(ch, ch) for ch in raw_chars)
        if as_digits.isdigit():
            num_digits = sum(1 for c in raw_chars if c.isdigit())
            can_be_digits = all(c.isdigit() or (c in CONFUSION_LETTER_TO_DIGIT) for c in raw_chars)
            if can_be_digits and (num_digits >= max(1, len(raw_chars) * 0.35) or raw_word.isdigit()):
                alts = [raw_word] if raw_word != as_digits else []
                return as_digits, alts

        # 4. SymSpell Fast O(1) Search for raw_word and as_letters
        suggestions = self.lookup(as_letters, max_dist=1)
        if not suggestions:
            suggestions = self.lookup(raw_word, max_dist=1)

        # Bigram contextual boost
        best_cand = None
        best_score = -999.0
        alts = []

        for cand, dist, freq in suggestions:
            score = freq - dist * 20.0
            if prev_word and (prev_word.upper(), cand) in self.bigrams:
                score += self.bigrams[(prev_word.upper(), cand)]
            if score > best_score:
                best_score = score
                best_cand = cand
            if cand != raw_word and cand not in alts:
                alts.append(cand)

        if best_cand:
            if raw_word != best_cand and raw_word not in alts:
                alts.insert(0, raw_word)
            return best_cand, alts[:4]

        # 5. Fallback: preserve letters if mostly alphabetic
        num_alpha = sum(1 for c in raw_chars if c.isalpha())
        if num_alpha >= len(raw_chars) * 0.6:
            cleaned = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)
            return cleaned, [raw_word] if raw_word != cleaned else []

        return raw_word, alts


# Alias for backward and shorthand compatibility
FastSymSpell = SymSpellCorrector

