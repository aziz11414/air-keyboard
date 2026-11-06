"""
Intelligent Text Corrector & Contextual Disambiguator for Smart Air Writer.
Handles:
  - Sequence-level scoring with Beam Search
  - Contextual letter vs digit confusion (O/0, I/1, Z/2, S/5, G/6, T/7, B/8)
  - Dictionary validation (French + English common words + custom words.txt)
  - Dynamic user learning (user_words.txt saved upon [Space] commit)
  - High-confidence character preservation for proper nouns (ANAS, HANA, GAYETH, etc.)
  - Alternative generation for interactive cycling via [Tab]
"""

import os
import re
from collections import Counter

# Optional wordfreq support
try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

CONFUSION_LETTER_TO_DIGIT = {
    'O': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
}

CONFUSION_DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
}

# Common alphanumeric patterns that shouldn't be flattened
ALPHANUM_PATTERNS = {
    "A4", "A3", "A5", "MP3", "MP4", "3D", "2D", "4K", "HD", "V1", "V2", "V3", "V4", "V5", "V6",
    "F1", "H2O", "CO2", "PS5", "PS4", "M1", "M2", "M3", "I7", "I5", "I9", "B1", "B2", "C1", "C2"
}

# Core built-in lexicon (French + English most common words)
BUILTIN_WORDS = {
    # French common
    "BONJOUR", "SALUT", "MERCI", "OUI", "NON", "BIEN", "FAIRE", "AVEC", "POUR", "DANS", "SUR",
    "TOUT", "PLUS", "TEMPS", "JOUR", "HEURE", "HOMME", "FEMME", "AMI", "VIE", "MONDE", "VOIR",
    "SAVOIR", "ALLER", "VENIR", "NOUVEAU", "GRAND", "PETIT", "MAISON", "TRAVAIL", "PAGE", "MOT",
    "LETTRE", "TEXTE", "ECRIRE", "DESSIN", "MAIN", "DOIGT", "AIR", "TABLEAU", "BLANC", "NOIR",
    "COULEUR", "CAMERA", "CLAVIER", "VISION", "INTELLIGENT", "PROJET", "CODE", "TEST", "FRANCE",
    "PARIS", "ECOLE", "UNIVERSITE", "ETUDIANT", "PROFESSEUR", "LIVRE", "CAHIER", "STYLOS", "GOMME",
    "ESPACE", "RETOUR", "AIDE", "OPTION", "FICHIER", "IMAGE", "VIDEO", "EXPORT", "SAUVER", "QUITTER",
    "UN", "DEUX", "TROIS", "QUATRE", "CINQ", "SIX", "SEPT", "HUIT", "NEUF", "DIX", "CENT", "MILLE",
    "LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE", "JANVIER", "FEVRIER",
    "MARS", "AVRIL", "MAI", "JUIN", "JUILLET", "AOUT", "SEPTEMBRE", "OCTOBRE", "NOVEMBRE", "DECEMBRE",
    # Pronouns & Short French Words
    "JE", "TU", "IL", "ELLE", "NOUS", "VOUS", "ILS", "ELLES", "ON", "CE", "CET", "CETTE",
    "MON", "TON", "SON", "NOTRE", "VOTRE", "LEUR", "LE", "LA", "LES", "UNE", "DES",
    "EST", "SONT", "SUIS", "ES", "AI", "AS", "AVONS", "AVEZ", "ONT", "FAIS", "FAIT", "VA", "VAS", "VONT",
    "BONSOIR", "COUCOU", "CA", "VA", "BIENVENUE", "AZIZ", "ANAS", "HANA", "GAYETH", "MEDAZ",
    # English common
    "HELLO", "WORLD", "THANKS", "YES", "GOOD", "FINE", "MAKE", "WITH", "FROM", "HAVE", "TIME",
    "YEAR", "PEOPLE", "WAY", "DAY", "MAN", "WOMAN", "CHILD", "LIFE", "WORK", "FIRST", "NEW",
    "HAND", "FINGER", "WRITE", "WRITER", "KEYBOARD", "SMART", "BOARD", "CLEAR", "COLOR", "SCREEN",
    "WHITE", "BLACK", "GREEN", "BLUE", "YELLOW", "PINK", "START", "STOP", "RECORD", "SAVE", "QUIT",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "HUNDRED",
    "TODAY", "NIGHT", "LIGHT", "WATER", "HOUSE", "SCHOOL", "COMPUTER", "DATA", "MODEL", "SYSTEM"
}


class TextCorrector:
    def __init__(self, project_dir=None):
        if project_dir is None:
            self.project_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.project_dir = project_dir

        self.user_words_file = os.path.join(self.project_dir, "user_words.txt")
        self.words_file = os.path.join(self.project_dir, "words.txt")

        self.user_words = set()
        self.dictionary = set(BUILTIN_WORDS)

        self._load_dictionary_files()

    def _load_dictionary_files(self):
        # 1. Custom words.txt if present
        if os.path.exists(self.words_file):
            try:
                with open(self.words_file, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().upper()
                        if w and re.match(r"^[A-Z0-9]+$", w):
                            self.dictionary.add(w)
            except Exception as e:
                print(f"[Corrector] Warning reading words.txt: {e}")

        # 2. Dynamic user words (learned on Space)
        if os.path.exists(self.user_words_file):
            try:
                with open(self.user_words_file, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().upper()
                        if w:
                            self.user_words.add(w)
                            self.dictionary.add(w)
            except Exception as e:
                print(f"[Corrector] Warning reading user_words.txt: {e}")

    def learn(self, word):
        """Learn a new word when committed with [Space]."""
        w = word.strip().upper()
        if not w or len(w) < 1:
            return
        if w in self.user_words:
            return
        self.user_words.add(w)
        self.dictionary.add(w)
        try:
            with open(self.user_words_file, "a", encoding="utf-8") as f:
                f.write(w + "\n")
            print(f"[Corrector] Learned new word: {w}")
        except Exception as e:
            print(f"[Corrector] Failed to save {w}: {e}")

    def word_frequency_score(self, word):
        """Return a prior score for the word based on user words, dictionary, or wordfreq."""
        if word in self.user_words:
            return 80.0   # Highest priority: user has written this word before
        if word in ALPHANUM_PATTERNS:
            return 40.0
        if word in self.dictionary:
            return 30.0

        if HAS_WORDFREQ:
            try:
                # Average fr and en Zipf frequencies (ranges ~1 to 7)
                fr = wordfreq.zipf_frequency(word.lower(), 'fr')
                en = wordfreq.zipf_frequency(word.lower(), 'en')
                m = max(fr, en)
                if m > 0:
                    return m * 4.0
            except Exception:
                pass

        return 0.0

    def correct(self, char_candidates, mode="auto"):
        """
        Input:
          char_candidates: list of lists of (char, prob) sorted by prob desc
                           e.g. [ [('A', 0.92), ('H', 0.08)], [('N', 0.75), ('M', 0.20)], ... ]
          mode: "auto" (letters + context digits), "digits" (force digits), "letters" (force letters)
        Returns:
          (best_word, alternatives_list)
        """
        if not char_candidates:
            return "", []

        # Normalize in case a list of dicts was passed: [{'A': 0.8, 'B': 0.1}, ...]
        normalized_candidates = []
        for cands in char_candidates:
            if isinstance(cands, dict):
                normalized_candidates.append(sorted(cands.items(), key=lambda x: x[1], reverse=True))
            elif isinstance(cands, list):
                normalized_candidates.append(cands)
            else:
                normalized_candidates.append([])
        char_candidates = normalized_candidates

        # 1. Base argmax
        raw_chars = [cands[0][0] if cands else '?' for cands in char_candidates]
        raw_word = "".join(raw_chars)
        raw_probs = [cands[0][1] if cands else 1.0 for cands in char_candidates]

        # Mode: digits forced
        if mode == "digits":
            # If the user happens to have written a registered user word / proper noun (e.g. AZIZ, SALUT, GAYETH),
            # don't ruin it into mutant pseudo-numbers like A212!
            as_letters = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)
            if as_letters in self.user_words or as_letters in self.dictionary:
                alts = [raw_word] if raw_word != as_letters else []
                return as_letters, alts

            digit_chars = [CONFUSION_LETTER_TO_DIGIT.get(ch, ch) for ch in raw_chars]
            best_dig = "".join(digit_chars)
            alts = [raw_word] if raw_word != best_dig else []
            return best_dig, alts

        # Mode: letters forced
        if mode == "letters":
            letter_chars = [CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars]
            best_let = "".join(letter_chars)
            alts = [raw_word] if raw_word != best_let else []
            return best_let, alts

        # -------------------------------------------------------------
        # Mode: AUTO (unified contextual resolution: words vs numbers)
        # -------------------------------------------------------------
        # 1. Alphanumeric patterns (A4, PS5, etc.)
        if raw_word in ALPHANUM_PATTERNS:
            return raw_word, []

        # 2. Check if converting confused digits to letters yields a registered word or user word
        # (e.g. A212 / AZ1Z -> AZIZ, 5ALUT -> SALUT, B8NJOUR -> BONJOUR, 6AYE7H -> GAYETH)
        as_letters = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)
        if as_letters in self.user_words or as_letters in self.dictionary:
            alts = [raw_word] if raw_word != as_letters else []
            return as_letters, alts

        # 3. Check if converting confused letters to digits yields a clean numeric sequence
        # (e.g. 2O25 -> 2025, ZOZS -> 2025, 4Z -> 42, 1Z3 -> 123)
        as_digits = "".join(CONFUSION_LETTER_TO_DIGIT.get(ch, ch) for ch in raw_chars)
        if as_digits.isdigit():
            num_digits = sum(1 for c in raw_chars if c.isdigit())
            can_be_digits = all(c.isdigit() or (c in CONFUSION_LETTER_TO_DIGIT) for c in raw_chars)
            if can_be_digits and (num_digits >= max(1, len(raw_chars) * 0.35) or raw_word.isdigit()):
                alts = [raw_word] if raw_word != as_digits else []
                return as_digits, alts

        # 4. Contextual letter preference: if a sequence is mostly alphabetic (e.g. ZAK where 2 of 3 are letters),
        # convert any isolated digit that has a letter confusion (e.g. 2AK -> ZAK)
        num_alpha = sum(1 for c in raw_chars if c.isalpha())
        if num_alpha >= len(raw_chars) * 0.6:
            cleaned_letters = "".join(CONFUSION_DIGIT_TO_LETTER.get(ch, ch) for ch in raw_chars)
            if cleaned_letters in self.user_words or cleaned_letters in self.dictionary:
                return cleaned_letters, [raw_word] if raw_word != cleaned_letters else []

        # -------------------------------------------------------------
        # Beam search over top candidate characters
        # -------------------------------------------------------------
        # Keep top 3 candidates per position
        beams = [("", 0.0, True)]  # (sequence_str, log_prob_sum, all_high_conf)

        for i, cands in enumerate(char_candidates):
            new_beams = []
            top_cands = cands[:3] if cands else [('?', 1.0)]

            # Check if primary candidate has very high confidence (> 0.88)
            is_locked = top_cands[0][1] >= 0.88

            for seq, score, locked_so_far in beams:
                for rank, (char, prob) in enumerate(top_cands):
                    p = max(float(prob), 1e-4)
                    penalty = 0.0
                    # If this position was locked to rank 0, penalize picking rank 1 or 2
                    if is_locked and rank > 0:
                        penalty = -15.0

                    new_seq = seq + char
                    new_score = score + p * 10.0 + penalty
                    new_beams.append((new_seq, new_score, locked_so_far and (rank == 0)))

            # Prune to top 15
            new_beams.sort(key=lambda b: b[1], reverse=True)
            beams = new_beams[:15]

        # Score final candidate words with language dictionary priors
        ranked_candidates = []
        seen = set()

        for seq, ocr_score, is_exact_raw in beams:
            if seq in seen:
                continue
            seen.add(seq)

            lex_bonus = self.word_frequency_score(seq)
            total_score = ocr_score + lex_bonus

            # Bonus for exact raw argmax if it didn't get lex bonus
            if seq == raw_word:
                total_score += 5.0

            ranked_candidates.append((seq, total_score))

        # Sort by total score descending
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        best_word = ranked_candidates[0][0]
        alts = [w for w, s in ranked_candidates[1:6] if w != best_word]

        # -------------------------------------------------------------
        # Fuzzy dictionary recovery (if best_word is not in dictionary/user_words)
        # Recovers common air-writing misreadings like SALDT/SALOT/SALVT -> SALUT
        # -------------------------------------------------------------
        if best_word not in self.dictionary and best_word not in self.user_words:
            best_dict_match = None
            best_dict_score = -1.0
            CONFUSION_PAIRS = {
                ('D', 'U'), ('O', 'U'), ('0', 'U'), ('V', 'U'),
                ('U', 'V'), ('V', 'U'),
                ('Q', 'A'), ('H', 'A'), ('4', 'A'),
                ('B', 'E'), ('8', 'B'),
                ('1', 'I'), ('7', 'T'), ('5', 'S'), ('2', 'Z'),
                ('I', 'L'), ('1', 'L'), ('T', 'L')
            }
            w_len = len(best_word)
            for dict_w in self.dictionary:
                if len(dict_w) == w_len:
                    diffs = [j for j in range(w_len) if best_word[j] != dict_w[j]]
                    if len(diffs) == 1:
                        j = diffs[0]
                        orig_c = best_word[j]
                        target_c = dict_w[j]
                        cand_chars = [c[0] for c in char_candidates[j]] if j < len(char_candidates) else []
                        if target_c in cand_chars:
                            p = dict(char_candidates[j]).get(target_c, 0.1)
                            score = self.word_frequency_score(dict_w) + p * 20.0
                            if score > best_dict_score:
                                best_dict_score = score
                                best_dict_match = dict_w
                        elif (orig_c, target_c) in CONFUSION_PAIRS:
                            score = self.word_frequency_score(dict_w) + 15.0
                            if score > best_dict_score:
                                best_dict_score = score
                                best_dict_match = dict_w

            if best_dict_match and best_dict_score > 0:
                if best_word not in alts:
                    alts.insert(0, best_word)
                best_word = best_dict_match

        # Always ensure raw_word is accessible in alternatives if different
        if raw_word != best_word and raw_word not in alts:
            alts.insert(0, raw_word)

        return best_word, alts
