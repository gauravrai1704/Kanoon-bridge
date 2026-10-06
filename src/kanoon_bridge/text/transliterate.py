"""Language detection, transliteration and the Hinglish legal lexicon.  [owner: C — working]

Goal: "mere bhai ko chaku maara", "मेरे भाई को चाकू मारा" and "my brother was stabbed"
should reach the same statutes. Three pieces:

  detect_lang(text)          -> "en" | "hi" (Devanagari) | "hinglish" (Roman Hindi)
  devanagari_to_roman(text)  -> Roman form, so both scripts share one lexicon lookup
  normalize_roman(word)      -> spelling-normalised Roman Hindi ("maaraa" -> "mara")
  LegalLexicon.lookup(word)  -> English legal terms with weights ("hatya" -> murder)

Lexicon file: data/lexicons/hinglish_legal_terms.csv
    roman, devanagari, english, weight
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field

from kanoon_bridge.config import Config, load_config, project_path

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Very common Roman-Hindi function words and verbs, used only for language detection. None of
# them is an ordinary English word ("me", "to", "par" are left out on purpose).
_HINGLISH_MARKERS = {
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "mai", "main", "mera", "mere", "meri", "aur",
    "kya", "nahi", "nahin", "tha", "thi", "the", "kiya", "kar", "karna", "karke", "diya", "liya", "wala",
    "wali", "bhai", "behen", "maara", "mara", "maar", "pe", "ne", "wo", "woh", "yeh", "ye", "unhone",
    "humne", "maine", "mujhe", "hume", "usne", "uska", "uski", "unka", "kuch", "bahut", "gaya", "gayi",
    "raha", "rahi", "rahe", "hua", "hui", "ho", "hoga", "hogi", "chahiye", "kaise", "kab", "kyun", "kaun",
    "liye", "saath", "baad", "pehle", "ghar", "bhi", "agar", "lekin", "abhi", "kal", "aaj", "sakta",
    "sakti", "sakte", "mila", "mili", "de", "di", "dena", "lena", "jab", "tab", "apne", "apni", "hamare",
    "banta", "banti", "bante", "kitni", "kitna", "kitne",
}
_ENGLISH_CLASH = {"the", "de", "di", "main", "ho", "ye"}   # count these only next to other markers
# Markers that are pure function words (never looked up in the legal lexicon).
# "case" in "kya case banta hai" means "is there a case": filler inside a Hinglish question.
FUNCTION_WORDS = frozenset((_HINGLISH_MARKERS - {"maara", "mara", "maar", "ghar", "mila", "mili"})
                           | {"case", "kese", "kaisa"})


def detect_lang(text: str) -> str:
    """"hi" if any Devanagari; "hinglish" if at least 20% of the words are Roman-Hindi markers
    (words that are also English, like "the", count only when a clear marker is present)."""
    if _DEVANAGARI.search(text):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "en"
    clear = sum(w in _HINGLISH_MARKERS and w not in _ENGLISH_CLASH for w in words)
    if not clear:
        return "en"
    share = sum(w in _HINGLISH_MARKERS for w in words) / len(words)
    return "hinglish" if share >= 0.2 else "en"


# --------------------------------------------------------------------------- Devanagari -> Roman

_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri", "ए": "e", "ऐ": "ai",
           "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_MATRAS = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
           "ौ": "au", "ॉ": "o", "ॅ": "e"}
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    # precomposed nukta letters
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f", "य़": "y",
}
_NUKTA_FORMS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f"}   # ड़ as Hinglish "d" (ladki)
_NUKTA, _VIRAMA = "़", "्"
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "ॐ": "om"}
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def _translit_word(word: str) -> str:
    out: list[str] = []
    i = 0
    pending_a = False                      # inherent vowel of the previous consonant not yet written
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in _CONSONANTS:
            if pending_a:
                out.append("a")
            roman = _CONSONANTS[ch]
            if nxt == _NUKTA:
                roman = _NUKTA_FORMS.get(ch, roman)
                i += 1
            out.append(roman)
            pending_a = True
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
            pending_a = False
        elif ch == _VIRAMA:
            pending_a = False
        elif ch in _VOWELS:
            if pending_a:
                out.append("a")
            out.append(_VOWELS[ch])
            pending_a = False
        elif ch in _SIGNS:
            if pending_a:
                out.append("a")
            out.append(_SIGNS[ch])
            pending_a = False
        elif ch in _DIGITS:
            if pending_a:
                out.append("a")
            out.append(_DIGITS[ch])
            pending_a = False
        elif ch == _NUKTA:
            pass
        else:
            if pending_a:
                out.append("a")
            out.append(ch)
            pending_a = False
        i += 1
    # word-final schwa deletion (Hindi): "दहेज" -> "dahej", not "daheja"
    return "".join(out)


def devanagari_to_roman(text: str) -> str:
    """Transliterate Devanagari to a simple Roman scheme with our own character table
    (no external package): consonants carry an inherent "a" unless a matra or virama follows,
    nukta letters map to q/z/f and ड़ to d (Hinglish "ladki"), anusvara and chandrabindu become "n", and the word-final
    inherent vowel is dropped (Hindi schwa deletion: दहेज -> dahej, चाकू -> chaakuu).
    Non-Devanagari characters pass through. Feed the words to normalize_roman afterwards.
    """
    return re.sub(r"[ऀ-ॿ]+", lambda m: _translit_word(m.group(0)), text)


# Normalisation rules, applied in this order (put this table in the report):
NORMALISATION_RULES: tuple[tuple[str, str], ...] = (
    (r"chh", "ch"),                   # chhura / chura
    (r"ph", "f"),                     # phaansi / faansi
    (r"([kgjtdpb])h", r"\1"),         # aspiration: kh->k gh->g jh->j th->t dh->d ph(done) bh->b
    (r"w", "v"),                      # wakil / vakil
    (r"q", "k"),                      # qatl / katl
    (r"z", "j"),                      # zamanat / jamanat
    (r"ee|ii|ey(?=$|[^aeiou])", "i"), # long i: ee / ii; "-ey" ending
    (r"oo|uu", "u"),                  # long u
    (r"ou|au", "o"),                  # chouri / chori
    (r"(.)\1+", r"\1"),              # collapse doubles: aa->a, tt->t, ll->l
    (r"(?<=[^aeiou])h$", ""),         # final aspirate after a consonant
)
_RULES = [(re.compile(a), b) for a, b in NORMALISATION_RULES]


def normalize_roman(word: str) -> str:
    """Normalise a Roman Hindi spelling so variants meet:
    "maaraa"/"mara", "hatya"/"hathya"/"hattya", "zamanat"/"jamanat", "chhura"/"chura"."""
    w = re.sub(r"[^a-z]", "", word.lower())
    for pattern, repl in _RULES:
        w = pattern.sub(repl, w)
    return w


@dataclass
class LegalLexicon:
    """Roman/Devanagari legal terms -> English terms used in judgments and statutes."""

    entries: dict[str, list[tuple[str, float]]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def load(cls, cfg: Config | None = None) -> "LegalLexicon":
        """Load the CSV (working). Keys are stored exactly as written; lookup normalises."""
        cfg = cfg or load_config()
        lex = cls()
        path = project_path(cfg.paths.lexicon_hinglish)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(line for line in f if not line.startswith("#")):
                weight = float(row.get("weight") or 1.0)
                for key in (row.get("roman", ""), row.get("devanagari", "")):
                    key = key.strip().lower()
                    if key:
                        lex.entries[key].append((row["english"].strip().lower(), weight))
        return lex

    def _indexes(self) -> tuple[dict[str, list[tuple[str, float]]], dict[str, set[str]]]:
        """normalised key -> entries; phonetic code -> normalised keys (built on first use).
        Devanagari keys are transliterated first, so both scripts land on the same keys."""
        if getattr(self, "_norm", None) is None:
            from kanoon_bridge.text.phonetic import hindi_soundex

            norm: dict[str, list[tuple[str, float]]] = defaultdict(list)
            codes: dict[str, set[str]] = defaultdict(set)
            for key, entries in self.entries.items():
                roman = devanagari_to_roman(key) if _DEVANAGARI.search(key) else key
                nk = normalize_roman(roman)
                if not nk:
                    continue
                for e in entries:
                    if e not in norm[nk]:
                        norm[nk].append(e)
                codes[hindi_soundex(nk)].add(nk)
            self._norm, self._codes = dict(norm), dict(codes)
        return self._norm, self._codes

    def lookup(self, word: str, use_phonetic: bool = True) -> list[tuple[str, float]]:
        """English terms (with weights) for a Hindi/Hinglish word, trying in order:
        1. the exact key (Roman or Devanagari),
        2. its normalised Roman spelling (Devanagari is transliterated first),
        3. (use_phonetic) keys with the same Hindi phonetic code and a similar length
           (within 2 letters), weight x 0.8, only for words of 5+ letters so short words
           ("mere" vs "mara") do not over-match.
        Returns [] when nothing matches."""
        w = word.strip().lower()
        if not w:
            return []
        if w in self.entries:
            return list(self.entries[w])
        norm, codes = self._indexes()
        nk = normalize_roman(devanagari_to_roman(w) if _DEVANAGARI.search(w) else w)
        if nk in norm:
            return list(norm[nk])
        if use_phonetic and len(nk) >= 5:
            from kanoon_bridge.text.phonetic import hindi_soundex

            out: list[tuple[str, float]] = []
            for key in sorted(k for k in codes.get(hindi_soundex(nk), ()) if abs(len(k) - len(nk)) <= 2):
                out += [(term, round(wt * 0.8, 3)) for term, wt in norm[key] if (term, round(wt * 0.8, 3)) not in out]
            return out
        return []
