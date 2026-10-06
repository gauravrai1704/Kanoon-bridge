"""Language detection, transliteration and the Hinglish legal lexicon.  [owner: C]

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

# Small seed list of very common Roman-Hindi function words, used only for language detection.
# TODO(C): extend, or train a tiny char n-gram classifier on the COMI-LINGUA LID data.
_HINGLISH_MARKERS = {
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "mera", "mere", "meri", "aur",
    "kya", "nahi", "tha", "thi", "kiya", "kar", "diya", "liya", "wala", "bhai", "maara", "mara",
}


def detect_lang(text: str) -> str:
    """Rough language guess. Working baseline; TODO(C): improve and evaluate on E4 queries."""
    if _DEVANAGARI.search(text):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "en"
    share = sum(w in _HINGLISH_MARKERS for w in words) / len(words)
    return "hinglish" if share >= 0.2 else "en"


def devanagari_to_roman(text: str) -> str:
    """Convert common Devanagari Hindi text to a simple Roman representation."""

    consonants = {
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
        "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v",
        "श": "sh", "ष": "sh", "स": "s", "ह": "h",
        "ड़": "r", "ढ़": "rh",
        "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z",
        "फ़": "f", "ड़": "r", "ढ़": "rh",
    }

    vowels = {
        "अ": "a", "आ": "a", "इ": "i", "ई": "i",
        "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai",
        "ओ": "o", "औ": "au",
    }

    matras = {
        "ा": "a", "ि": "i", "ी": "i",
        "ु": "u", "ू": "u", "ृ": "ri",
        "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    }

    result = []
    i = 0

    while i < len(text):
        ch = text[i]

        # Preserve spaces and punctuation.
        if ch.isspace() or not _DEVANAGARI.search(ch):
            result.append(ch)
            i += 1
            continue

        if ch in vowels:
            result.append(vowels[ch])
            i += 1
            continue

        if ch in consonants:
            roman = consonants[ch]

            # Look for a following matra.
            if i + 1 < len(text) and text[i + 1] in matras:
                result.append(roman + matras[text[i + 1]])
                i += 2
                continue

            # Virama means no inherent vowel.
            if i + 1 < len(text) and text[i + 1] == "्":
                result.append(roman)
                i += 2
                continue

            # Default inherent vowel.
            result.append(roman + "a")
            i += 1
            continue

        # Anusvara / chandrabindu — approximate as nasal n.
        if ch in {"ं", "ँ"}:
            result.append("n")
            i += 1
            continue

        if ch == "ः":
            result.append("h")
            i += 1
            continue

        # Unknown Devanagari character: preserve it.
        result.append(ch)
        i += 1

    return normalize_roman("".join(result))

def normalize_roman(word: str) -> str:
    """Normalise common Roman-Hindi spelling variants.

    The goal is to make different spellings of the same Hindi word
    converge before lexicon and phonetic lookup.
    """
    word = word.lower().strip()

    if not word:
        return ""

    # Normalize repeated vowels / vowel length.
    word = re.sub(r"a{2,}", "a", word)
    word = re.sub(r"e{2,}", "i", word)
    word = re.sub(r"i{2,}", "i", word)
    word = re.sub(r"o{2,}", "u", word)
    word = re.sub(r"u{2,}", "u", word)

    # Common Roman-Hindi spelling variants.
    replacements = (
        ("chh", "ch"),
        ("ph", "f"),
        ("th", "t"),
        ("dh", "d"),
        ("kh", "k"),
        ("gh", "g"),
        ("bh", "b"),
        ("sh", "s"),
        ("w", "v"),
    )

    for old, new in replacements:
        word = word.replace(old, new)

    # Collapse repeated consonants:
    # maara -> mara, hattya -> hatya
    word = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+", r"\1", word)

    return word


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

    def lookup(self, word: str) -> list[tuple[str, float]]:
        """Look up a Hindi/Hinglish word in the legal lexicon."""
        if not word:
            return []
    
        # 1. Exact lookup
        key = word.strip().lower()
        if key in self.entries:
            return list(self.entries[key])
    
        # 2. Normalized Roman lookup
        normalized = normalize_roman(key)
        if normalized in self.entries:
            return list(self.entries[normalized])
    
        # 3. Phonetic lookup
        from kanoon_bridge.text.phonetic import PhoneticIndex
    
        vocabulary = list(self.entries.keys())
        phonetic_index = PhoneticIndex.build(vocabulary)
        matches = phonetic_index.matches(normalized)
    
        results: list[tuple[str, float]] = []
    
        for match in matches:
            results.extend(self.entries.get(match, []))
    
        return results