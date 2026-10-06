"""Soundex-style phonetic codes for Romanised Hindi.  [owner: C]

Classic Soundex was designed for English surnames and fails on Roman Hindi
("hatya" / "hathya" / "hattya", "chori" / "chouri"). We need our own code where
aspirated and unaspirated consonants, and vowel-length variants, collapse together.

    hindi_soundex("hatya") == hindi_soundex("hathya") == hindi_soundex("hattya")

`PhoneticIndex` maps codes to vocabulary words so a misspelt query word finds its
lexicon entry or index term.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def soundex(word: str) -> str:
    """Standard American Soundex (working) — the baseline to compare against in the report."""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    word = "".join(ch for ch in word.lower() if ch.isalpha())
    if not word:
        return ""
    out, last = word[0].upper(), codes.get(word[0], "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != last:
            out += code
        if ch not in "hw":
            last = code
    return (out + "000")[:4]


def hindi_soundex(word: str) -> str:
    """Phonetic code tuned for Romanised Hindi."""
    from kanoon_bridge.text.transliterate import normalize_roman

    word = normalize_roman(word.lower())
    if not word:
        return ""

    # Normalize common Hindi phonetic variants.
    replacements = (
        ("ph", "f"),
        ("bh", "b"),
        ("chh", "ch"),
        ("kh", "k"),
        ("gh", "g"),
        ("th", "t"),
        ("dh", "d"),
        ("q", "k"),
        ("w", "v"),
        ("z", "j"),
    )

    for old, new in replacements:
        word = word.replace(old, new)

    # Aspirated h should no longer affect the phonetic representation.
    word = word.replace("h", "")

    # Collapse repeated consonants.
    result = []
    for ch in word:
        if result and ch == result[-1] and ch not in "aeiou":
            continue
        result.append(ch)

    return "".join(result)



@dataclass
class PhoneticIndex:
    """code -> words in the vocabulary that share it."""

    buckets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def build(cls, vocabulary: list[str]) -> "PhoneticIndex":
        index = cls()

        for word in vocabulary:
            word = word.strip().lower()
            if word:
                code = hindi_soundex(word)
                if code:
                    index.buckets[code].add(word)

        return index

    def matches(self, word: str) -> set[str]:
        code = hindi_soundex(word)
        if not code:
            return set()
        return set(self.buckets.get(code, set()))