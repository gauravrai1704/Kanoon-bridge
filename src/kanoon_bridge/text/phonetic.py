"""Soundex-style phonetic codes for Romanised Hindi.  [owner: C — working]

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


# Consonants Hindi speakers interchange in Roman spelling share a class (after normalize_roman
# has already merged aspirates, w/v, q/k, z/j and ph/f).
_CLASSES = {"c": "c", "s": "s", "k": "k", "g": "g", "j": "j", "t": "t", "d": "d", "p": "p", "b": "b", "f": "f",
            "v": "v", "m": "m", "n": "n", "r": "r", "l": "l", "y": "y", "h": "h", "x": "k"}


def hindi_soundex(word: str) -> str:
    """Phonetic code tuned for Roman Hindi.

    1. normalise the spelling (text.transliterate.normalize_roman: aspirates, vowel length,
       doubled letters, w/v, z/j, q/k, ph/f)
    2. keep the first letter; then keep consonant classes, dropping vowels
       ("sh" folds into "s"; "h" after a vowel is kept only at the start)
    3. collapse repeated codes, and do NOT truncate (Hindi legal words are short;
       truncation would over-merge)

        hindi_soundex("hatya") == hindi_soundex("hathya") == hindi_soundex("hattya") == "hty"
    """
    from kanoon_bridge.text.transliterate import normalize_roman

    w = normalize_roman(word).replace("sh", "s")
    if not w:
        return ""
    out = [w[0]]
    for ch in w[1:]:
        code = _CLASSES.get(ch)
        if code is None or code == "h":
            continue
        if out[-1] != code:
            out.append(code)
    return "".join(out)


@dataclass
class PhoneticIndex:
    """code -> words in the vocabulary that share it."""

    buckets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def build(cls, vocabulary: list[str], min_len: int = 3) -> "PhoneticIndex":
        """Bucket every Roman-script word (letters only, min_len+) by hindi_soundex.
        Section/offence tokens and words with digits are skipped."""
        idx = cls()
        for word in vocabulary:
            if len(word) >= min_len and word.isascii() and word.isalpha():
                idx.buckets[hindi_soundex(word)].add(word)
        idx.buckets = dict(idx.buckets)
        return idx

    def matches(self, word: str) -> set[str]:
        """Vocabulary words that sound like `word` (the word itself excluded)."""
        if not word or not word.isascii():
            return set()
        return set(self.buckets.get(hindi_soundex(word), set())) - {word}
