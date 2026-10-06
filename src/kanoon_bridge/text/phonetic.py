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
    """Phonetic code tuned for Roman Hindi.

    TODO(C): before coding, normalise (text.transliterate.normalize_roman), drop 'h' after
    consonants (aspiration), map consonant groups that Hindi speakers interchange
    (v/w, z/j, f/ph, q/k), keep the first letter, and do NOT truncate to 4 characters
    (Hindi legal words are short, truncation over-merges). Test on tests/test_phonetic.py.
    """
    raise NotImplementedError("TODO(C): Hindi-aware phonetic code")


@dataclass
class PhoneticIndex:
    """code -> words in the vocabulary that share it."""

    buckets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def build(cls, vocabulary: list[str]) -> "PhoneticIndex":
        """TODO(C): bucket every Roman-script word by hindi_soundex."""
        raise NotImplementedError("TODO(C): build phonetic buckets")

    def matches(self, word: str) -> set[str]:
        """Vocabulary words that sound like `word`. TODO(C)."""
        raise NotImplementedError("TODO(C): phonetic lookup")
