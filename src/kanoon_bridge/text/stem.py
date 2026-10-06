"""Stemming for English and Hindi.  [owner: C]

English: Porter stemmer (NLTK) — working.
Hindi:   light suffix stripper — TODO(C).

Section tokens ("sec:ipc:302") and offence tokens ("off:murder") are never stemmed.
The ablation in eval/ablation.py compares stemming vs no stemming, so keep `enabled`.
"""

from __future__ import annotations

from functools import lru_cache

from nltk.stem import PorterStemmer

_porter = PorterStemmer()

# Common Hindi inflectional suffixes, longest first. TODO(C): check against a Hindi stemmer paper
# (e.g. Ramanathan & Rao's light stemmer) and test on legal terms.
HINDI_SUFFIXES: tuple[str, ...] = (
    "ियों", "ाओं", "ाएं", "ाएँ", "ियां", "ियाँ", "ों", "ें", "ीं", "ता", "ती", "ते", "ना", "नी", "ने", "ा", "ी", "े",
)


def _protected(token: str) -> bool:
    return token.startswith(("sec:", "off:")) or token.isdigit()


@lru_cache(maxsize=200_000)
def stem_english(token: str) -> str:
    return token if _protected(token) else _porter.stem(token)


def stem_hindi(token: str) -> str:
    """Strip one Hindi suffix from a Devanagari token.

    TODO(C): implement using HINDI_SUFFIXES (keep a minimum stem length of 2 characters),
    then add cases to tests/test_stem.py. Romanised Hindi goes through text/transliterate.py
    first, not through here.
    """
    raise NotImplementedError("TODO(C): Hindi light stemmer")


def _is_devanagari(token: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in token)


def stem_tokens(tokens: list[str], lang: str = "en", enabled: bool = True) -> list[str]:
    """Stem each token with the right stemmer for its script."""
    if not enabled:
        return tokens
    out = []
    for t in tokens:
        if _protected(t):
            out.append(t)
        elif _is_devanagari(t):
            try:
                out.append(stem_hindi(t))
            except NotImplementedError:
                out.append(t)
        else:
            out.append(stem_english(t))
    return out
