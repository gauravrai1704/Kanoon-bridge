"""Stemming for English and Hindi.  [owner: C — working]

English: Porter stemmer (NLTK) — working.
Hindi:   light suffix stripper (Ramanathan & Rao 2003 style) — working.

Section tokens ("sec:ipc:302") and offence tokens ("off:murder") are never stemmed.
The ablation in eval/ablation.py compares stemming vs no stemming, so keep `enabled`.
"""

from __future__ import annotations

from functools import lru_cache

from nltk.stem import PorterStemmer

_porter = PorterStemmer()

# Hindi inflectional suffixes after Ramanathan & Rao (2003), "A Lightweight Stemmer for Hindi":
# noun plural/oblique endings, verb aspect/tense endings, adjective agreement. Longest match first.
HINDI_SUFFIXES: tuple[str, ...] = tuple(sorted({
    "ियों", "ियां", "ियाँ", "ाओं", "ाएं", "ाएँ", "ुओं", "ुएं", "ुएँ", "ाइयों", "ाइयां", "ाइयाँ",
    "ाऊंगा", "ाऊंगी", "ाएगा", "ाएगी", "ाओगे", "ाओगी", "ेंगे", "ेंगी", "ूंगा", "ूंगी", "ोगे", "ोगी", "ेगा", "ेगी",
    "ाकर", "ाते", "ाती", "ाता", "ाना", "ाने", "ानी", "ाया", "ाये", "ाई", "ाए", "ावा",
    "कर", "ता", "ती", "ते", "ना", "नी", "ने", "या", "ये", "ई", "ए",
    "ों", "ें", "ीं", "ां", "ाँ", "ा", "ी", "े", "ो", "ि", "ु", "ू",
}, key=len, reverse=True))
MIN_HINDI_STEM = 2                    # characters (code points) kept at least


def _protected(token: str) -> bool:
    return token.startswith(("sec:", "off:")) or token.isdigit()


@lru_cache(maxsize=200_000)
def stem_english(token: str) -> str:
    return token if _protected(token) else _porter.stem(token)


def stem_hindi(token: str) -> str:
    """Strip the longest matching Hindi suffix from a Devanagari token, keeping at least
    MIN_HINDI_STEM characters: लड़कियों -> लड़क, हत्याओं -> हत्य, मारता -> मार.
    Romanised Hindi goes through text/transliterate.py first, not through here."""
    for suffix in HINDI_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_HINDI_STEM:
            return token[: -len(suffix)]
    return token


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
