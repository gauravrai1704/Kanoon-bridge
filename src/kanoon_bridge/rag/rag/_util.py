"""Small helpers shared by the RAG modules."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

# split after . ! ? (not before a citation "[n]") and after a citation that closes a sentence ". [1] Next"
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(0-9])|(?<=[.!?]\s\[\d\])\s+(?=[A-Z\"'(])|(?<=[.!?]\s\[\d\d\])\s+(?=[A-Z\"'(])")
_CITE_RE = re.compile(r"\[(\d+)\]")
_res = None


def text_resources():
    """The shared text pipeline resources (loaded once)."""
    global _res
    if _res is None:
        from kanoon_bridge.text.pipeline import TextResources

        _res = TextResources.load()
    return _res


def terms(text: str) -> list[str]:
    from kanoon_bridge.text.pipeline import analyze_text

    return analyze_text(text, text_resources())


def idf_lookup(idf) -> Callable[[str], float]:
    """Accept a dict term -> idf, a callable, or None (all terms weigh 1)."""
    if idf is None:
        return lambda t: 1.0
    if callable(idf):
        return idf
    default = max(idf.values(), default=1.0) or 1.0      # unseen term = as rare as the rarest seen
    return lambda t: idf.get(t, default)


def tfidf(tokens: list[str], idf: Callable[[str], float]) -> dict[str, float]:
    return {t: (1 + math.log10(c)) * idf(t) for t, c in Counter(tokens).items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


_ABBREV_RE = re.compile(r"(?:\b(?:v|vs|no|nos|sec|secs|s|ss|art|ltd|co|dr|mr|mrs|ms|smt|sh|j|jj|cj|hon'?ble|ors|anr|etc|viz|i\.e|e\.g|cr\.?p\.?c|i\.p\.c)|\b[A-Z])\.$", re.I)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    out: list[str] = []
    for piece in (p.strip() for p in _SENT_RE.split(text) if p and p.strip()):
        if out and _ABBREV_RE.search(out[-1]):          # "State v. Arjun", "Sec. 302", "A. K. Roy"
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def citations(sentence: str) -> list[int]:
    return [int(n) for n in _CITE_RE.findall(sentence)]


def strip_citations(sentence: str) -> str:
    return _CITE_RE.sub("", sentence).strip()
