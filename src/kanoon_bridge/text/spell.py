"""Tolerant retrieval: spelling correction, wildcard terms, "did you mean".  [owner: Gaurav — working]

Lecture: IIR ch. 3 (Manning, Raghavan & Schütze) — k-gram indexes, edit distance, isolated-term
correction, wildcard queries via k-grams.

    k-gram index   every vocabulary term is broken into character bigrams with boundary marks
                   ("$m mu ur rd de er r$"); gram -> terms. Used for
                   * candidate generation: terms sharing enough bigrams with the typo
                     (Jaccard >= 0.3; Zobel & Dart, "Finding approximate matches in large
                     lexicons", Software: Practice & Experience 1995)
                   * wildcards: "extort*" -> terms holding the grams $e ex xt to or rt, then
                     post-filtered with the full pattern (IIR §3.2.2)
    edit distance  Damerau-Levenshtein (Damerau 1964: insert, delete, substitute, transpose
                   adjacent letters — the four errors behind most typos; Kukich 1992 survey),
                   bounded: 1 edit for words under 8 letters, 2 from 8 letters
    ranking        fewest edits, then whole-word matches before stem matches, then same first
                   letter, then highest document frequency
                   (the corpus is the "dictionary"; a correction must occur in >= min_df docs)

When a word is corrected (analyzer step 5b): the query term is UNSEEN in both indexes; the
correction is ADDED to the query with weight 0.9 (the typo stays, harmlessly scoring 0), so a
rare-but-real word is never lost. Rare words (df <= 2) with a far more frequent neighbour get a
"did you mean" suggestion only. Section tokens, numbers, offence ids, Hinglish function words,
lexicon hits and short words are never corrected.

Terms here are INDEX terms (Porter stems); `surface` maps a stem back to its most frequent
spelling for display ("dowri" -> "dowry").
"""

from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from dataclasses import dataclass, field

_WORDLIKE = re.compile(r"^[a-z]+$")


def bigrams(term: str) -> list[str]:
    t = f"${term}$"
    return [t[i:i + 2] for i in range(len(t) - 1)]


def damerau_levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Optimal-string-alignment distance; returns max_dist + 1 as soon as it is exceeded."""
    if a == b:
        return 0
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[-1]


@dataclass
class KGramIndex:
    """Character-bigram index over a vocabulary (with $ boundary marks)."""

    grams: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def build(cls, terms) -> "KGramIndex":
        idx = cls()
        for t in terms:
            for g in set(bigrams(t)):
                idx.grams[g].add(t)
        idx.grams = dict(idx.grams)
        return idx

    def candidates(self, term: str, min_jaccard: float = 0.3, max_len_diff: int = 2) -> list[tuple[str, float]]:
        """Terms whose bigram set overlaps `term`'s with Jaccard >= min_jaccard."""
        q = set(bigrams(term))
        counts: dict[str, int] = defaultdict(int)
        for g in q:
            for t in self.grams.get(g, ()):
                if abs(len(t) - len(term)) <= max_len_diff:
                    counts[t] += 1
        out = []
        for t, inter in counts.items():
            jac = inter / (len(q) + len(set(bigrams(t))) - inter)
            if jac >= min_jaccard:
                out.append((t, jac))
        return out

    def wildcard(self, pattern: str) -> set[str]:
        """Terms matching a pattern with '*' (any letters): k-gram intersection, then fnmatch."""
        pattern = pattern.lower()
        padded = f"${pattern}$"
        pieces = [p for p in padded.split("*") if p]
        needed: set[str] = set()
        for p in pieces:
            needed.update(p[i:i + 2] for i in range(len(p) - 1))
        if not needed:
            return set()
        sets = sorted((self.grams.get(g, set()) for g in needed), key=len)
        found = set(sets[0])
        for s in sets[1:]:
            found &= s
            if not found:
                return set()
        return {t for t in found if fnmatch.fnmatchcase(t, pattern)}


@dataclass
class Correction:
    word: str          # as typed (lower-cased)
    source: str        # its analysed form (the unseen / rare index term)
    term: str          # index term it maps to
    display: str       # readable spelling for "did you mean"
    edits: int
    df: int
    applied: bool      # True: added to the query; False: suggestion only


@dataclass
class Speller:
    df: dict[str, int]                                   # index term -> document frequency
    surface: dict[str, str] = field(default_factory=dict)          # stem -> display spelling
    words: dict[str, str] = field(default_factory=dict)            # corpus spelling -> stem
    min_df: int = 2
    min_len: int = 4
    rare_df: int = 2                                     # df at or below this may get a suggestion
    rare_ratio: float = 50.0                             # ... if a neighbour is this many times more frequent
    _kgrams: KGramIndex | None = None
    _wild: KGramIndex | None = None
    _word_term: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, df: dict[str, int], surface: dict | None = None, **kw) -> "Speller":
        """`surface`: the dict saved by scripts/02 ({"display": ..., "words": ...})."""
        surface = surface or {}
        if "display" in surface or "words" in surface:
            return cls(df=df, surface=surface.get("display", {}), words=surface.get("words", {}), **kw)
        return cls(df=df, surface=surface, **kw)

    @property
    def kgrams(self) -> KGramIndex:
        """Bigram index over frequent index terms AND their surface spellings, so a typo is
        compared both with stems ("murdr" ~ "murder") and with whole words ("punishmnt" ~
        "punishment", whose stem "punish" is 3 edits away)."""
        if self._kgrams is None:                         # built on first use
            terms = {t for t, n in self.df.items() if n >= self.min_df and _WORDLIKE.match(t)}
            pairs = dict(self.words) or {w: st for st, w in self.surface.items()}
            pairs = {w: st for w, st in pairs.items() if self.df.get(st, 0) >= self.min_df and _WORDLIKE.match(w)}
            self._kgrams = KGramIndex.build(terms | set(pairs))
            self._word_term = pairs
        return self._kgrams

    def _term_of(self, cand: str) -> str:
        """Index term for a candidate: itself if it is a term, else the stem of a surface word."""
        if self.df.get(cand, 0) >= self.min_df:
            return cand
        return self._word_term.get(cand, cand)

    @property
    def wild_index(self) -> KGramIndex:
        """Wildcards match spellings people type (surface forms) as well as stems."""
        if self._wild is None:
            words = ({t for t in self.df if _WORDLIKE.match(t)} | {w for w in self.surface.values() if _WORDLIKE.match(w)}
                     | set(self.words))
            self._wild = KGramIndex.build(words)
        return self._wild

    def display(self, term: str) -> str:
        return self.surface.get(term, term)

    @staticmethod
    def max_edits(term: str) -> int:
        return 1 if len(term) < 8 else 2

    def best(self, term: str, word: str | None = None) -> tuple[str, int, str] | None:
        """Closest frequent index term within the edit budget, or None. The typed `word` is
        compared with surface spellings, its analysed `term` with index terms."""
        scored = []
        kg = self.kgrams
        for rank, probe in enumerate(dict.fromkeys(p for p in (word, term) if p)):
            limit = self.max_edits(probe)
            for cand, _jac in kg.candidates(probe, max_len_diff=limit):
                if cand in (term, word):
                    continue
                d = damerau_levenshtein(probe, cand, limit)
                if d <= limit:
                    target = self._term_of(cand)
                    if target != term:
                        shown = cand if cand in self._word_term else self.display(target)
                        # fewest edits; then the typed word's match over the stem's; same first
                        # letter; most frequent
                        scored.append((d, rank, cand[:1] != probe[:1], -self.df.get(target, 0), target, shown))
        if not scored:
            return None
        d, _, _, _neg_df, target, shown = min(scored)
        return target, d, shown

    def check(self, word: str, term: str) -> Correction | None:
        """Correction for one query word (`term` = its analysed form), or None."""
        if len(term) < self.min_len or not _WORDLIKE.match(term):
            return None
        n = self.df.get(term, 0)
        if n > self.rare_df:
            return None
        found = self.best(term, word)
        if found is None:
            return None
        cand, d, shown = found
        cdf = self.df.get(cand, 0)
        if n == 0:
            return Correction(word, term, cand, shown, d, cdf, applied=True)
        if cdf >= self.rare_ratio * n and d == 1:
            return Correction(word, term, cand, shown, d, cdf, applied=False)
        return None

    def expand_wildcard(self, pattern: str, limit: int = 30) -> list[str]:
        """Index terms for a wildcard pattern, most frequent first (at most `limit`)."""
        from kanoon_bridge.text.stem import stem_english

        if len(pattern.replace("*", "")) < 2:
            return []
        terms = set()
        for w in self.wild_index.wildcard(pattern):
            for t in (w, self.words.get(w) or stem_english(w)):
                if self.df.get(t, 0) > 0:
                    terms.add(t)
        return sorted(terms, key=lambda t: (-self.df[t], t))[:limit]


def surface_forms(docs, min_count: int = 2, max_docs: int | None = None) -> dict[str, dict[str, str]]:
    """Spellings seen in the corpus:
        {"display": stem -> its most frequent spelling ("dowri" -> "dowry"),
         "words":   spelling -> stem, for every spelling seen at least `min_count` times}"""
    from collections import Counter

    from kanoon_bridge.text.stem import stem_english
    from kanoon_bridge.text.tokenize import tokenize

    counts: Counter = Counter()
    for i, d in enumerate(docs):
        if max_docs is not None and i >= max_docs:
            break
        for p in d.paragraphs:
            counts.update(w for w in tokenize(p.text, keep_sections=False) if len(w) >= 3 and _WORDLIKE.match(w))
    display: dict[str, tuple[int, str]] = {}
    words: dict[str, str] = {}
    for w, n in counts.items():
        st = stem_english(w)
        if n >= min_count:
            words[w] = st
        if st not in display or n > display[st][0]:
            display[st] = (n, w)
    return {"display": {st: w for st, (_, w) in display.items()}, "words": words}
