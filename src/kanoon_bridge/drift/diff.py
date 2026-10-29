"""Align and diff an old provision (IPC, possibly several merged sections) with its successor.  [owner: Gaurav — working]

Two measurements, both asymmetric on purpose:

    retention   share of the OLD text that survives in the new one   (what a precedent read)
    novelty     share of the NEW text that the old one did not have  (what no precedent read yet)

Each is computed per clause by word-bigram containment against the best-matching clause on
the other side (so merged or reordered sections still align), then averaged with clause
length as the weight. A clause whose best containment is below `match_min` counts as
added (new side) or lost (old side).

Before comparing, both texts are normalised the same way: lower case, IPC cross-references
renumbered into BNS numbering (provisions.renumber_references), the BNS's systematic wording
updates mapped to one form (COSMETIC), and punctuation dropped.

`token_diff` returns difflib opcodes over words, for showing a side-by-side diff in the app.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# systematic BNS wording updates that do not change substance (applied to both sides)
COSMETIC = [
    # the replaced Acts' names, old and new, as one token each (references to them are not changes)
    (r"\bindian penal code(?:,? 1860)?(?: \(45 of 1860\))?|\bbharatiya nyaya sanhita(?:,? 2023)?", "penalcode"),
    (r"\bcode of criminal procedure(?:,? 1973)?(?: \(2 of 1974\))?|\bbharatiya nagarik suraksha sanhita(?:,? 2023)?",
     "procedurecode"),
    (r"\bindian evidence act(?:,? 1872)?(?: \(1 of 1872\))?|\bbharatiya sakshya adhiniyam(?:,? 2023)?", "evidencecode"),
    (r"\bsanhita\b", "code"),
    (r"\badhiniyam\b", "act"),
    (r"\bperson of unsound mind\b", "unsound mind"),
    (r"\b(?:lunatic|idiot|insane)\b", "unsound mind"),
    (r"\btransportation for life\b", "imprisonment for life"),
    (r"\bstate government\b", "government"),
    (r"\bcentral government\b", "government"),
    (r"\bgovernment of india\b", "government"),
    (r"\bqueen\b|\bher majesty\b|\bthe crown\b", "government"),
    (r"\bbritish india\b", "india"),
    (r"\bthe said\b", "the"),
    (r"\bwhosoever\b", "whoever"),
    (r"₹\s*([\d,]+)", r"\1 rupees"),
]
_COSMETIC_RE = [(re.compile(a, re.I), b) for a, b in COSMETIC]


_LABEL = re.compile(r"^\s*(?:\(\s*(?:\d{1,2}[a-z]?|[a-z]|[ivx]{1,4})\s*\)\s*)+", re.I)


def normalise(text: str) -> list[str]:
    """Lower case, cosmetic updates mapped to one form, a leading clause label ("(1)", "(a)")
    dropped (numbering is not content), punctuation removed."""
    t = _LABEL.sub("", (text or "")).lower()
    for rx, rep in _COSMETIC_RE:
        t = rx.sub(rep, t)
    return re.findall(r"[a-z0-9]+", t)


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    if len(tokens) == 1:
        return {(tokens[0], "")}
    return set(zip(tokens, tokens[1:]))


def containment(a: list[str], b: list[str]) -> float:
    """Share of a's word bigrams that also occur in b."""
    ba = _bigrams(a)
    return len(ba & _bigrams(b)) / len(ba) if ba else 1.0


@dataclass
class Alignment:
    retention: float = 1.0                      # length-weighted mean containment of old clauses in new
    novelty: float = 0.0                        # 1 - length-weighted mean containment of new clauses in old
    novelty_matched: float = 0.0                # the same over new clauses that DO have an old counterpart
                                                # (rewording of old law, as opposed to added clauses)
    added_clauses: list[str] = field(default_factory=list)
    lost_clauses: list[str] = field(default_factory=list)
    pairs: list[tuple[int, int, float]] = field(default_factory=list)   # (new clause, best old clause, containment)
    old_scores: list[float] = field(default_factory=list)               # per old clause: best containment in new
    old_clauses: list[str] = field(default_factory=list)                # the clauses aligned (empty ones dropped)
    new_clauses: list[str] = field(default_factory=list)
    edit_similarity: float = 1.0                # difflib ratio over the whole normalised texts
    old_words: int = 0
    new_words: int = 0


def align(old_clauses: list[str], new_clauses: list[str], match_min: float = 0.3,
          min_clause_words: int = 6) -> Alignment:
    """Clause-level alignment (see module docstring). `pairs[i] = (i, j, c)`: new clause i's best
    old clause j and its containment c; `old_scores[j]`: old clause j's best containment in the new text."""
    old = [(c, normalise(c)) for c in old_clauses if normalise(c)]
    new = [(c, normalise(c)) for c in new_clauses if normalise(c)]
    al = Alignment(old_words=sum(len(t) for _, t in old), new_words=sum(len(t) for _, t in new))
    al.old_clauses, al.new_clauses = [c for c, _ in old], [c for c, _ in new]
    if not old or not new:
        al.retention = 0.0 if old else 1.0
        al.novelty = al.novelty_matched = 1.0 if new else 0.0
        return al
    old_all = [w for _, t in old for w in t]
    new_all = [w for _, t in new for w in t]
    # new -> old (novelty)
    num = den = num_m = den_m = 0.0
    for i, (clause, nt) in enumerate(new):
        scores = [containment(nt, ot) for _, ot in old]
        j = max(range(len(scores)), key=scores.__getitem__)
        best = max(scores[j], containment(nt, old_all))
        al.pairs.append((i, j, best))
        num += best * len(nt)
        den += len(nt)
        if best >= match_min:
            num_m += best * len(nt)
            den_m += len(nt)
        elif len(nt) >= min_clause_words:
            al.added_clauses.append(clause)
    al.novelty = 1.0 - num / den
    al.novelty_matched = 1.0 - num_m / den_m if den_m else 1.0
    # old -> new (retention)
    num = den = 0.0
    for clause, ot in old:
        best = max(max(containment(ot, nt) for _, nt in new), containment(ot, new_all))
        al.old_scores.append(best)
        num += best * len(ot)
        den += len(ot)
        if best < match_min and len(ot) >= min_clause_words:
            al.lost_clauses.append(clause)
    al.retention = num / den
    al.edit_similarity = difflib.SequenceMatcher(None, old_all, new_all, autojunk=False).ratio()
    return al


def edit_counts(old_text: str, new_text: str, renumber: dict[str, str] | None = None,
                stop: set[str] | None = None, typo_ratio: float = 0.8) -> tuple[int, int]:
    """(content words inserted, content words deleted) between two texts, in absolute numbers.

    Proportional measures (novelty, retention) cannot see a six-word qualifier added to a long
    section ("but within six hours from such arrest"), which is often the whole change in the
    procedure codes. Clause labels, numbers, renumbered references, stop words and one-word
    replacements that are near-identical strings (PDF damage such as "Judicial" -> "udicial")
    are not counted.
    """
    from kanoon_bridge.drift.elements import STOP

    stop = STOP if stop is None else stop

    def words(chunk: str) -> list[str]:
        return [w for w in normalise(chunk) if len(w) > 2 and not w.isdigit() and w not in stop]

    ins = dele = 0
    for op, a, b in token_diff(old_text, new_text, renumber):
        if op in ("equal", "renumbered"):
            continue
        wa, wb = words(a), words(b)
        if op == "replace" and len(wa) == len(wb) == 1 and \
                difflib.SequenceMatcher(None, wa[0], wb[0]).ratio() >= typo_ratio:
            continue
        ins += len(wb)
        dele += len(wa)
    return ins, dele


def token_diff(old_text: str, new_text: str, renumber: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
    """[(op, old words, new words)] over raw words, op in equal | replace | delete | insert |
    renumbered (a section number replaced by its BNS successor's number, per `renumber`)."""
    a, b = (old_text or "").split(), (new_text or "").split()
    na, nb = [normalise(w) or [w.lower()] for w in a], [normalise(w) or [w.lower()] for w in b]
    sm = difflib.SequenceMatcher(None, [" ".join(x) for x in na], [" ".join(x) for x in nb], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if op == "replace" and renumber:
            o, n = re.sub(r"[^\w]", "", old).lower(), re.sub(r"[^\w]", "", new).lower()
            if renumber.get(o) == n:
                op = "renumbered"
        out.append((op, old, new))
    return out


__all__ = ["COSMETIC", "normalise", "containment", "align", "Alignment", "token_diff", "edit_counts"]
