"""Judicial treatment of precedents: overruled, doubted, followed.  [owner: Gaurav — working]

The drift layer catches precedent made stale by a STATUTE change. A precedent can also be made
stale by a COURT: overruled, declared per incuriam, disapproved. Indian judgments have no
citator signals (no Shepard's / KeyCite), but the judgments say it in their own words, most
regularly in the headnotes:

    "Sita Ram Goel v. The Municipal Board, Kanpur & Ors., [1959] SCR 1148, overruled."
    "Karmi v. Amru (1972) 4 SCC 86 was per incuriam ..."
    "... is not good law and has been overruled by decisions of larger benches ..."

Pipeline (no test leakage: citing texts are the precedent pool and TRAIN-split queries only):
    1. sentences with a treatment cue (abbreviation-aware splitting, rag/_util.split_sentences)
    2. case names "A v. B" in the sentence; the cue is attached to the case it governs:
       "overruled (the decision) in B v. C" -> the case after; otherwise the nearest case name
       before the cue; negated cues ("not overruled", "cannot be said to be overruled") skipped
    3. a case name is linked to a corpus judgment by party key (first two content words of each
       party, from the judgment's title); ambiguous keys are not linked
    4. per judgment: strong negatives (overruled, per incuriam, no longer / not good law),
       weak negatives (doubted, disapproved), positives (followed, relied on, approved, affirmed),
       each with the citing judgment and sentence as evidence
    5. status: "overruled" if a strong negative comes from a judgment decided on or after it;
       "doubted" on weak negatives without a positive after them; else "good"

Ranking (SearchOptions.treatment): an overruled precedent loses mu_overruled x the score spread,
a doubted one mu_doubted; both stay in the list and are explained ("overruled in X (year)").
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

STRONG = re.compile(r"\b(over-?rul(?:ed|ing)|per\s+incuriam|no\s+longer\s+(?:a\s+)?good\s+law|not\s+(?:a\s+)?good\s+law)\b", re.I)
WEAK = re.compile(r"\b(doubted|disapprov(?:ed|ing))\b", re.I)
POS = re.compile(r"\b(followed|relied\s+(?:up)?on|approved|affirmed)\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|cannot|could\s+not|was\s+not|has\s+not|have\s+not|neither|nor)\b[^.;]{0,30}$", re.I)
_WORD = r"(?:[A-Z][\w.'&/-]*|of|and|the|&|alias|@|for)"
CASE = re.compile(rf"\b([A-Z][\w.'&/-]*(?:\s+{_WORD}){{0,8}})\s+[Vv](?:s|ersus)?\.?\s+({_WORD}(?:\s+{_WORD}){{0,8}})")
_STOP = {"of", "the", "and", "&", "ors", "ors.", "anr", "anr.", "others", "another", "alias", "@", "for", "m/s",
         "mr", "mrs", "smt", "shri", "sri", "dr", "ltd", "pvt", "co", "the", "state", "union", "india", "in", "re"}


def party_key(name: str) -> tuple[str, ...]:
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in name.split()]
    words = [w for w in words if w and w not in _STOP and len(w) > 1]
    return tuple(words[:2])


def case_key(petitioner: str, respondent: str) -> tuple:
    return party_key(petitioner), party_key(respondent)


def title_key(title: str) -> tuple | None:
    t = re.sub(r",\s*\d{4}\s*$", "", title or "")
    m = re.match(r"(.+?)\s+[Vv](?:s|ersus)?\.?\s+(.+)", t)
    if not m:
        return None
    k = case_key(m.group(1), m.group(2))
    return k if k[0] and k[1] else None


@dataclass
class Treatment:
    doc_id: str
    strong: list = field(default_factory=list)   # (citing_id, date, cue, sentence)
    weak: list = field(default_factory=list)
    positive: list = field(default_factory=list)
    status: str = "good"

    def explain(self, titles=None) -> str:
        if self.status == "good":
            return ""
        ev = (self.strong or self.weak)[0]
        by = (titles or {}).get(ev[0], ev[0])
        return f"{self.status} ({ev[2]}) in {by}" + (f", {ev[1][:4]}" if ev[1] else "")


def mentions(sentence: str) -> list[tuple[int, int, tuple]]:
    out = []
    for m in CASE.finditer(sentence):
        k = case_key(m.group(1), m.group(2))
        if k[0] and k[1]:
            out.append((m.start(), m.end(), k))
    return out


_AFTER = re.compile(r"\s*(?:the\s+(?:earlier\s+|majority\s+)?(?:decisions?|judgments?|views?)\s+(?:of\s+this\s+court\s+)?)?in\s+(?!view\b|the\s+light\b)", re.I)
_BLOCK = re.compile(r"\b(?:decision|judgment|view|provisions?|section|rule|whether|contention|argument|plea|objection|court|bench)s?\b", re.I)


def attach(sentence: str, cue: re.Match, cases: list) -> tuple | None:
    """The case a cue governs (see module docstring). Precision over recall: a cue is attached
    only to a case right after "overruled (the decision) in" / "overruling", or to a case
    named shortly before it with nothing between that could be the cue's real object ("the
    decision in <another case>", "the provisions of", "the contention")."""
    if _NEGATED.search(sentence[: cue.start()]):
        return None
    for c in cases:
        gap = sentence[cue.end(): c[0]]
        if 0 <= c[0] - cue.end() <= 60 and (_AFTER.fullmatch(gap) or (cue.group(1).lower().endswith("ing") and not gap.strip())):
            return c[2]
    before = [c for c in cases if c[1] <= cue.start() and cue.start() - c[1] <= 80]
    if not before:
        return None
    gap = sentence[before[-1][1]: cue.start()]
    return None if _BLOCK.search(gap) else before[-1][2]


def extract(text: str) -> list[tuple[str, tuple, str]]:
    """(kind, case key, sentence) for every treatment found in a judgment's text."""
    from kanoon_bridge.rag._util import split_sentences

    found = []
    for s in split_sentences(text):
        if len(s) > 800 or " v" not in s.lower():
            continue
        cases = mentions(s)
        if not cases:
            continue
        for kind, rx in (("strong", STRONG), ("weak", WEAK), ("positive", POS)):
            for cue in rx.finditer(s):
                k = attach(s, cue, cases)
                if k is not None:
                    found.append((kind, k, s, cue.group(1).lower()))
    return found


def build(docs) -> dict[str, Treatment]:
    """`docs`: iterable of schema.Document (the corpus). See module docstring for what is read."""
    from kanoon_bridge.schema import DocType

    docs = list(docs)
    pool = [d for d in docs if d.doc_type == DocType.PRECEDENT]
    index: dict[tuple, list[str]] = defaultdict(list)
    dates = {}
    for d in pool:
        k = title_key(d.title)
        if k:
            index[k].append(d.doc_id)
        dates[d.doc_id] = d.decision_date.isoformat() if d.decision_date else ""
    citing = pool + [d for d in docs if d.doc_type == DocType.QUERY_CASE and d.split == "train"]
    out: dict[str, Treatment] = {}
    for d in citing:
        text = " ".join(p.text for p in d.paragraphs)
        when = d.decision_date.isoformat() if d.decision_date else ""
        for kind, key, sentence, cue in extract(text):
            ids = index.get(key, [])
            if len(ids) != 1 or ids[0] == d.doc_id:
                continue
            t = out.setdefault(ids[0], Treatment(ids[0]))
            getattr(t, kind).append((d.doc_id, when, cue, sentence[:400]))
    for t in out.values():
        mine = dates.get(t.doc_id, "")
        later = [e for e in t.strong if not mine or not e[1] or e[1] >= mine]
        if later:
            t.status = "overruled"
        elif t.weak and not any(not e[1] or e[1] >= max(w[1] for w in t.weak) for e in t.positive):
            t.status = "doubted"
    return out


def save(table: dict[str, Treatment], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: vars(v) for k, v in table.items()}, f, ensure_ascii=False)


def load(path) -> dict[str, Treatment]:
    with open(path, encoding="utf-8") as f:
        return {k: Treatment(**v) for k, v in json.load(f).items()}


@dataclass
class TreatmentScorer:
    table: dict
    mu_overruled: float = 0.3
    mu_doubted: float = 0.1

    @classmethod
    def load(cls, cfg) -> "TreatmentScorer":
        from kanoon_bridge.config import project_path

        t = cfg.get("treatment") or {}
        return cls(load(project_path(cfg.paths.index_dir) / "treatment.json"),
                   t.get("mu_overruled", 0.3), t.get("mu_doubted", 0.1))

    def status(self, doc_id: str) -> str:
        t = self.table.get(doc_id)
        return t.status if t is not None else "good"

    def rerank(self, final: dict[str, float], components: dict, depth: int = 100) -> tuple[dict[str, float], list[str]]:
        if not final:
            return final, []
        head = sorted(final, key=final.get, reverse=True)[:depth]
        spread = max(final[head[0]] - final[head[-1]], abs(final[head[0]])) or 1.0
        out, hit = dict(final), []
        for d in head:
            st = self.status(d)
            mu = self.mu_overruled if st == "overruled" else self.mu_doubted if st == "doubted" else 0.0
            if mu:
                out[d] = final[d] - mu * spread
                components.setdefault(d, {})["treatment"] = -mu
                hit.append(d)
        return out, hit


__all__ = ["build", "extract", "save", "load", "TreatmentScorer", "Treatment", "title_key", "party_key"]
