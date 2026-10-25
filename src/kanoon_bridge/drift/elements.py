"""The parts of an offence that decide guilt: mental state, thresholds, content.  [owner: Gaurav — working]

Indian criminal law states each offence as ingredients: an act, a circumstance and a mental
state ("dishonestly", "with intent to", "knowing it to be likely"). A precedent that reads one
ingredient carries over to the new code only if the ingredient survived. Three readings:

    mens_rea(text)    the mental-state words, normalised to a closed set of markers
                      (intention, knowledge, reason_to_believe, dishonesty, fraud, voluntariness,
                      rashness, negligence, malice, common_intention, common_object, wilfulness)
    thresholds(text)  quantities that are part of the definition, not the punishment: ages
                      ("under eighteen years of age"), periods ("within seven years of her
                      marriage"), amounts and counts ("five or more persons") -> {(kind, value)}
    content(text)     content-word stems (stop words and punishment boilerplate removed), the
                      vocabulary of the conduct the provision describes

compare_* functions return what was added and removed between the IPC and the BNS text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kanoon_bridge.drift.punishment import _NUM, words_to_number

MENS_REA = {
    "intention": r"\bintention(?:ally)?\b|\bintend(?:s|ing|ed)?\b|\bwith\s+(?:the\s+)?intent\b|\bintent\b",
    "knowledge": r"\bknowing(?:ly)?\b|\bknowledge\b|\bknows?\b",
    "reason_to_believe": r"\breason\s+to\s+believe\b",
    "dishonesty": r"\bdishonest(?:ly)?\b",
    "fraud": r"\bfraudulent(?:ly)?\b",
    "voluntariness": r"\bvoluntar(?:y|ily)\b",
    "rashness": r"\brash(?:ly|ness)?\b",
    "negligence": r"\bnegligen(?:t|tly|ce)\b",
    "malice": r"\bmalicious(?:ly)?\b|\bmalignant(?:ly)?\b|\bwanton(?:ly)?\b",
    "common_intention": r"\bcommon\s+intention\b",
    "common_object": r"\bcommon\s+object\b",
    "wilfulness": r"\bwil(?:l)?ful(?:ly)?\b|\bdeliberate(?:ly)?\b",
}
_MENS_REA_RE = {k: re.compile(v, re.I) for k, v in MENS_REA.items()}

# punishment boilerplate is compared by punishment.py, not here
_PUNISH_SPAN = re.compile(r"shall\s+be\s+punished.*?(?:\.|;|$)|shall\s+also\s+be\s+liable\s+to\s+fine|"
                          r"imprisonment\s+of\s+either\s+description|for\s+a\s+term\s+which\s+may\s+extend\s+to"
                          rf"|{_NUM}\s+(?:years?|months?)|fine\s+which\s+may\s+extend\s+to\s+{_NUM}\s+rupees", re.I | re.S)

_AGE = re.compile(rf"(?P<num>{_NUM})\s+years\s+of\s+age|age\s+of\s+(?P<num2>{_NUM})\s+years", re.I)
_PERIOD = re.compile(rf"within\s+(?P<num>{_NUM})\s+(?P<unit>years?|months?|days?|hours?)", re.I)
_COUNT = re.compile(rf"(?P<num>{_NUM})\s+or\s+more\s+(?P<what>persons|times|occasions)", re.I)
_AMOUNT = re.compile(rf"(?:value|amount)\s+of\s+(?:the\s+property\s+)?(?:stolen\s+)?(?:does\s+not\s+exceed|exceeding|exceeds|above|"
                     rf"less\s+than|more\s+than)\s+(?P<num>{_NUM})\s+rupees", re.I)

STOP = set("""a an the of and or to in on by for with as at be is are was were been being any such which who
whom whose that this these those it its he she him her his hers they them their there here shall may
will would should can could not no nor than then so if when where while whoever whatever every each
other another same said thereof therein thereby within without under upon into from also either both
person persons section sections sub clause code sanhita act offence offences commits committed commit
punished punishable punishment liable term extend description imprisonment fine rupees years year
months month life death""".split())


def mens_rea(text: str) -> set[str]:
    low = text or ""
    return {k for k, rx in _MENS_REA_RE.items() if rx.search(low)}


def thresholds(text: str) -> set[tuple[str, float]]:
    """{('age', 18.0), ('period_years', 7.0), ('count_persons', 5.0), ('amount', 5000.0)}"""
    out: set[tuple[str, float]] = set()
    low = " ".join((text or "").lower().split())
    for m in _AGE.finditer(low):
        n = words_to_number(m.group("num") or m.group("num2") or "")
        if n is not None:
            out.add(("age", n))
    for m in _PERIOD.finditer(low):
        n = words_to_number(m.group("num"))
        if n is not None:
            out.add((f"period_{m.group('unit').rstrip('s')}s", n))
    for m in _COUNT.finditer(low):
        n = words_to_number(m.group("num"))
        if n is not None:
            out.add((f"count_{m.group('what')}", n))
    for m in _AMOUNT.finditer(low):
        n = words_to_number(m.group("num"))
        if n is not None:
            out.add(("amount", n))
    return out


def _stem(w: str) -> str:
    for suf in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content(text: str) -> set[str]:
    """Content-word stems of the conduct described (punishment boilerplate removed)."""
    t = _PUNISH_SPAN.sub(" ", text or "")
    return {_stem(w) for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP}


@dataclass
class ElementDelta:
    mens_rea_added: set[str] = field(default_factory=set)
    mens_rea_removed: set[str] = field(default_factory=set)
    thresholds_added: set[tuple[str, float]] = field(default_factory=set)
    thresholds_removed: set[tuple[str, float]] = field(default_factory=set)
    content_jaccard: float = 1.0
    content_added: set[str] = field(default_factory=set)
    content_removed: set[str] = field(default_factory=set)

    @property
    def mens_rea_changed(self) -> bool:
        return bool(self.mens_rea_added or self.mens_rea_removed)

    @property
    def threshold_changed(self) -> bool:
        """A threshold of the same kind with a different value (15 -> 18 years of age)."""
        kinds_old = {k for k, _ in self.thresholds_removed}
        kinds_new = {k for k, _ in self.thresholds_added}
        return bool(kinds_old & kinds_new)


def compare(old_text: str, new_text: str) -> ElementDelta:
    mo, mn = mens_rea(old_text), mens_rea(new_text)
    to, tn = thresholds(old_text), thresholds(new_text)
    co, cn = content(old_text), content(new_text)
    union = co | cn
    return ElementDelta(mens_rea_added=mn - mo, mens_rea_removed=mo - mn,
                        thresholds_added=tn - to, thresholds_removed=to - tn,
                        content_jaccard=len(co & cn) / len(union) if union else 1.0,
                        content_added=cn - co, content_removed=co - cn)


__all__ = ["MENS_REA", "mens_rea", "thresholds", "content", "compare", "ElementDelta"]
