"""What punishment a provision prescribes, read from its bare-act text.  [owner: Gaurav — working]

    extract("... shall be punished with imprisonment of either description for a term which may
             extend to one year, or with fine which may extend to one thousand rupees, or with both")
      -> Punishment(max_years=1.0, fine=True, fine_max=1000, kinds={"imprisonment", "fine"})

Read: death, imprisonment for life, the longest maximum term and the longest minimum term
(years; months and days converted), rigorous / simple / either description, fine with its
largest maximum and minimum (rupees, "lakh" and "crore" included), community service.
A section with several sub-sections gets the union of what they prescribe (the largest maximum,
the largest minimum), which is what changes when the BNS raises a limit in one sub-section.

compare(old, new) -> PunishmentDelta: which fields changed and whether the change is harsher.
Number words ("two thousand and five hundred") are parsed by `words_to_number`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000,
           "crores": 10_000_000}
_NUMWORD = "|".join(sorted(list(_UNITS) + list(_SCALES), key=len, reverse=True))
_NUM = rf"(?:(?:\d[\d,]*(?:\.\d+)?)|(?:(?:{_NUMWORD})(?:[\s-]+(?:and\s+)?(?:{_NUMWORD}))*))"
_TERM = re.compile(rf"(?P<kw>may\s+extend\s+to|not\s+(?:be\s+)?less\s+than|not\s+exceed(?:ing)?|up\s+to|"
                   rf"minimum\s+of|maximum\s+of)\s+(?:imprisonment\s+for\s+)?(?:a\s+term\s+of\s+)?"
                   rf"(?P<num>{_NUM})\s+(?P<unit>years?|months?|days?)", re.I)
_FINE_AMOUNT = re.compile(rf"(?:(?P<kw>may\s+extend\s+to|not\s+(?:be\s+)?less\s+than|not\s+exceed(?:ing)?|up\s+to)\s+)?"
                          rf"(?:(?:₹|rs\.?|rupees)\s*(?P<num1>\d[\d,]*)|(?P<num2>{_NUM})\s+(?P<unit>rupees|lakh|lakhs|crore|crores))",
                          re.I)


_PREDICATE = re.compile(r"\b(?:an?\s+|any\s+|the\s+)?offences?\s+(?:which\s+is\s+|that\s+is\s+)?punishable\s+"
                        r"(?:under\s+[^,;]{0,60}?\s+)?with\s+[^;:]{0,140}?(?=,|;|:|\bshall\b|\bif\b|$)")


def words_to_number(text: str) -> float | None:
    """'two thousand and five hundred' -> 2500; '1,000' -> 1000; 'ten lakh' -> 1_000_000."""
    t = text.strip().lower().replace(",", "")
    if not t:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    total, current, seen = 0.0, 0.0, False
    for w in re.split(r"[\s-]+", t):
        if w == "and" or not w:
            continue
        if w in _UNITS:
            current += _UNITS[w]
            seen = True
        elif w in ("hundred",):
            current = (current or 1) * 100
            seen = True
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0.0
            seen = True
        elif re.fullmatch(r"\d+(?:\.\d+)?", w):
            current += float(w)
            seen = True
        else:
            return None
    return total + current if seen else None


def _years(num: str, unit: str) -> float | None:
    n = words_to_number(num)
    if n is None:
        return None
    unit = unit.lower()
    return n if unit.startswith("year") else n / 12 if unit.startswith("month") else n / 365


@dataclass
class Punishment:
    death: bool = False
    life: bool = False
    max_years: float | None = None
    min_years: float | None = None
    rigorous: bool = False
    simple: bool = False
    fine: bool = False
    fine_max: float | None = None
    fine_min: float | None = None
    community_service: bool = False
    kinds: set[str] = field(default_factory=set)

    @property
    def empty(self) -> bool:
        return not self.kinds

    def describe(self) -> str:
        if self.empty:
            return "no punishment in the text"
        bits = []
        if self.death:
            bits.append("death")
        if self.life:
            bits.append("life imprisonment")
        if self.max_years is not None or self.min_years is not None:
            term = "imprisonment"
            if self.min_years is not None:
                term += f" not less than {_fmt_years(self.min_years)}"
            if self.max_years is not None:
                term += f" up to {_fmt_years(self.max_years)}"
            bits.append(term)
        if self.fine:
            f = "fine"
            if self.fine_min:
                f += f" not less than ₹{self.fine_min:,.0f}"
            if self.fine_max:
                f += f" up to ₹{self.fine_max:,.0f}"
            bits.append(f)
        if self.community_service:
            bits.append("community service")
        return ", ".join(bits)


def _fmt_years(y: float) -> str:
    if y >= 1 and abs(y - round(y)) < 1e-6:
        return f"{int(round(y))} year{'s' if round(y) != 1 else ''}"
    if y * 12 >= 1 and abs(y * 12 - round(y * 12)) < 1e-6:
        m = int(round(y * 12))
        return f"{m} month{'s' if m != 1 else ''}"
    return f"{round(y * 365)} days"


def _max(a, b):
    return b if a is None else a if b is None else max(a, b)


def extract(text: str) -> Punishment:
    """Punishment prescribed by a provision text (see module docstring)."""
    p = Punishment()
    low = " ".join((text or "").lower().split())
    if not re.search(r"punish|imprisonment|fine|liable", low):
        return p
    # "abets an offence punishable with death or imprisonment for life ... shall be punished with
    # ... seven years": the first phrase describes ANOTHER offence, not this section's punishment
    low = _PREDICATE.sub(" ", low)
    if re.search(r"punish\w*\s+with\s+death|with\s+death\b|\bdeath\s+(?:sentence|penalty)|\bdeath,?\s+or\b"
                 r"|\bor\s+(?:with\s+)?death\b|^\s*death\b", low):
        p.death = True
        p.kinds.add("death")
    if re.search(r"imprisonment\s+for\s+life|life\s+imprisonment|remainder\s+of\s+(?:that\s+)?person'?s?\s+natural\s+life", low):
        p.life = True
        p.kinds.add("life")
    if "rigorous imprisonment" in low:
        p.rigorous = True
    if "simple imprisonment" in low:
        p.simple = True
    for m in _TERM.finditer(low):
        y = _years(m.group("num"), m.group("unit"))
        if y is None:
            continue
        if "less" in m.group("kw"):
            p.min_years = _max(p.min_years, y)
        else:
            p.max_years = _max(p.max_years, y)
        p.kinds.add("imprisonment")
    if "imprisonment" in low:
        p.kinds.add("imprisonment")
    if re.search(r"\bfine\b", low):
        p.fine = True
        p.kinds.add("fine")
        for m in _FINE_AMOUNT.finditer(low):
            num = m.group("num1") or m.group("num2")
            n = words_to_number(num)
            if n is None:
                continue
            unit = (m.group("unit") or "rupees").lower()
            n *= _SCALES.get(unit, 1)
            # only amounts in a fine context: the word "fine" within the 60 characters before
            if "fine" not in low[max(0, m.start() - 60):m.start()]:
                continue
            kw = (m.group("kw") or "").lower()
            if "less" in kw:
                p.fine_min = _max(p.fine_min, n)
            else:
                p.fine_max = _max(p.fine_max, n)
    if "community service" in low:
        p.community_service = True
        p.kinds.add("community_service")
    return p


@dataclass
class PunishmentDelta:
    changed: list[str] = field(default_factory=list)        # field names that differ
    harsher: list[str] = field(default_factory=list)
    lighter: list[str] = field(default_factory=list)
    comparable: bool = True                                 # both texts state a punishment

    @property
    def any(self) -> bool:
        return bool(self.changed)

    @property
    def direction(self) -> str:
        if not self.changed:
            return "same"
        if self.harsher and not self.lighter:
            return "harsher"
        if self.lighter and not self.harsher:
            return "lighter"
        return "mixed"


def compare(old: Punishment, new: Punishment) -> PunishmentDelta:
    """Field-by-field change from the IPC punishment to the BNS punishment."""
    d = PunishmentDelta()
    if old.empty or new.empty:            # a definition, or a text that lost its punishment clause
        d.comparable = False
        return d
    for name in ("death", "life", "community_service", "fine"):
        a, b = getattr(old, name), getattr(new, name)
        if a != b:
            d.changed.append(name)
            # adding community service is a lighter option; adding death/life/fine is harsher
            (d.lighter if (name == "community_service") == b else d.harsher).append(name)
    for name in ("max_years", "min_years", "fine_max", "fine_min"):
        a, b = getattr(old, name), getattr(new, name)
        if a is None and b is None:
            continue
        if a is None or b is None or abs(a - b) > 1e-6:
            d.changed.append(name)
            if a is None:                 # a new minimum is harsher; a new cap on its own is unclear
                if name in ("min_years", "fine_min"):
                    d.harsher.append(name)
            elif b is None:
                if name in ("min_years", "fine_min"):
                    d.lighter.append(name)
            elif b > a:
                d.harsher.append(name)
            else:
                d.lighter.append(name)
    return d


__all__ = ["Punishment", "PunishmentDelta", "extract", "compare", "words_to_number"]
