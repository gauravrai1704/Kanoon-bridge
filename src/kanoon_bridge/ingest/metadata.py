"""Court, states and dates for every judgment — the spatio-temporal facets.  [owner: Gaurav — working]

    court_name     "Delhi High Court"         IL-PCSR `jurisdiction` field, else the judgment header
    court          Court.HIGH_COURT
    states         ["delhi"]                   via data/lexicons/court_to_states.csv
    decision_date  date(2016, 3, 4)            IL-PCSR `date` field, else parsed from the header
    code           Code.IPC                    code in force on the decision date

Jurisdiction matters because a High Court binds only within its territory; the Supreme Court
binds everywhere (states = ["*"]). See rank/authority.py.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date

from kanoon_bridge.config import Config, load_config, project_path
from kanoon_bridge.schema import Code, Court, Document

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9

_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_LONG_DATE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})\b", re.I)
_US_DATE = re.compile(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


@dataclass
class CourtTable:
    """court name pattern -> (court level, states). Loaded from court_to_states.csv."""

    rows: list[tuple[re.Pattern, Court, list[str], str]] = field(default_factory=list)

    @classmethod
    def load(cls, cfg: Config | None = None) -> "CourtTable":
        cfg = cfg or load_config()
        table = cls()
        with open(project_path(cfg.paths.court_to_states), encoding="utf-8") as f:
            for row in csv.DictReader(line for line in f if not line.startswith("#")):
                pattern = re.compile(row["name_pattern"], re.I)
                states = [s.strip().lower() for s in row["states"].split(";") if s.strip()]
                table.rows.append((pattern, Court(row["level"].strip()), states, row["court_name"].strip()))
        return table

    def lookup(self, text: str) -> tuple[Court, list[str], str]:
        """First matching court in `text`; (Court.UNKNOWN, [], '') if none."""
        for pattern, level, states, name in self.rows:
            if pattern.search(text):
                return level, states, name
        if re.search(r"high\s+court", text, re.I):
            return Court.HIGH_COURT, [], _guess_name(text)
        return Court.UNKNOWN, [], ""


def _guess_name(text: str) -> str:
    m = re.search(r"((?:[A-Z][a-z]+\s){0,3}High Court(?:\s+of\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)?)", text)
    return m.group(1).strip() if m else "High Court"


def _valid(y: int, m: int, d: int) -> date | None:
    try:
        out = date(y, m, d)
    except ValueError:
        return None
    return out if date(1950, 1, 1) <= out <= date(2026, 12, 31) else None


def find_dates(text: str) -> list[date]:
    """Every plausible date (1950-2026) in the text, in the order found."""
    found: list[tuple[int, date]] = []
    for m in _ISO_DATE.finditer(text):
        d = _valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append((m.start(), d))
    for m in _NUMERIC_DATE.finditer(text):
        d = _valid(int(m.group(3)), int(m.group(2)), int(m.group(1)))   # Indian order: dd.mm.yyyy
        if d:
            found.append((m.start(), d))
    for m in _LONG_DATE.finditer(text):
        mon = _MONTHS.get(m.group(2).lower())
        d = _valid(int(m.group(3)), mon, int(m.group(1))) if mon else None
        if d:
            found.append((m.start(), d))
    for m in _US_DATE.finditer(text):
        mon = _MONTHS.get(m.group(1).lower())
        d = _valid(int(m.group(3)), mon, int(m.group(2))) if mon else None
        if d:
            found.append((m.start(), d))
    return [d for _, d in sorted(found)]


def find_date(header: str) -> date | None:
    """Decision date from a judgment header.

    Prefers a date right after 'DATED'/'DATE OF JUDGMENT'/'decided on'; otherwise the latest
    plausible date in the header (earlier dates there are usually FIR or lower-court dates).
    """
    m = re.search(r"(?:dated?|date\s+of\s+(?:judgment|decision|order)|decided\s+on|pronounced\s+on)\s*[:\-]?\s*(.{0,40})",
                  header, re.I)
    if m:
        near = find_dates(m.group(1))
        if near:
            return near[0]
    all_dates = find_dates(header)
    return max(all_dates) if all_dates else None


def code_in_force(d: date | None, bns_from: date = date(2024, 7, 1)) -> Code:
    """IPC before 1 July 2024, BNS from then on."""
    if d is None:
        return Code.UNKNOWN
    return Code.BNS if d >= bns_from else Code.IPC


def enrich(doc: Document, table: CourtTable, header_chars: int = 1500, bns_from: date = date(2024, 7, 1)) -> Document:
    """Fill court, court_name, states, decision_date and code (in place, also returned).

    IL-PCSR metadata (meta['jurisdiction'], decision_date) wins; the judgment header is the fallback.
    """
    header = doc.text[:header_chars]
    source = doc.meta.get("jurisdiction") or doc.title + " " + header
    level, states, name = table.lookup(source)
    if level == Court.UNKNOWN and doc.meta.get("jurisdiction"):
        level, states, name = table.lookup(doc.title + " " + header)
    doc.court, doc.states, doc.court_name = level, states, name or doc.meta.get("jurisdiction", "")
    if doc.decision_date is None:
        doc.decision_date = find_date(header)
    doc.code = code_in_force(doc.decision_date, bns_from)
    doc.meta["court_source"] = "metadata" if doc.meta.get("jurisdiction") else "header"
    return doc
