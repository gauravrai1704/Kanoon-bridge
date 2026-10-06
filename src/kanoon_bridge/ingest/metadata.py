"""Court, states and dates for every judgment — the spatio-temporal facets.  [owner: A]

    court_name     "Delhi High Court"         from the judgment header / IL-PCSR metadata
    court          Court.HIGH_COURT
    states         ["delhi"]                   via data/lexicons/court_to_states.csv
    decision_date  date(2016, 3, 4)            from the header
    code           Code.IPC                    code in force on the decision date

Jurisdiction matters because a High Court binds only within its territory; the Supreme
Court binds everywhere (states = ["*"]). See rank/authority.py.
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


@dataclass
class CourtTable:
    """court name pattern -> (court level, states). Loaded from court_to_states.csv (working)."""

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
        """First matching court in `text`; (Court.UNKNOWN, [], '') if none (working)."""
        for pattern, level, states, name in self.rows:
            if pattern.search(text):
                return level, states, name
        return Court.UNKNOWN, [], ""


def find_date(header: str) -> date | None:
    """Decision date from a header like 'DATED: 04.03.2016' or '4 March, 2016'.

    TODO(A): cover the formats in 20 real headers; return the latest plausible date
    (1950-2026) when several appear.
    """
    raise NotImplementedError("TODO(A): parse judgment dates")


def code_in_force(d: date | None, bns_from: date = date(2024, 7, 1)) -> Code:
    """IPC before 1 July 2024, BNS from then on (working)."""
    if d is None:
        return Code.UNKNOWN
    return Code.BNS if d >= bns_from else Code.IPC


def enrich(doc: Document, table: CourtTable, header_chars: int = 1500) -> Document:
    """Fill court, court_name, states, decision_date, code from the start of the text.

    TODO(A): use table.lookup and find_date on doc.text[:header_chars]; prefer IL-PCSR
    metadata fields when present. Count docs with missing court/date and print it in
    scripts/00_check_access.py (risk table in the proposal).
    """
    raise NotImplementedError("TODO(A): fill spatio-temporal metadata")
