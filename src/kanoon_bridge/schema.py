"""Shared data classes. Every module passes these objects, nothing ad hoc.  [all — working]

Agree on this file in hour 1. If you need a new field, add it here (with a default)
and tell the team, rather than stuffing it into `meta`.

Conventions
-----------
* Section references are strings "CODE:SECTION", lower-case, e.g. "ipc:302", "bns:103(1)",
  "ipc:498a". A bare number whose code is unknown is "?:302". Build them with `section_ref()`.
* Canonical offence IDs (from data/crosswalk/offence_ids.csv) look like "off:murder".
* Zones are the strings in `ZONES`.
* States are lower-case slugs, e.g. "delhi", "maharashtra", "uttar-pradesh".
* Dates are `datetime.date`; in JSON they are ISO strings "YYYY-MM-DD".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------- enums


class DocType(str, Enum):
    STATUTE = "statute"            # one section of an Act (IPC, BNS, ...)
    PRECEDENT = "precedent"        # a candidate judgment that can be cited
    QUERY_CASE = "query_case"      # an IL-PCSR query judgment (citations masked)


class Code(str, Enum):
    IPC = "ipc"                    # Indian Penal Code, 1860 (in force until 30 Jun 2024)
    BNS = "bns"                    # Bharatiya Nyaya Sanhita, 2023 (from 1 Jul 2024)
    CRPC = "crpc"
    BNSS = "bnss"
    OTHER = "other"                # any other Act
    UNKNOWN = "?"


class Court(str, Enum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    OTHER = "other"
    UNKNOWN = "unknown"


ZONES: tuple[str, ...] = ("facts", "arguments", "ratio", "decision", "statute", "other")

# --------------------------------------------------------------------------- helpers


def section_ref(code: Code | str, section: str) -> str:
    """Build the canonical section reference string, e.g. section_ref(Code.IPC, '302') -> 'ipc:302'."""
    code_str = code.value if isinstance(code, Code) else str(code).lower()
    return f"{code_str}:{section.strip().lower()}"


def parse_section_ref(ref: str) -> tuple[str, str]:
    """'ipc:302' -> ('ipc', '302')."""
    code, _, section = ref.partition(":")
    return code, section


def _to_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


# --------------------------------------------------------------------------- documents


@dataclass
class Paragraph:
    """One paragraph of a document, labelled with its zone."""

    text: str
    zone: str = "other"
    para_id: int = 0


@dataclass
class Document:
    """A statute section, a precedent judgment, or an IL-PCSR query judgment."""

    doc_id: str
    doc_type: DocType
    title: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)

    # --- metadata used by the facets (filled by ingest/metadata.py) -------------------
    court: Court = Court.UNKNOWN
    court_name: str = ""                       # e.g. "Delhi High Court"
    states: list[str] = field(default_factory=list)   # states the court's rulings bind
    decision_date: date | None = None
    code: Code = Code.UNKNOWN                  # code in force for this doc (statutes: its Act)

    # --- citations -------------------------------------------------------------------
    statutes_cited: list[str] = field(default_factory=list)    # section refs, e.g. "ipc:302"
    precedents_cited: list[str] = field(default_factory=list)  # doc_ids

    # --- statute-only fields -----------------------------------------------------------
    section: str | None = None                 # e.g. "103" for BNS 103
    offence_ids: list[str] = field(default_factory=list)       # e.g. ["off:murder"]

    # --- bookkeeping -------------------------------------------------------------------
    split: str | None = None                   # "train" | "val" | "test" for IL-PCSR queries
    lang: str = "en"
    meta: dict[str, Any] = field(default_factory=dict)

    # Convenience ---------------------------------------------------------------------
    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    def zone_text(self, zone: str) -> str:
        return "\n\n".join(p.text for p in self.paragraphs if p.zone == zone)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["doc_type"] = self.doc_type.value
        d["court"] = self.court.value
        d["code"] = self.code.value
        d["decision_date"] = self.decision_date.isoformat() if self.decision_date else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        d = dict(d)
        d["doc_type"] = DocType(d["doc_type"])
        d["court"] = Court(d.get("court", Court.UNKNOWN.value))
        d["code"] = Code(d.get("code", Code.UNKNOWN.value))
        d["decision_date"] = _to_date(d.get("decision_date"))
        d["paragraphs"] = [Paragraph(**p) for p in d.get("paragraphs", [])]
        return cls(**d)


# --------------------------------------------------------------------------- queries


@dataclass
class Query:
    """What the user typed, plus the facets that steer ranking."""

    text: str
    query_id: str | None = None
    lang: str | None = None                    # "en" | "hi" (Devanagari) | "hinglish"; None = detect
    state: str | None = None                   # user's state, for jurisdiction-aware authority
    incident_date: date | None = None          # decides IPC vs BNS
    filters: dict[str, Any] = field(default_factory=dict)  # e.g. {"code": "bns", "court": "supreme_court"}

    def __post_init__(self) -> None:
        self.incident_date = _to_date(self.incident_date)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Query":
        keys = {"text", "query_id", "lang", "state", "incident_date", "filters"}
        q = {k: v for k, v in d.items() if k in keys}
        if "id" in d and "query_id" not in q:
            q["query_id"] = d["id"]
        return cls(**q)


@dataclass
class AnalyzedQuery:
    """Output of query/analyzer.py: every rewritten form is kept so the demo can show it."""

    query: Query
    detected_lang: str = "en"
    transliterated: str = ""                   # Roman/Devanagari normalised text
    tokens: list[str] = field(default_factory=list)          # after tokenise + stem
    expanded_terms: dict[str, float] = field(default_factory=dict)  # term -> weight (lexicon, phonetic, crosswalk)
    sections: list[str] = field(default_factory=list)        # section refs found in the query
    offence_ids: list[str] = field(default_factory=list)     # canonical offences after version normalisation
    code_in_force: Code = Code.UNKNOWN         # from incident_date
    trace: list[tuple[str, str]] = field(default_factory=list)  # (step name, output) for --debug

    def weighted_terms(self) -> dict[str, float]:
        """All query terms with weights: tokens weight 1.0 plus expansions."""
        terms: dict[str, float] = {}
        for t in self.tokens:
            terms[t] = terms.get(t, 0.0) + 1.0
        for t, w in self.expanded_terms.items():
            terms[t] = max(terms.get(t, 0.0), w)
        for o in self.offence_ids:
            terms[o] = max(terms.get(o, 0.0), 1.0)
        return terms


# --------------------------------------------------------------------------- index + results


@dataclass
class Posting:
    """One document's entry in a term's postings list."""

    doc_id: str
    positions: list[int] = field(default_factory=list)
    zone: str | None = None                    # None in the whole-document index

    @property
    def tf(self) -> int:
        return len(self.positions)


@dataclass
class ScoredDoc:
    """A ranked result. `components` keeps the score breakdown for --debug and the report."""

    doc_id: str
    score: float
    doc_type: DocType = DocType.PRECEDENT
    components: dict[str, float] = field(default_factory=dict)  # e.g. {"bm25f": 7.1, "authority": 0.3}
    rank: int = 0


@dataclass
class SearchResult:
    query: AnalyzedQuery
    statutes: list[ScoredDoc] = field(default_factory=list)
    precedents: list[ScoredDoc] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- JSONL I/O


def write_documents(docs: Iterable[Document], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_documents(path: str | Path) -> Iterator[Document]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Document.from_dict(json.loads(line))


def read_queries(path: str | Path) -> list[Query]:
    """Read a hand-built query set (data/queries/*.jsonl)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Query.from_dict(json.loads(line)))
    return out
