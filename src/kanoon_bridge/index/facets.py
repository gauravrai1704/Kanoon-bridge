"""Parametric (facet) index for spatio-temporal filters.  [owner: A]

Facets: doc type, court level, states, decision date, code in force, sections cited.
They answer filters such as `state:delhi`, `date:2025-03-01`, `code:bns` quickly, as
doc-id sets that ranking then intersects with.

The where-and-when design follows the spatio-temporal querying line of work of
Dr. Sonia Khetarpaul (see proposal, Related work).

    fx = FacetIndex.build(docs)
    fx.filter(doc_type="statute", code="bns")            # set of doc ids
    fx.filter(states=["delhi"], date_from=date(2015,1,1))
    fx.meta("d1")                                        # DocMeta for authority weighting
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from kanoon_bridge.schema import Document


@dataclass
class DocMeta:
    doc_type: str
    court: str
    court_name: str
    states: list[str]
    decision_date: date | None
    code: str
    statutes_cited: list[str]


@dataclass
class FacetIndex:
    by_type: dict[str, set[str]] = field(default_factory=dict)
    by_court: dict[str, set[str]] = field(default_factory=dict)
    by_state: dict[str, set[str]] = field(default_factory=dict)     # "*" = binds everywhere (Supreme Court)
    by_code: dict[str, set[str]] = field(default_factory=dict)
    by_section: dict[str, set[str]] = field(default_factory=dict)   # statute section ref -> precedents citing it
    dates: list[tuple[date, str]] = field(default_factory=list)     # sorted (decision_date, doc_id)
    metas: dict[str, DocMeta] = field(default_factory=dict)

    @classmethod
    def build(cls, docs: Iterable[Document]) -> "FacetIndex":
        """Fill every facet dict from Document fields; `dates` sorted for bisect range search."""
        fx = cls()
        for d in docs:
            m = DocMeta(doc_type=d.doc_type.value, court=d.court.value, court_name=d.court_name,
                        states=list(d.states), decision_date=d.decision_date, code=d.code.value,
                        statutes_cited=list(d.statutes_cited))
            fx.metas[d.doc_id] = m
            fx.by_type.setdefault(m.doc_type, set()).add(d.doc_id)
            fx.by_court.setdefault(m.court, set()).add(d.doc_id)
            fx.by_code.setdefault(m.code, set()).add(d.doc_id)
            for s in m.states:
                fx.by_state.setdefault(s, set()).add(d.doc_id)
            for ref in m.statutes_cited:
                fx.by_section.setdefault(ref, set()).add(d.doc_id)
            if m.decision_date is not None:
                fx.dates.append((m.decision_date, d.doc_id))
        fx.dates.sort()
        return fx

    def date_range(self, date_from: date | None = None, date_to: date | None = None) -> set[str]:
        """Docs decided in [date_from, date_to] (either end open), by bisect on the sorted dates."""
        lo = 0 if date_from is None else bisect.bisect_left(self.dates, (_as_date(date_from), ""))
        hi = len(self.dates) if date_to is None else bisect.bisect_right(self.dates, (_as_date(date_to), "\uffff"))
        return {doc_id for _, doc_id in self.dates[lo:hi]}

    def filter(
        self,
        doc_type: str | None = None,
        court: str | None = None,
        states: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        code: str | None = None,
        cites_any: list[str] | None = None,
    ) -> set[str]:
        """Intersection of all given facets; None means 'no constraint' (all docs).

        A states filter also admits docs in by_state["*"] (Supreme Court binds everywhere).
        The sets are intersected smallest first, and the date range is computed only if needed.
        """
        sets: list[set[str]] = []
        if doc_type:
            sets.append(self.by_type.get(doc_type, set()))
        if court:
            sets.append(self.by_court.get(court, set()))
        if code:
            sets.append(self.by_code.get(code, set()))
        if states:
            sets.append(set().union(*(self.by_state.get(s, set()) for s in states), self.by_state.get("*", set())))
        if cites_any:
            sets.append(set().union(*(self.by_section.get(r, set()) for r in cites_any)))
        if date_from is not None or date_to is not None:
            sets.append(self.date_range(date_from, date_to))
        if not sets:
            return set(self.metas)
        sets.sort(key=len)
        out = set(sets[0])
        for s in sets[1:]:
            out &= s
            if not out:
                break
        return out

    def meta(self, doc_id: str) -> DocMeta:
        return self.metas[doc_id]


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))
