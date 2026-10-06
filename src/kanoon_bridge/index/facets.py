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
        """TODO(A): fill every dict above from Document fields; sort `dates` for range search (bisect)."""
        raise NotImplementedError("TODO(A): build facet index")

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
        """Intersection of all given facets; None means 'no constraint'.

        TODO(A): intersect smallest set first; date range via bisect on `dates`;
        a states filter also admits docs in by_state["*"].
        """
        raise NotImplementedError("TODO(A): facet filtering")

    def meta(self, doc_id: str) -> DocMeta:
        return self.metas[doc_id]
