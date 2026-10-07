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

from bisect import bisect_left, bisect_right
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
    by_state: dict[str, set[str]] = field(default_factory=dict)
    by_code: dict[str, set[str]] = field(default_factory=dict)
    by_section: dict[str, set[str]] = field(default_factory=dict)
    dates: list[tuple[date, str]] = field(default_factory=list)
    metas: dict[str, DocMeta] = field(default_factory=dict)

    @classmethod
    def build(cls, docs: Iterable[Document]) -> "FacetIndex":
        """Build all facet indexes from the supplied documents."""
        index = cls()

        def add_to_index(mapping: dict[str, set[str]], key: str, doc_id: str) -> None:
            mapping.setdefault(key, set()).add(doc_id)

        for doc in docs:
            doc_id = doc.doc_id

            # Normalise scalar facet values.
            doc_type = str(doc.doc_type).lower()
            court = str(doc.court).lower()
            court_name = str(doc.court_name)

            # States may be stored as a list/tuple/set.
            states = [
                str(state).lower()
                for state in (doc.states or [])
            ]

            code = str(doc.code).lower()

            # Sections/statutes cited may be stored as strings.
            statutes_cited = [
                str(section).lower()
                for section in (doc.statutes_cited or [])
            ]

            decision_date = doc.decision_date

            # Store metadata.
            index.metas[doc_id] = DocMeta(
                doc_type=doc_type,
                court=court,
                court_name=court_name,
                states=states,
                decision_date=decision_date,
                code=code,
                statutes_cited=statutes_cited,
            )

            # Type.
            add_to_index(index.by_type, doc_type, doc_id)

            # Court.
            add_to_index(index.by_court, court, doc_id)

            # States.
            for state in states:
                add_to_index(index.by_state, state, doc_id)

            # Code.
            add_to_index(index.by_code, code, doc_id)

            # Sections/statutes cited.
            for section in statutes_cited:
                add_to_index(index.by_section, section, doc_id)

            # Decision date.
            if decision_date is not None:
                index.dates.append((decision_date, doc_id))

        # Sort by date so bisect can be used for range queries.
        index.dates.sort(key=lambda item: item[0])

        return index

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

        Uses the smallest candidate set first. Date ranges are handled with
        binary search over the sorted `dates` list. A state filter also
        includes documents in by_state["*"].
        """
        candidates: list[set[str]] = []

        # Document type.
        if doc_type is not None:
            candidates.append(
                self.by_type.get(doc_type.lower(), set())
            )

        # Court.
        if court is not None:
            candidates.append(
                self.by_court.get(court.lower(), set())
            )

        # States.
        if states:
            state_sets = []

            for state in states:
                state = state.lower()

                matching = set(self.by_state.get(state, set()))

                # "*" means the document applies everywhere.
                matching.update(self.by_state.get("*", set()))

                state_sets.append(matching)

            if state_sets:
                # A document must satisfy all requested state filters.
                candidates.append(
                    set.intersection(*state_sets)
                )

        # Code.
        if code is not None:
            candidates.append(
                self.by_code.get(code.lower(), set())
            )

        # Sections/statutes cited.
        if cites_any:
            cited_sets = [
                self.by_section.get(section.lower(), set())
                for section in cites_any
            ]

            # "cites_any" means a document can match any one of them.
            if cited_sets:
                candidates.append(set.union(*cited_sets))

        # Date range.
        if date_from is not None or date_to is not None:
            dates_only = [item[0] for item in self.dates]

            if date_from is None:
                left = 0
            else:
                left = bisect_left(dates_only, date_from)

            if date_to is None:
                right = len(self.dates)
            else:
                # bisect_right makes date_to inclusive.
                right = bisect_right(dates_only, date_to)

            date_candidates = {
                doc_id
                for _, doc_id in self.dates[left:right]
            }

            candidates.append(date_candidates)

        # No filters means return every indexed document.
        if not candidates:
            return set(self.metas)

        # Start with the smallest candidate set to reduce intersection work.
        candidates.sort(key=len)

        result = set(candidates[0])

        for candidate in candidates[1:]:
            result.intersection_update(candidate)

            if not result:
                break

        return result

    def meta(self, doc_id: str) -> DocMeta:
        return self.metas[doc_id]