"""Positional inverted index.  [owner: A]

    term -> {doc_id: [positions]}

Supports document frequency, idf, postings lists and phrase queries. The zone index
(index/zones.py) holds one of these per zone; the whole-document index is used for
Boolean/phrase/proximity queries.

Build it from tokens that came out of text.pipeline.analyze_text, never raw text.

    idx = PositionalIndex()
    idx.add("d1", ["murder", "sec:ipc:302", "off:murder"])
    idx.df("off:murder")            # 1
    idx.phrase(["criminal", "breach", "trust"])   # set of doc ids
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from kanoon_bridge.schema import Posting


@dataclass
class PositionalIndex:
    postings: dict[str, dict[str, list[int]]] = field(default_factory=lambda: defaultdict(dict))
    doc_len: dict[str, int] = field(default_factory=dict)       # tokens per document

    # ------------------------------------------------------------------ build
    def add(self, doc_id: str, tokens: list[str]) -> None:
        """Add one document's tokens with their positions.

        TODO(A): record positions for every token; set doc_len[doc_id]. Adding the same
        doc_id twice should raise ValueError (catches double-ingest bugs).
        """
        raise NotImplementedError("TODO(A): add a document to the positional index")

    # ------------------------------------------------------------------ statistics (implement after add)
    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avg_doc_len(self) -> float:
        return sum(self.doc_len.values()) / self.n_docs if self.n_docs else 0.0

    def vocabulary(self) -> list[str]:
        return list(self.postings)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        """log10(N / df); 0.0 for unseen terms (lecture definition)."""
        df = self.df(term)
        return math.log10(self.n_docs / df) if df else 0.0

    def tf(self, term: str, doc_id: str) -> int:
        return len(self.postings.get(term, {}).get(doc_id, []))

    def postings_for(self, term: str) -> list[Posting]:
        """Postings sorted by doc_id (needed for linear-merge intersection)."""
        return [Posting(d, pos) for d, pos in sorted(self.postings.get(term, {}).items())]

    def docs_with(self, term: str) -> set[str]:
        return set(self.postings.get(term, {}))

    # ------------------------------------------------------------------ phrase queries
    def phrase(self, terms: list[str]) -> set[str]:
        """Documents containing `terms` consecutively.

        TODO(A): positional intersection — start from the rarest term's docs, check that
        term i appears at position p+i for some p. Test with tests/test_positional.py.
        """
        raise NotImplementedError("TODO(A): phrase query over positions")
