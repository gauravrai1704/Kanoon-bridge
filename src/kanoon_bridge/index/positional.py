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
        """Add one document's tokens with their positions (0-based, in token order).

        Adding the same doc_id twice raises ValueError (catches double-ingest bugs).
        """
        if doc_id in self.doc_len:
            raise ValueError(f"document added twice: {doc_id}")
        postings = self.postings
        for pos, term in enumerate(tokens):
            plist = postings[term].get(doc_id)
            if plist is None:
                postings[term][doc_id] = [pos]
            else:
                plist.append(pos)
        self.doc_len[doc_id] = len(tokens)

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
        """Documents containing `terms` consecutively (positional intersection).

        Candidate docs = intersection of the terms' postings, rarest term first (so the
        running set is as small as possible). In each candidate, a start position p matches
        when term i occurs at p + i for every i: the start set is narrowed term by term.
        """
        if not terms:
            return set()
        if any(t not in self.postings for t in terms):
            return set()
        by_df = sorted(set(terms), key=self.df)
        docs = set(self.postings[by_df[0]])
        for t in by_df[1:]:
            docs &= self.postings[t].keys()
            if not docs:
                return set()
        out = set()
        for d in docs:
            starts = set(self.postings[terms[0]][d])
            for offset, t in enumerate(terms[1:], start=1):
                starts &= {p - offset for p in self.postings[t][d]}
                if not starts:
                    break
            if starts:
                out.add(d)
        return out
