"""tf-idf vector space model with SMART lnc.ltc weighting.  [owner: Gaurav — working]

The classic lecture baseline, reported next to BM25 in E1.

    document: l (1 + log10 tf), n (no idf), c (cosine normalisation)
    query:    l (1 + log10 tf), t (idf = log10 N/df), c (cosine normalisation)
    score(q, d) = sum over shared terms of w_q * w_d

Efficient cosine scoring: walk each query term's postings once, accumulate into a dict of doc
scores, divide by the precomputed document vector lengths (prepare()).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kanoon_bridge.index.positional import PositionalIndex


def _log_tf(tf: float) -> float:
    """1 + log10 tf for counts >= 1; fractional expansion weights (< 1) are used as they are."""
    return 1.0 + math.log10(tf) if tf >= 1 else max(tf, 0.0)


@dataclass
class TfidfScorer:
    index: PositionalIndex
    doc_norms: dict[str, float] = field(default_factory=dict)   # precomputed |d| for lnc
    max_query_terms: int = 300                                   # same cap as BM25F for long queries

    def prepare(self) -> "TfidfScorer":
        """Precompute each document's lnc vector length: sqrt(sum over terms (1 + log10 tf)^2)."""
        sq: dict[str, float] = {}
        for postings in self.index.postings.values():
            for doc, positions in postings.items():
                w = _log_tf(len(positions))
                sq[doc] = sq.get(doc, 0.0) + w * w
        self.doc_norms = {d: math.sqrt(v) for d, v in sq.items()}
        return self

    def score(self, terms: dict[str, float], candidates: set[str] | None = None) -> dict[str, float]:
        """lnc.ltc cosine for every doc sharing a term with the query (restricted to candidates)."""
        if not self.doc_norms:
            self.prepare()
        q = {t: _log_tf(w) * self.index.idf(t) for t, w in terms.items()}
        q = dict(sorted(((t, w) for t, w in q.items() if w > 0), key=lambda kv: -kv[1])[: self.max_query_terms])
        q_norm = math.sqrt(sum(w * w for w in q.values())) or 1.0
        acc: dict[str, float] = {}
        for term, wq in q.items():
            for doc, positions in self.index.postings.get(term, {}).items():
                if candidates is not None and doc not in candidates:
                    continue
                acc[doc] = acc.get(doc, 0.0) + (wq / q_norm) * _log_tf(len(positions))
        return {d: s / (self.doc_norms.get(d) or 1.0) for d, s in acc.items()}
