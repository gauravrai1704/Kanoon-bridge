"""tf-idf vector space model with SMART lnc.ltc weighting.  [owner: D]

The classic lecture baseline, reported next to BM25 in E1.

    document: l (1 + log10 tf), n (no idf), c (cosine normalisation)
    query:    l (1 + log10 tf), t (idf = log10 N/df), c (cosine normalisation)
    score(q, d) = sum over shared terms of w_q * w_d

Use efficient cosine scoring: walk the postings of each query term once, accumulate into
a dict of doc scores, and divide by precomputed document vector lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.index.positional import PositionalIndex


@dataclass
class TfidfScorer:
    index: PositionalIndex
    doc_norms: dict[str, float] = field(default_factory=dict)   # precomputed |d| for lnc

    def prepare(self) -> "TfidfScorer":
        """Precompute each document's lnc vector length.

        TODO(D): for every term and posting, add (1 + log10 tf)^2 to the doc's sum; sqrt at the end.
        """
        raise NotImplementedError("TODO(D): precompute lnc document norms")

    def score(self, terms: dict[str, float], candidates: set[str] | None = None) -> dict[str, float]:
        """lnc.ltc cosine for every doc sharing a term with the query (restricted to candidates).

        `terms` maps query term -> raw query weight (count or expansion weight from AnalyzedQuery).
        TODO(D): ltc query weights, accumulate over postings, divide by doc_norms.
        """
        raise NotImplementedError("TODO(D): lnc.ltc cosine scoring")
