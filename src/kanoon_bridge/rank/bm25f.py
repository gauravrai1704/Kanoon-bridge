"""BM25F: BM25 with per-zone weights.  [owner: Gaurav — main lexical scorer, working]

For a query term t and document d with zones z:

    tf~(t, d) = sum_z  w_z * tf(t, d, z) / (1 - b + b * len(d, z) / avglen(z))
    score(q, d) = sum_t  qw(t) * idf(t) * tf~ * (k1 + 1) / (tf~ + k1)
    idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )            (never negative)

Zone weights, k1 and b come from configs/default.yaml (zones.*, bm25.*).
With use_zones=False the whole-document index is used with plain BM25 length normalisation —
the first rung of the ablation ladder.

Efficiency: term-at-a-time accumulation over postings (one pass per query term and zone), with
per-zone length norms cached once. Very long queries (whole judgments, as in IL-PCSR E1) keep
their `max_query_terms` highest idf x weight terms.

IL-PCSR found BM25 to be the strongest single method for precedent retrieval, so this file
matters most for E1.
"""

from __future__ import annotations

from kanoon_bridge.index.positional import count

import math
from dataclasses import dataclass, field

from kanoon_bridge.index.zones import ZoneIndex


@dataclass
class BM25F:
    zidx: ZoneIndex
    zone_weights: dict[str, float] = field(default_factory=dict)
    k1: float = 1.6
    b: float = 0.7
    use_zones: bool = True
    max_query_terms: int = 300
    last_breakdown: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _norms: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, zidx: ZoneIndex, cfg, use_zones: bool = True) -> "BM25F":
        return cls(zidx=zidx, zone_weights=dict(cfg.zones), k1=cfg.bm25.k1, b=cfg.bm25.b, use_zones=use_zones,
                   max_query_terms=cfg.bm25.get("max_query_terms", 300))

    # ------------------------------------------------------------------ statistics
    def idf(self, term: str) -> float:
        n, df = self.zidx.n_docs, self.zidx.df(term)
        if df == 0 or n == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _zone_norm(self, zone: str) -> dict[str, float]:
        """doc -> (1 - b + b * len / avglen) for one zone ('*' = whole document)."""
        if zone not in self._norms:
            idx = self.zidx.whole if zone == "*" else self.zidx.zones[zone]
            lengths = idx.doc_len
            avg = (sum(lengths.values()) / len(lengths)) if lengths else 1.0
            self._norms[zone] = {d: 1 - self.b + self.b * (ln / avg if avg else 0.0) for d, ln in lengths.items()}
        return self._norms[zone]

    def _select_terms(self, terms: dict[str, float]) -> dict[str, float]:
        if len(terms) <= self.max_query_terms:
            return terms
        ranked = sorted(terms.items(), key=lambda kv: -(self.idf(kv[0]) * kv[1]))
        return dict(ranked[: self.max_query_terms])

    # ------------------------------------------------------------------ scoring
    def score(self, terms: dict[str, float], candidates: set[str] | None = None) -> dict[str, float]:
        """BM25F score for every candidate doc containing at least one query term.

        `terms`: term -> query weight (AnalyzedQuery.weighted_terms()). `last_breakdown` keeps,
        per doc, each zone's share of tf~ for --debug.
        """
        terms = self._select_terms(terms)
        zones = [(z, w) for z, w in self.zone_weights.items() if w > 0 and z in self.zidx.zones] if self.use_zones else [("*", 1.0)]
        scores: dict[str, float] = {}
        breakdown: dict[str, dict[str, float]] = {}
        for term, qw in terms.items():
            idf = self.idf(term)
            if idf <= 0 or qw <= 0:
                continue
            tf_tilde: dict[str, float] = {}
            for zone, weight in zones:
                idx = self.zidx.whole if zone == "*" else self.zidx.zones[zone]
                postings = idx.postings.get(term)
                if not postings:
                    continue
                norms = self._zone_norm(zone)
                for doc, positions in postings.items():
                    if candidates is not None and doc not in candidates:
                        continue
                    part = weight * count(positions) / norms.get(doc, 1.0)
                    tf_tilde[doc] = tf_tilde.get(doc, 0.0) + part
                    if self.use_zones:
                        br = breakdown.setdefault(doc, {})
                        br[zone] = br.get(zone, 0.0) + part
            for doc, tft in tf_tilde.items():
                scores[doc] = scores.get(doc, 0.0) + qw * idf * tft * (self.k1 + 1) / (tft + self.k1)
        self.last_breakdown = breakdown
        return scores
