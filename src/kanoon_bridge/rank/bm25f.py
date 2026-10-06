"""BM25F: BM25 with per-zone weights.  [owner: D — main lexical scorer]

For a query term t and document d with zones z:

    tf~(t, d) = sum_z  w_z * tf(t, d, z) / (1 - b + b * len(d, z) / avglen(z))
    score(q, d) = sum_t  qw(t) * idf(t) * tf~ * (k1 + 1) / (tf~ + k1)
    idf(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )

Zone weights, k1 and b come from configs/default.yaml (zones.*, bm25.*).
With use_zones=False every zone weight is 1.0 and lengths are whole-document, which is
plain BM25 — the first rung of the ablation ladder.

IL-PCSR found BM25 to be the strongest single method for precedent retrieval, so this
file matters most for E1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.index.zones import ZoneIndex


@dataclass
class BM25F:
    zidx: ZoneIndex
    zone_weights: dict[str, float] = field(default_factory=dict)
    k1: float = 1.6
    b: float = 0.7
    use_zones: bool = True

    @classmethod
    def from_config(cls, zidx: ZoneIndex, cfg, use_zones: bool = True) -> "BM25F":
        return cls(zidx=zidx, zone_weights=dict(cfg.zones), k1=cfg.bm25.k1, b=cfg.bm25.b, use_zones=use_zones)

    def idf(self, term: str) -> float:
        """TODO(D): BM25 idf with the +1 inside the log (never negative)."""
        raise NotImplementedError("TODO(D): BM25 idf")

    def score(self, terms: dict[str, float], candidates: set[str] | None = None) -> dict[str, float]:
        """BM25F score for every candidate doc containing at least one query term.

        `terms`: term -> query weight (from AnalyzedQuery.weighted_terms()).
        TODO(D): iterate query terms -> their postings in each zone -> accumulate tf~ per doc,
        then apply the saturation formula. Skip docs not in `candidates` when it is given.
        Keep per-zone contributions if cheap; they make a good --debug breakdown.
        """
        raise NotImplementedError("TODO(D): BM25F scoring")
