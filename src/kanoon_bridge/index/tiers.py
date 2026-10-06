"""Tiered index and champion lists, for the efficiency experiment.  [owner: Gaurav — working]

Tier 1 = precedents with high authority g(d) (top 20% by default, configs: tiers.*).
A query is scored on tier 1 first; tier 2 is added only if tier 1 gives fewer than
`min_results` candidates. Champion lists keep, per term, the r docs with the highest
tf x (1 + g(d)) — the lecture's "champion lists with static quality".

eval/efficiency.py compares exhaustive scoring vs tiered vs champion lists on latency and
Recall@20. Candidates from here are passed to BM25F.score(candidates=...).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from kanoon_bridge.index.positional import PositionalIndex
from kanoon_bridge.index.zones import ZoneIndex


@dataclass
class TieredIndex:
    tier1: set[str] = field(default_factory=set)
    tier2: set[str] = field(default_factory=set)
    champions: dict[str, list[str]] = field(default_factory=dict)   # term -> top-r doc ids

    @classmethod
    def build(cls, zidx: ZoneIndex, authority: dict[str, float], quantile: float = 0.8,
              champion_size: int = 50) -> "TieredIndex":
        whole: PositionalIndex = zidx.whole
        docs = list(whole.doc_len)
        k = max(1, round((1 - quantile) * len(docs))) if docs else 0       # top (1 - quantile) share
        ranked = sorted(docs, key=lambda d: -authority.get(d, 0.0))
        tier1 = {d for d in ranked[:k] if authority.get(d, 0.0) > 0}
        champions = {
            term: [d for _, d in heapq.nlargest(
                champion_size, ((len(pos) * (1 + authority.get(d, 0.0)), d) for d, pos in postings.items()))]
            for term, postings in whole.postings.items()
        }
        return cls(tier1=tier1, tier2=set(docs) - tier1, champions=champions)

    def candidates(self, terms: list[str], min_results: int = 20, use_champions: bool = False,
                   index: PositionalIndex | None = None) -> set[str]:
        """Docs to score for these terms.

        Champion mode: union of the terms' champion lists (no index needed).
        Tiered mode: docs containing any term, from tier 1 first, adding tier 2 when tier 1 yields
        fewer than `min_results`; needs the whole-document index.
        """
        if use_champions:
            out: set[str] = set()
            for t in terms:
                out.update(self.champions.get(t, ()))
            return out
        if index is None:
            raise ValueError("tiered mode needs the whole-document PositionalIndex")
        matching: set[str] = set()
        for t in terms:
            matching.update(index.postings.get(t, {}))
        first = matching & self.tier1
        return first if len(first) >= min_results else first | (matching & self.tier2)
