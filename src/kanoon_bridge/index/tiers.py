"""Tiered index and champion lists, for the efficiency experiment.  [owner: D]

Tier 1 = precedents with high authority g(d) (top 20% by default, configs: tiers.*).
A query is scored on tier 1 first; tier 2 is used only if tier 1 gives fewer than
`min_results_before_tier2` candidates. Champion lists keep, per term, the r docs with the
highest tf (or tf x g(d)).

eval/efficiency.py compares: exhaustive scoring vs tiered vs champion lists, on latency
and Recall@20.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanoon_bridge.index.zones import ZoneIndex


@dataclass
class TieredIndex:
    tier1: set[str] = field(default_factory=set)
    tier2: set[str] = field(default_factory=set)
    champions: dict[str, list[str]] = field(default_factory=dict)   # term -> top-r doc ids

    @classmethod
    def build(cls, zidx: ZoneIndex, authority: dict[str, float], quantile: float = 0.8,
              champion_size: int = 50) -> "TieredIndex":
        """TODO(D): split docs by the authority quantile; build champion lists from
        whole-index tf (optionally weighted by authority)."""
        raise NotImplementedError("TODO(D): tiered index + champion lists")

    def candidates(self, terms: list[str], min_results: int = 20, use_champions: bool = False) -> set[str]:
        """Docs to score for these terms.

        TODO(D): champion mode -> union of champion lists; tiered mode -> docs in tier1
        containing any term, falling back to tier2 when fewer than min_results.
        """
        raise NotImplementedError("TODO(D): tiered candidate generation")
