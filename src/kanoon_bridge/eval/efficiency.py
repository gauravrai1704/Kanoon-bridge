"""Efficiency: exhaustive vs tiered vs champion-list scoring.  [owner: D]

For each mode, over the E1 queries, record median and p95 latency (ms) and Recall@20
relative to exhaustive scoring. Output: results/tables/efficiency.csv and one chart.
"""

from __future__ import annotations


def compare_modes(n_queries: int = 200) -> list[dict]:
    """TODO(D): time SearchEngine.search with candidates from index/tiers.TieredIndex
    (tiered, champions) vs None (exhaustive); compute recall overlap with exhaustive top-20."""
    raise NotImplementedError("TODO(D): efficiency comparison")
