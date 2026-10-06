"""Heap-based top-K selection and the final net score.  [owner: D — working]

    net(d) = relevance(d) + lambda * g(d | state)

Selecting K results with a heap is O(n log K) instead of sorting all n scores —
the lecture's "efficient top-K" point; mention it in the video.
"""

from __future__ import annotations

import heapq

from kanoon_bridge.schema import DocType, ScoredDoc


def top_k(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    """K highest (doc_id, score) pairs, best first; ties broken by doc_id for determinism."""
    best = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], kv[0]))
    return best


def net_score(relevance: float, authority: float, lam: float) -> float:
    return relevance + lam * authority


def to_scored(ranked: list[tuple[str, float]], doc_type: DocType,
              components: dict[str, dict[str, float]] | None = None) -> list[ScoredDoc]:
    components = components or {}
    return [
        ScoredDoc(doc_id=d, score=s, doc_type=doc_type, components=components.get(d, {}), rank=i + 1)
        for i, (d, s) in enumerate(ranked)
    ]
