"""Merge ranked lists from several sub-queries.  [layer 2 — working]

Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009), the default:

    RRF(d) = sum over lists L containing d of  w_L / (k + rank_L(d))      k = 60

It needs no score normalisation, which is why it suits mixing Boolean, facet and free-text
lists whose scores are not comparable.

CombSUM (Fox & Shaw, TREC-2 1994), the comparison point in eval/agent_eval.py:

    CombSUM(d) = sum over lists L containing d of  w_L * minmax_L(score_L(d))

Which sub-results feed which list: a sub-query feeds its `target` list; the 'original'
sub-query feeds both, so a statutes-first or facet sub-query refines rather than replaces
the user's own query.
"""

from __future__ import annotations

from kanoon_bridge.agent.executor import SubResult


def reciprocal_rank_fusion(rankings: list[list[str]], weights: list[float] | None = None,
                           k: int = 60) -> dict[str, float]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return fused


def _contributing(subresults: list[SubResult], target: str) -> list[SubResult]:
    return [sr for sr in subresults if sr.subquery.target == target or sr.subquery.kind == "original"]


def fuse_subresults(subresults: list[SubResult], target: str, k: int = 60) -> dict[str, float]:
    """RRF over the sub-results that contribute to `target` ('precedent' or 'statute')."""
    srs = _contributing(subresults, target)
    return reciprocal_rank_fusion([sr.ranking(target) for sr in srs], [sr.subquery.weight for sr in srs], k)


def combsum(subresults: list[SubResult], target: str) -> dict[str, float]:
    """Min-max normalise each list's scores to [0, 1], then the weighted sum per document."""
    fused: dict[str, float] = {}
    for sr in _contributing(subresults, target):
        ids, scores = sr.ranking(target), sr.scores(target)
        if not ids:
            continue
        lo, hi = min(scores), max(scores)
        for d, s in zip(ids, scores):
            norm = (s - lo) / (hi - lo) if hi > lo else 1.0
            fused[d] = fused.get(d, 0.0) + sr.subquery.weight * norm
    return fused


def fuse(subresults: list[SubResult], target: str, method: str = "rrf", k: int = 60) -> dict[str, float]:
    if method == "combsum":
        return combsum(subresults, target)
    if method == "rrf":
        return fuse_subresults(subresults, target, k)
    raise ValueError(f"unknown fusion method {method!r} (rrf | combsum)")
