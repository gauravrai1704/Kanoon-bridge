"""Merge ranked lists from several sub-queries.  [agent owner]

Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009), the default:

    RRF(d) = sum over lists L containing d of  w_L / (k + rank_L(d))      k = 60

It needs no score normalisation, which is why it suits mixing Boolean, facet and free-text
lists whose scores are not comparable.

Working: reciprocal_rank_fusion, fuse_subresults.
TODO(agent): combsum (normalised score sum) as the comparison point in eval/agent_eval.py.
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


def fuse_subresults(subresults: list[SubResult], target: str, k: int = 60) -> dict[str, float]:
    """RRF over the sub-results that contribute to `target` ('precedent' or 'statute').

    The original sub-query always contributes to both lists, so a statutes-first or facet
    sub-query refines rather than replaces the user's own query.
    """
    rankings, weights = [], []
    for sr in subresults:
        if sr.subquery.target == target or sr.subquery.kind == "original":
            rankings.append(sr.ranking(target))
            weights.append(sr.subquery.weight)
    return reciprocal_rank_fusion(rankings, weights, k)


def combsum(subresults: list[SubResult], target: str) -> dict[str, float]:
    """TODO(agent): min-max normalise each list's scores, then sum per doc (CombSUM)."""
    raise NotImplementedError("TODO(agent): CombSUM baseline for fusion")
