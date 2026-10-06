"""Combine lexical and dense scores.  [owner: C]

    fused(d) = alpha * norm(lexical)(d) + (1 - alpha) * norm(dense)(d)

Score ranges differ (BM25 is unbounded, cosine is in [-1, 1]), so normalise per query
first. IL-PCSR uses z-score normalisation; min-max is the simple alternative — compare both.

Working: minmax, zscore, fuse. TODO(C): choose alpha per query with rank/qpp.py
(when configs fusion.qpp_gated is true) and report the effect in the ablation.
"""

from __future__ import annotations

import statistics


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def zscore(scores: dict[str, float]) -> dict[str, float]:
    if len(scores) < 2:
        return {d: 0.0 for d in scores}
    mu = statistics.fmean(scores.values())
    sd = statistics.pstdev(scores.values()) or 1.0
    return {d: (s - mu) / sd for d, s in scores.items()}


def fuse(lexical: dict[str, float], dense: dict[str, float], alpha: float, method: str = "minmax") -> dict[str, float]:
    """Weighted sum of normalised scores; a doc missing from one channel gets that channel's minimum."""
    norm = minmax if method == "minmax" else zscore
    lx, dn = norm(lexical), norm(dense)
    lx_min = min(lx.values(), default=0.0)
    dn_min = min(dn.values(), default=0.0)
    return {d: alpha * lx.get(d, lx_min) + (1 - alpha) * dn.get(d, dn_min) for d in set(lx) | set(dn)}
