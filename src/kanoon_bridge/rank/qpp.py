"""Query performance prediction (QPP) without relevance labels.  [owner: Gaurav — working]

Predicts how well retrieval is going for this query. Used by:
  * search.py / fusion: set the lexical/dense weight alpha per query
  * agent/reflect.py: decide whether to reformulate
  * rag/abstain.py: refuse to answer when retrieval looks weak

Pre-retrieval predictors (query + index statistics):
    max_idf, avg_idf           specific terms predict good retrieval
    query_scope                share of the collection touched by the query terms (lower = sharper)
Post-retrieval predictors (the ranked score list):
    score_gap = (s1 - s2) / s1          a clear winner predicts a good ranking
    normalised_top = s1 / mean(s1..s10)
    top_std = std(s1..s10) / mean       spread of the head of the ranking

Background: Tian et al., "What can predicted query performance tell us about agentic RAG",
IR-RAG workshop at SIGIR 2025. Thresholds below are simple and should be tuned on VALIDATION.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class QPPFeatures:
    max_idf: float = 0.0
    avg_idf: float = 0.0
    query_scope: float = 0.0
    score_gap: float = 0.0
    normalised_top: float = 0.0
    top_std: float = 0.0


def pre_retrieval(terms: list[str], idf: dict[str, float], df: dict[str, int], n_docs: int) -> QPPFeatures:
    """Index-statistics predictors. Terms unseen in the index (idf 0) are ignored."""
    vals = [idf.get(t, 0.0) for t in terms if idf.get(t, 0.0) > 0]
    scope = min(1.0, sum(df.get(t, 0) for t in terms) / n_docs) if n_docs else 0.0
    return QPPFeatures(max_idf=max(vals, default=0.0), avg_idf=statistics.fmean(vals) if vals else 0.0,
                       query_scope=scope)


def post_retrieval(features: QPPFeatures, scores: list[float], head: int = 10) -> QPPFeatures:
    """Fill the score-based predictors from a descending score list."""
    top = [s for s in scores[:head] if s is not None]
    if not top or top[0] <= 0:
        features.score_gap = features.normalised_top = features.top_std = 0.0
        return features
    mean = statistics.fmean(top)
    features.score_gap = (top[0] - top[1]) / top[0] if len(top) > 1 else 1.0
    features.normalised_top = top[0] / mean if mean else 0.0
    features.top_std = (statistics.pstdev(top) / mean) if len(top) > 1 and mean else 0.0
    return features


def confidence(f: QPPFeatures, idf_scale: float = 3.0, gap_scale: float = 0.2) -> float:
    """A single 0..1 retrieval-confidence score: half term specificity, half head separation."""
    specificity = min(1.0, f.max_idf / idf_scale) if idf_scale else 0.0
    separation = min(1.0, f.score_gap / gap_scale) if gap_scale else 0.0
    return 0.5 * specificity + 0.5 * separation


def alpha_from_qpp(f: QPPFeatures, default: float = 0.7, lo: float = 0.4, hi: float = 0.9) -> float:
    """Lexical weight in [lo, hi]: confident lexical retrieval -> lean on BM25F; weak -> let the
    dense channel help. `default` is returned when no signal is available."""
    if f.max_idf == 0 and f.score_gap == 0:
        return default
    return lo + (hi - lo) * confidence(f)


def should_abstain(f: QPPFeatures, gap_threshold: float = 0.05, top_threshold: float = 1.1,
                   min_idf: float = 0.3) -> bool:
    """True when retrieval looks too weak to ground an answer: no specific term, or a flat head
    (tiny gap AND top score barely above the mean)."""
    if f.max_idf < min_idf:
        return True
    return f.score_gap < gap_threshold and f.normalised_top < top_threshold
