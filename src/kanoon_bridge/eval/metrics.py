"""Evaluation metrics.  [owner: D — working]

All functions take a ranked list of doc ids and a relevance dict {doc_id: grade}
(grade > 0 means relevant). Run-level helpers take {query_id: ranked list} and
{query_id: {doc_id: grade}}.

IL-PCSR protocol (to sit next to their Table 3): macro-F1@k with k chosen on validation
(`best_k`), plus MAP and MRR.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable

Ranking = list[str]
Rels = dict[str, int]


def _relevant(rels: Rels) -> set[str]:
    return {d for d, g in rels.items() if g > 0}


def precision_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    if k <= 0:
        return 0.0
    rel = _relevant(rels)
    return sum(d in rel for d in ranking[:k]) / k


def recall_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    rel = _relevant(rels)
    if not rel:
        return 0.0
    return sum(d in rel for d in ranking[:k]) / len(rel)


def f1_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    p, r = precision_at_k(ranking, rels, k), recall_at_k(ranking, rels, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision(ranking: Ranking, rels: Rels) -> float:
    rel = _relevant(rels)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def reciprocal_rank(ranking: Ranking, rels: Rels) -> float:
    rel = _relevant(rels)
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    """Graded nDCG with gain (2^g - 1); used for E6 (binding = 2, persuasive = 1)."""
    def dcg(grades: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg([rels.get(d, 0) for d in ranking[:k]])
    ideal = dcg(sorted((g for g in rels.values() if g > 0), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


# --------------------------------------------------------------------------- run level


def mean_over_queries(run: dict[str, Ranking], qrels: dict[str, Rels],
                      metric: Callable[..., float], *args) -> float:
    """Mean of `metric` over queries that have judgments (queries without qrels are skipped)."""
    vals = [metric(run.get(q, []), rels, *args) for q, rels in qrels.items() if _relevant(rels)]
    return statistics.fmean(vals) if vals else 0.0


def macro_f1_at_k(run: dict[str, Ranking], qrels: dict[str, Rels], k: int) -> float:
    return mean_over_queries(run, qrels, f1_at_k, k)


def best_k(run: dict[str, Ranking], qrels: dict[str, Rels], ks: range = range(1, 11)) -> int:
    """IL-PCSR protocol: choose k on VALIDATION by macro-F1@k, then report test at that k."""
    return max(ks, key=lambda k: macro_f1_at_k(run, qrels, k))


def evaluate(run: dict[str, Ranking], qrels: dict[str, Rels], ks: list[int] = (1, 5, 10, 20)) -> dict[str, float]:
    """All standard metrics in one dict, e.g. {'P@10': 0.31, 'R@10': ..., 'MAP': ..., 'MRR': ...}."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"] = mean_over_queries(run, qrels, precision_at_k, k)
        out[f"R@{k}"] = mean_over_queries(run, qrels, recall_at_k, k)
        out[f"F1@{k}"] = macro_f1_at_k(run, qrels, k)
        out[f"nDCG@{k}"] = mean_over_queries(run, qrels, ndcg_at_k, k)
    out["MAP"] = mean_over_queries(run, qrels, average_precision)
    out["MRR"] = mean_over_queries(run, qrels, reciprocal_rank)
    out["n_queries"] = float(sum(1 for r in qrels.values() if _relevant(r)))
    return out
