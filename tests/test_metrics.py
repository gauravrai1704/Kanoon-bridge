"""Metric formulas on hand-computed examples.  [D]"""

import math

from kanoon_bridge.eval.agreement import cohens_kappa, percent_agreement
from kanoon_bridge.eval.metrics import (average_precision, best_k, evaluate, f1_at_k, ndcg_at_k,
                                        precision_at_k, recall_at_k, reciprocal_rank)

RANK = ["a", "b", "c", "d"]
RELS = {"a": 1, "c": 1, "z": 1}


def test_precision_recall():
    assert precision_at_k(RANK, RELS, 2) == 0.5
    assert recall_at_k(RANK, RELS, 4) == 2 / 3


def test_f1():
    p, r = 0.5, 1 / 3
    assert math.isclose(f1_at_k(RANK, RELS, 2), 2 * p * r / (p + r))


def test_average_precision():
    assert math.isclose(average_precision(RANK, RELS), (1 / 1 + 2 / 3) / 3)


def test_mrr():
    assert reciprocal_rank(["x", "c"], RELS) == 0.5


def test_ndcg_perfect_is_one():
    assert math.isclose(ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2), 1.0)


def test_evaluate_and_best_k():
    run = {"q1": RANK}
    qrels = {"q1": RELS}
    out = evaluate(run, qrels, [1, 2])
    assert out["P@1"] == 1.0 and out["n_queries"] == 1.0
    assert best_k(run, qrels, range(1, 5)) in range(1, 5)


def test_agreement():
    a = {"q": {"d1": 1, "d2": 0, "d3": 1}}
    b = {"q": {"d1": 1, "d2": 1, "d3": 1}}
    assert math.isclose(percent_agreement(a, b), 2 / 3)
    assert cohens_kappa(a, a) == 1.0
