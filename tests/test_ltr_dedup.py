"""Learning to rank (coordinate ascent) and near-duplicate detection."""

import numpy as np

from kanoon_bridge.rank.ltr import (FEATURES, LinearRanker, average_precision, coordinate_ascent,
                                    cross_validate, mean_ap)


def test_average_precision():
    assert average_precision(np.array([3.0, 2.0, 1.0]), np.array([1, 0, 1]), 2) == (1 + 2 / 3) / 2
    assert average_precision(np.array([1.0]), np.array([0]), 1) == 0.0


def _synthetic(n_queries=30, seed=0):
    """Relevance depends on feature 'offence_match'; lexical alone is noisy."""
    rng = np.random.default_rng(seed)
    j = FEATURES.index("offence_match")
    data = []
    for _ in range(n_queries):
        X = rng.random((20, len(FEATURES)))
        y = (X[:, j] > 0.8).astype(float)
        if y.sum() == 0:
            y[0] = 1
            X[0, j] = 0.9
        data.append((X, y, int(y.sum())))
    return data


def test_coordinate_ascent_learns_the_useful_feature():
    data = _synthetic()
    w, train_map = coordinate_ascent(data)
    assert train_map > mean_ap(np.eye(len(FEATURES))[0], data) + 0.2
    assert w[FEATURES.index("offence_match")] > 0.5
    cv = cross_validate(data, folds=3)
    assert cv["cv_map_ltr"] > cv["cv_map_handset"]


def test_ranker_roundtrip():
    r = LinearRanker(weights=np.arange(len(FEATURES), dtype=float), depth=50)
    r2 = LinearRanker.from_dict(r.to_dict())
    assert np.allclose(r.weights, r2.weights) and r2.depth == 50


def test_near_duplicates():
    import random

    from kanoon_bridge.index.dedup import near_duplicate_groups
    from kanoon_bridge.schema import Document, DocType, Paragraph

    rng = random.Random(3)
    words = "accused murder land quarrel court appeal evidence witness knife injury death sentence".split()
    base = " ".join(rng.choice(words) + str(i % 7) for i in range(300))
    other = " ".join(rng.choice(words) + str(i % 5) for i in range(300))
    docs = [Document("a", DocType.PRECEDENT, paragraphs=[Paragraph(base)]),
            Document("b", DocType.PRECEDENT, paragraphs=[Paragraph(base.replace(base.split()[150], "changed", 1))]),
            Document("c", DocType.PRECEDENT, paragraphs=[Paragraph(other)])]
    assert near_duplicate_groups(docs) == {"a": "a", "b": "a"}
