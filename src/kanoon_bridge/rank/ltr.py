"""Learning to rank: a linear re-ranker trained by coordinate ascent.  [owner: Gaurav — working]

The hand-set combination  relevance + bridge + lambda * g(d | state)  becomes a learned one.
Model and optimiser: Metzler & Croft, "Linear feature-based models for information retrieval",
Information Retrieval 10(3), 2007 (UMass Amherst) — coordinate ascent that directly maximises
MAP (a listwise, metric-driven method; RankLib's default linear learner). Lecture: IIR ch. 15
"machine-learned relevance".

Features per (query, precedent), all in [0, 1] (per-query max-normalised where unbounded):

    lexical        BM25F (or tf-idf) score / best in the query
    zone_facts, zone_ratio, zone_arguments, zone_decision, zone_other
                   each zone's share of the BM25F match (SAILER, SIGIR 2023, and IL-PCSR both
                   find that a judgment's reasoning and facts carry different signal)
    bridge         statute-bridge boost (shares a statute ranked high for the query)
    authority      g(d | state), jurisdiction-aware PageRank
    dense          dense cosine (0 when the dense channel is off)
    offence_match  share of the query's offences (IPC or BNS, via the crosswalk) the judgment cites
                   ("key element" matching; cf. LeCaRD, Ma et al. SIGIR 2021)
    binding        1 binding in the user's state, 0.5 unknown, 0 persuasive
    supreme_court  1 for Supreme Court judgments
    recency        exp(-|incident year - decision year| / 10)

Training data: IL-PCSR VALIDATION queries (the citation graph that feeds `authority` is built
from TRAIN qrels, so training on train queries would reward authority for edges those very
queries created). k for F1@k is also chosen on val; test stays untouched.
scripts/06_train_ltr.py trains, reports 5-fold cross-validated MAP on val, and saves
data/processed/index/ltr.json; search.py applies it with SearchOptions(ltr=True) to the top
`ltr.depth` candidates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

FEATURES = ("lexical", "zone_facts", "zone_ratio", "zone_arguments", "zone_decision", "zone_other",
            "bridge", "authority", "dense", "offence_match", "binding", "supreme_court", "recency", "ngram")


def feature_rows(hits, aq, engine) -> np.ndarray:
    """Feature matrix [len(hits), len(FEATURES)] for one query's precedent hits (ScoredDoc with
    the components search.py fills)."""
    from kanoon_bridge.rank.authority import binding_status

    norm = engine.analyzer.text_res.normalizer
    q_off = set(aq.offence_ids) | {t for t in aq.expanded_terms if t.startswith("off:")}
    year = aq.query.incident_date.year if aq.query.incident_date else None
    X = np.zeros((len(hits), len(FEATURES)), dtype=np.float64)
    for i, h in enumerate(hits):
        c = h.components or {}
        meta = engine.facets.metas.get(h.doc_id)
        X[i, 0] = c.get("lexical", 0.0)
        for j, z in enumerate(("facts", "ratio", "arguments", "decision", "other"), start=1):
            X[i, j] = c.get(f"zone_{z}", 0.0)
        X[i, 6] = c.get("bridge", 0.0)
        X[i, 7] = c.get("authority", 0.0)
        X[i, 8] = c.get("dense", 0.0)
        if meta is not None:
            if q_off and norm is not None:
                cited = {o for ref in meta.statutes_cited for o in norm.offences_for(ref)}
                X[i, 9] = len(cited & q_off) / len(q_off)
            status = binding_status(meta, aq.query.state)
            X[i, 10] = {"binding": 1.0, "unknown": 0.5}.get(status, 0.0)
            X[i, 11] = 1.0 if meta.court == "supreme_court" else 0.0
            if year and meta.decision_date:
                X[i, 12] = math.exp(-abs(year - meta.decision_date.year) / 10)
            else:
                X[i, 12] = 0.5
        X[i, 13] = c.get("ngram", 0.0)
    for j in (0, 1, 2, 3, 4, 5, 6, 13):                       # unbounded columns: per-query max-normalise
        top = X[:, j].max() if len(X) else 0.0
        if top > 0:
            X[:, j] /= top
    return X


def average_precision(scores: np.ndarray, labels: np.ndarray, n_relevant: int) -> float:
    if n_relevant <= 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    rel = labels[order] > 0
    if not rel.any():
        return 0.0
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    return float((hits[rel] / ranks[rel]).sum() / n_relevant)


def mean_ap(w: np.ndarray, data: list[tuple[np.ndarray, np.ndarray, int]]) -> float:
    return float(np.mean([average_precision(X @ w, y, n) for X, y, n in data])) if data else 0.0


def coordinate_ascent(data: list[tuple[np.ndarray, np.ndarray, int]], init: np.ndarray | None = None,
                      rounds: int = 4, deltas=(-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
                      fixed: tuple[int, ...] = (0,)) -> tuple[np.ndarray, float]:
    """Maximise MAP one weight at a time (line search over `deltas`); weight 0 (lexical) is fixed
    at 1 so the scale is identified. Returns (weights, training MAP)."""
    w = np.array(init if init is not None else _default_init(), dtype=np.float64)
    best = mean_ap(w, data)
    for _ in range(rounds):
        improved = False
        for j in range(len(w)):
            if j in fixed:
                continue
            base = w[j]
            for d in deltas:
                w[j] = base + d
                m = mean_ap(w, data)
                if m > best + 1e-9:
                    best, base, improved = m, w[j], True
            w[j] = base
        if not improved:
            break
    return w, best


def _default_init() -> np.ndarray:
    """Start from the hand-set system: relevance (unigram 0.3 + trigram 0.7) + bridge + 0.2 * authority."""
    w = np.zeros(len(FEATURES))
    w[FEATURES.index("lexical")] = 0.3
    w[FEATURES.index("ngram")] = 0.7
    w[FEATURES.index("bridge")] = 1.0
    w[FEATURES.index("authority")] = 0.2
    return w


def cross_validate(data, folds: int = 5, seed: int = 0, **kw) -> dict[str, float]:
    """k-fold CV MAP of coordinate ascent vs the hand-set initial weights (on held-out folds)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    parts = np.array_split(idx, folds)
    base, learned = [], []
    for k in range(folds):
        test = [data[i] for i in parts[k]]
        train = [data[i] for p, part in enumerate(parts) if p != k for i in part]
        if not test or not train:
            continue
        w, _ = coordinate_ascent(train, **kw)
        base.append(mean_ap(_default_init(), test))
        learned.append(mean_ap(w, test))
    return {"cv_map_handset": float(np.mean(base)) if base else 0.0,
            "cv_map_ltr": float(np.mean(learned)) if learned else 0.0, "folds": len(base)}


@dataclass
class LinearRanker:
    weights: np.ndarray
    features: tuple[str, ...] = FEATURES
    depth: int = 100
    info: dict = field(default_factory=dict)

    def score(self, hits, aq, engine) -> np.ndarray:
        return feature_rows(hits, aq, engine) @ self.weights

    def to_dict(self) -> dict:
        return {"features": list(self.features), "weights": [round(float(x), 6) for x in self.weights],
                "depth": self.depth, "info": self.info}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearRanker":
        if tuple(d["features"]) != FEATURES:
            raise ValueError("ltr.json was trained with a different feature set - retrain (scripts/06)")
        return cls(weights=np.array(d["weights"], dtype=np.float64), depth=int(d.get("depth", 100)),
                   info=d.get("info", {}))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
