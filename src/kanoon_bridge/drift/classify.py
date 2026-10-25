"""A learned change detector over the drift vector: L2-regularised logistic regression.  [owner: Gaurav — working]

The rule taxonomy in model.py is set by hand. This module asks a second question: how much of
each gold source's notion of "changed" is recoverable from the drift vector at all? Features
are z-scored on the training fold; the model is fitted by Newton's method (IRLS), which
converges in a few iterations on 15 features and a few hundred sections. Numpy only.

    fit(X, y, l2) -> LogReg;  LogReg.predict_proba(X)
    cross_val_predict(X, y, folds=5, seed=0)   out-of-fold probabilities for every row
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogReg:
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray

    def decision(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, float) - self.mu) / self.sd) @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def coefficients(self, names: list[str]) -> list[tuple[str, float]]:
        """Standardised coefficients, largest magnitude first (for the report)."""
        return sorted(zip(names, self.w.tolist()), key=lambda x: -abs(x[1]))


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> LogReg:
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    beta = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[-1] = 0.0                                   # the intercept is not penalised
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
        g = Z.T @ (p - y) + reg * beta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return LogReg(beta[:-1], float(beta[-1]), mu, sd)


def cross_val_predict(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, l2: float = 1.0) -> np.ndarray:
    """Out-of-fold P(changed) for every row, stratified by label."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % folds
    out = np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        if te.any():
            out[te] = fit(X[tr], y[tr], l2).predict_proba(X[te])
    return out


__all__ = ["LogReg", "fit", "cross_val_predict"]
