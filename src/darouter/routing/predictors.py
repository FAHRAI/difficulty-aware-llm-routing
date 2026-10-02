"""Per-model success predictors used as baselines: k nearest neighbours and per-model logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

KNN_KS = (10, 25, 50, 100)
LR_CS = (0.01, 0.1, 1.0, 10.0)


def _logloss(p: np.ndarray, Y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())


class KNNRouter:
    """p̂_m(q) = (sum of Y_m over k nearest train queries + 1) / (k + 2); cosine distance on normalised embeddings."""

    def __init__(self, seed: int = 20261001):
        self.seed = seed

    @staticmethod
    def _proba(nn: NearestNeighbors, Ytr: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
        _, ind = nn.kneighbors(X, n_neighbors=k)
        return (Ytr[ind].sum(axis=1) + 1) / (k + 2)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> KNNRouter:
        losses = {k: 0.0 for k in KNN_KS}
        for tr, ho in KFold(5, shuffle=True, random_state=self.seed).split(X):
            nn = NearestNeighbors(metric="cosine").fit(X[tr])
            _, ind = nn.kneighbors(X[ho], n_neighbors=max(KNN_KS))
            for k in KNN_KS:
                p = (Y[tr][ind[:, :k]].sum(axis=1) + 1) / (k + 2)
                losses[k] += _logloss(p, Y[ho])
        self.k_ = min(losses, key=losses.get)
        self.nn_, self.Y_ = NearestNeighbors(metric="cosine").fit(X), Y
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(self.nn_, self.Y_, X, self.k_)


class PerModelLR:
    """One L2 logistic regression per model on the same features (CARROT-style linear plug-in estimate)."""

    def __init__(self, seed: int = 20261001):
        self.seed = seed

    def fit(self, X: np.ndarray, Y: np.ndarray) -> PerModelLR:
        losses = {c: 0.0 for c in LR_CS}
        for tr, ho in KFold(5, shuffle=True, random_state=self.seed).split(X):
            for c in LR_CS:
                p = np.column_stack(
                    [
                        LogisticRegression(C=c, max_iter=2000).fit(X[tr], Y[tr, j]).predict_proba(X[ho])[:, 1]
                        for j in range(Y.shape[1])
                    ]
                )
                losses[c] += _logloss(p, Y[ho])
        self.C_ = min(losses, key=losses.get)
        self.models_ = [LogisticRegression(C=self.C_, max_iter=2000).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
