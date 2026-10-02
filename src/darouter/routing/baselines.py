"""Query-independent and response-aware baselines."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from darouter.difficulty.features import head_tail


def best_mix_weights(acc: np.ndarray, cost: np.ndarray, target: float) -> np.ndarray | None:
    """Cheapest random mixture over models with expected accuracy >= target (LP). None if infeasible."""
    m = len(acc)
    res = linprog(
        c=cost,
        A_ub=-acc[None, :],
        b_ub=[-target],
        A_eq=np.ones((1, m)),
        b_eq=[1.0],
        bounds=[(0, 1)] * m,
        method="highs",
    )
    return res.x if res.status == 0 else None


class StaticCategory:
    """Category x model success rates from train, shrunk toward the model's overall rate (k pseudo-observations)."""

    def __init__(self, k: float = 20.0):
        self.k = k

    def fit(self, categories: np.ndarray, Y: np.ndarray) -> StaticCategory:
        overall = Y.mean(axis=0)
        self.overall_, self.table_ = overall, {}
        for c in np.unique(categories):
            mask = categories == c
            self.table_[c] = (Y[mask].sum(axis=0) + self.k * overall) / (mask.sum() + self.k)
        return self

    def proba(self, categories: np.ndarray) -> np.ndarray:
        return np.vstack([self.table_.get(c, self.overall_) for c in categories])


class Cascade:
    """FrugalGPT-style cascade along a fixed chain; scorer g_k(query ⊕ response) per non-final stage."""

    def __init__(self, chain: list[int]):
        self.chain = chain

    @staticmethod
    def _docs(queries, responses):
        return [head_tail(q) + " [RESPONSE] " + head_tail(r or "") for q, r in zip(queries, responses)]

    def fit(self, queries, responses: dict[int, list[str]], Y: np.ndarray) -> Cascade:
        self.scorers_ = []
        for m in self.chain[:-1]:
            v = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, min_df=2)
            X = v.fit_transform(self._docs(queries, responses[m]))
            self.scorers_.append((v, LogisticRegression(C=1.0, max_iter=2000).fit(X, Y[:, m])))
        return self

    def scores(self, queries, responses: dict[int, list[str]]) -> np.ndarray:
        """(n, len(chain)-1) acceptance scores, computed once and reused for every threshold."""
        return np.column_stack(
            [
                lr.predict_proba(v.transform(self._docs(queries, responses[m])))[:, 1]
                for (v, lr), m in zip(self.scorers_, self.chain[:-1])
            ]
        )

    def outcome(self, g: np.ndarray, t: float, Y: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(g)
        done = np.zeros(n, bool)
        acc, cost = np.zeros(n), np.zeros(n)
        for k, m in enumerate(self.chain):
            active = ~done
            cost[active] += C[active, m]
            accept = active if k == len(self.chain) - 1 else active & (g[:, k] >= t)
            acc[accept] = Y[accept, m]
            done |= accept
        return acc, cost
