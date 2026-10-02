"""Policy evaluation, operating-point selection and paired stratified bootstrap.

A policy's outcome on a query set is a pair of per-query arrays (acc_q, cost_q); for randomised policies these are
expectations, so every method is evaluated and bootstrapped the same way.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def outcome_of_choice(choice: np.ndarray, Y: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.arange(len(choice))
    return Y[r, choice].astype(float), C[r, choice]


def outcome_of_mix(w: np.ndarray, Y: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return Y @ w, C @ w


def pick_operating_point(knobs, val_acc, val_cost, target: float):
    """Knob with the lowest validation cost among those reaching the target (ties: first in grid order).

    Returns None if no knob value reaches the target.
    """
    val_acc, val_cost = np.asarray(val_acc), np.asarray(val_cost)
    feasible = np.flatnonzero(val_acc >= target - 1e-12)
    if len(feasible) == 0:
        return None
    best = feasible[np.argmin(val_cost[feasible])]
    return knobs[best]


def stratified_bootstrap_indices(strata: np.ndarray, B: int, seed: int) -> np.ndarray:
    """(B, n) index matrix; resampling within each stratum keeps the domain mix fixed. Shared by all methods."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty((B, len(strata)), dtype=np.int64)
    for b in range(B):
        out[b] = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
    return out


def boot_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return x[idx].mean(axis=1)


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def p_greater_than_zero(samples: np.ndarray) -> float:
    """One-sided bootstrap p-value for H1: statistic > 0."""
    return float((np.sum(samples <= 0) + 1) / (len(samples) + 1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else float("nan")


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [
        (float(p[which == b].mean()), float(y[which == b].mean()), int((which == b).sum()))
        for b in range(bins)
        if (which == b).any()
    ]
