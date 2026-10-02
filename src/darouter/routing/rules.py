"""Model-selection rules, vectorised over queries.

p:  (n, M) estimated success probabilities
ch: (n, M) plug-in routing cost Ĉ (known before generation)
Returns chosen model index per query.
"""

from __future__ import annotations

import numpy as np


def _cost_order(ch: np.ndarray) -> np.ndarray:
    # ties -> lower model index (stable sort)
    return np.argsort(ch, axis=1, kind="stable")


def select_r1(p: np.ndarray, ch: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """argmin Ĉ s.t. p̂ >= tau; fallback argmax p̂ (ties -> lower Ĉ). Returns (choice, fallback_mask)."""
    order = _cost_order(ch)
    ok_sorted = np.take_along_axis(p >= tau, order, axis=1)
    any_ok = ok_sorted.any(axis=1)
    first = order[np.arange(len(p)), ok_sorted.argmax(axis=1)]
    # fallback: highest p̂, ties broken by lower Ĉ
    p_sorted = np.take_along_axis(p, order, axis=1)
    best = order[np.arange(len(p)), p_sorted.argmax(axis=1)]
    return np.where(any_ok, first, best), ~any_ok


def select_r2(p: np.ndarray, ch: np.ndarray, lam: float, cbar: float) -> np.ndarray:
    """argmax p̂ − lam·Ĉ/cbar (ties -> lower Ĉ)."""
    order = _cost_order(ch)
    u_sorted = np.take_along_axis(p - lam * ch / cbar, order, axis=1)
    return order[np.arange(len(p)), u_sorted.argmax(axis=1)]


TAU_GRID = np.round(np.arange(0, 1.0001, 0.005), 3)
LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-3, 1, 81)])
