"""Rasch (1PL) model P(Y_im = 1) = sigmoid(theta_m - b_i), MAP with b_i ~ N(0, prior_sd^2), theta flat."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_rasch(Y: np.ndarray, prior_sd: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Y: (n_items, n_models) binary. Returns (theta, b)."""
    n, m = Y.shape
    Y = Y.astype(float)
    prec = 1.0 / prior_sd**2

    def fg(x):
        theta, b = x[:m], x[m:]
        p = expit(theta[None, :] - b[:, None])
        eps = 1e-12
        nll = -(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)).sum() + 0.5 * prec * (b**2).sum()
        r = Y - p  # d loglik / d logit
        g_theta = -r.sum(axis=0)
        g_b = r.sum(axis=1) + prec * b
        return nll, np.concatenate([g_theta, g_b])

    res = minimize(fg, np.zeros(m + n), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-6})
    if not res.success:
        raise RuntimeError(f"Rasch fit did not converge: {res.message}")
    return res.x[:m], res.x[m:]


def item_difficulty(Y: np.ndarray, theta: np.ndarray, prior_sd: float = 2.0, iters: int = 50) -> np.ndarray:
    """MAP difficulty of each item given fixed abilities (vectorised Newton). Depends on Y only via row sums."""
    prec = 1.0 / prior_sd**2
    s = Y.sum(axis=1).astype(float)
    b = np.zeros(len(Y))
    for _ in range(iters):
        p = expit(theta[None, :] - b[:, None])
        g = p.sum(axis=1) - s - prec * b  # d logpost / d b
        h = -(p * (1 - p)).sum(axis=1) - prec  # second derivative (< 0)
        b = b - g / h
    return b
