"""Model pool definitions and the cost model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pool:
    models: list[str]
    price_in: np.ndarray  # USD per 1M input tokens
    price_out: np.ndarray  # USD per 1M output tokens

    @classmethod
    def from_config(cls, cfg: dict, name: str, all_models: list[str]) -> Pool:
        models = all_models if cfg["pools"][name] == "all" else list(cfg["pools"][name])
        p = np.array([cfg["prices"][m] for m in models], dtype=float)
        return cls(models, p[:, 0], p[:, 1])


def matrix(df: pd.DataFrame, prefix: str, models: list[str]) -> np.ndarray:
    return df[[f"{prefix}__{m}" for m in models]].to_numpy()


def realised_cost(df: pd.DataFrame, pool: Pool) -> np.ndarray:
    """C[i, m]: USD actually spent if query i is sent to model m."""
    nin, nout = matrix(df, "nin", pool.models), matrix(df, "nout", pool.models)
    return (nin * pool.price_in + nout * pool.price_out) / 1e6


def expected_cost(df: pd.DataFrame, pool: Pool, mean_nout: np.ndarray) -> np.ndarray:
    """Ĉ[i, m]: cost known at routing time (input tokens exact, output tokens = training mean)."""
    nin = matrix(df, "nin", pool.models)
    return (nin * pool.price_in + mean_nout[None, :] * pool.price_out) / 1e6
