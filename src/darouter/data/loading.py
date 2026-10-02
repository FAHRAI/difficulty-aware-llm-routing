"""Assemble response matrices, costs and the feature store for one model pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from darouter.difficulty.estimators import FeatureStore
from darouter.difficulty.features import head_tail
from darouter.models.registry import Pool, expected_cost, matrix, realised_cost
from darouter.paths import CONFIGS, DATA_PROCESSED


def load_config(name: str = "default") -> dict:
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_sprout() -> tuple[pd.DataFrame, FeatureStore]:
    df = pd.read_parquet(DATA_PROCESSED / "sprout.parquet")
    emb = np.load(DATA_PROCESSED / "emb_minilm.npy")
    heur = np.load(DATA_PROCESSED / "heur.npy")
    fs = FeatureStore([head_tail(t) for t in df.prompt], heur, emb, df.category.to_numpy())
    return df, fs


def pool_arrays(df: pd.DataFrame, cfg: dict, pool_name: str, train_idx: np.ndarray):
    """Pool, success matrix Y, realised cost C and plug-in routing cost CH (mean output length from train_idx)."""
    all_models = [c[3:] for c in df.columns if c.startswith("y__")]
    pool = Pool.from_config(cfg, pool_name, all_models)
    Y = matrix(df, "y", pool.models).astype(np.int8)
    C = realised_cost(df, pool)
    mean_nout = matrix(df.iloc[train_idx], "nout", pool.models).mean(axis=0)
    CH = expected_cost(df, pool, mean_nout)
    return pool, Y, C, CH


def load_responses(models: list[str]) -> dict[int, np.ndarray]:
    r = pd.read_parquet(DATA_PROCESSED / "sprout_responses.parquet", columns=["key"] + models)
    return {j: r[m].fillna("").to_numpy() for j, m in enumerate(models)}
