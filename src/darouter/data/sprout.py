"""Load the SPROUT response matrix into compact per-query tables.

Raw parquet files (pinned HF revision) -> one row per query with, for every model m:
score_m (judge score), y_m (binarised success), nin_m / nout_m (token counts).
Response texts are kept in a separate table; they are needed only by the cascade baseline and the exact-match check.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

META = ["key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"]
MODEL_PREFIXES = ("aws-", "openai-", "wxai-")


def domain_of(dataset: str) -> str:
    if "MATH" in dataset:
        return "math"
    if "MMLU" in dataset:
        return "mmlu_pro"
    if "gpqa" in dataset:
        return "gpqa"
    if "MuSR" in dataset:
        return "musr"
    if "ragbench" in dataset:
        return "rag"
    if "openhermes" in dataset:
        return "openhermes"
    raise ValueError(dataset)


def category_of(dataset: str, level: str | None) -> str:
    """Source benchmark (sub)category; MATH is split by its official difficulty level."""
    if "MATH" in dataset:
        return f"math/{level}"
    return dataset.split("/", 1)[0] if "openhermes" in dataset else dataset


def load_raw(raw_dir: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "sprout_*.parquet"))):
        split = os.path.basename(f).split("sprout_")[1].split("-")[0]
        parts.append(pd.read_parquet(f).assign(split=split))
    return pd.concat(parts, ignore_index=True)


def model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(MODEL_PREFIXES)]


def to_tables(raw: pd.DataFrame, success_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = model_columns(raw)
    out = raw[META + ["split"]].copy()
    out["domain"] = raw["dataset"].map(domain_of)
    out["category"] = [category_of(d, l) for d, l in zip(raw["dataset"], raw["dataset_level"])]
    responses = raw[["key"]].copy()
    for m in models:
        cell = raw[m]
        out[f"score__{m}"] = cell.map(lambda x: x["score"]).astype(float)
        out[f"y__{m}"] = (out[f"score__{m}"] >= success_threshold).astype(np.int8)
        out[f"nin__{m}"] = cell.map(lambda x: x["num_input_tokens"]).astype(np.int32)
        out[f"nout__{m}"] = cell.map(lambda x: x["num_output_tokens"]).astype(np.int32)
        responses[m] = cell.map(lambda x: x["response"])
    order = {"train": 0, "validation": 1, "test": 2}
    out = out.sort_values(["split", "key"], key=lambda s: s.map(order) if s.name == "split" else s)
    out = out.reset_index(drop=True)
    responses = responses.set_index("key").loc[out["key"]].reset_index()
    return out, responses
