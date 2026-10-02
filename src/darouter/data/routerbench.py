"""RouterBench 0-shot (Hu et al., 2024) converted to the per-query table layout used for SPROUT."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

N_MODELS = 11


def task_group(name: str) -> str:
    if name.startswith("mmlu"):
        return "mmlu"
    if name.lower().startswith("chinese"):
        return "chinese"
    if name.startswith("mtbench"):
        return "mtbench"
    return {
        "hellaswag": "hellaswag",
        "grade-school-math": "gsm8k",
        "arc-challenge": "arc",
        "winogrande": "winogrande",
        "mbpp": "mbpp",
    }.get(name, "other")


def to_table(raw: pd.DataFrame, seed: int, success_threshold: float = 0.5):
    models = list(raw.columns[3 : 3 + N_MODELS])
    out = pd.DataFrame(
        {
            "key": raw.sample_id.astype(str),
            "prompt": raw.prompt.astype(str),
            "category": raw.eval_name,
            "domain": raw.eval_name.map(task_group),
        }
    )
    responses = {}
    for m in models:
        out[f"score__{m}"] = raw[m].astype(float)
        out[f"y__{m}"] = (out[f"score__{m}"] >= success_threshold).astype(np.int8)
        out[f"cost__{m}"] = raw[f"{m}|total_cost"].astype(float)
        responses[m] = raw[f"{m}|model_response"].astype(str).to_numpy()
    idx = np.arange(len(out))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=seed, stratify=out.domain)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=out.domain.iloc[rest])
    split = np.empty(len(out), dtype=object)
    split[tr], split[va], split[te] = "train", "validation", "test"
    out["split"] = split
    return out, models, responses
