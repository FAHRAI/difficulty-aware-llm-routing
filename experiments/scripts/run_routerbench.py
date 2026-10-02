"""Routing experiment on RouterBench (0-shot release), 60/20/20 split stratified by task group.

Outputs: data/processed/rb_*.npy, data/processed/routerbench_table.parquet, results/routerbench/result.pkl
"""

import pickle
import time

import numpy as np
import pandas as pd

from darouter.data.loading import load_config
from darouter.data.routerbench import to_table
from darouter.difficulty.embedding import embed
from darouter.difficulty.estimators import FeatureStore
from darouter.difficulty.features import head_tail, heuristic_matrix
from darouter.paths import DATA_PROCESSED, DATA_RAW, RESULTS
from darouter.routing.experiment import Partition, run


def main() -> None:
    cfg = load_config()
    raw = pd.read_pickle(DATA_RAW / cfg["datasets"]["routerbench"]["file"])
    df, models, responses = to_table(raw, cfg["seed"], cfg["success_threshold"])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    emb_path, heur_path = DATA_PROCESSED / "rb_emb_minilm.npy", DATA_PROCESSED / "rb_heur.npy"
    cached = emb_path.exists() and heur_path.exists() and len(np.load(emb_path, mmap_mode="r")) == len(df)
    if not cached:
        np.save(heur_path, heuristic_matrix(df.prompt))
        np.save(emb_path, embed(df.prompt))
    fs = FeatureStore([head_tail(t) for t in df.prompt], np.load(heur_path), np.load(emb_path), df.category.to_numpy())

    split = df.split.to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "validation", "test"))
    Y = df[[f"y__{m}" for m in models]].to_numpy().astype(np.int8)
    C = df[[f"cost__{m}" for m in models]].to_numpy()
    CH = np.repeat(C[tr].mean(axis=0)[None, :], len(df), axis=0)  # no token counts: mean training cost per model

    start = time.time()
    res = run(
        fs,
        Y,
        C,
        CH,
        Partition(tr, va, te),
        models,
        cfg["targets_eps"],
        responses={j: responses[m] for j, m in enumerate(models)},
        prompts=df.prompt.tolist(),
        mid_model=None,
        seed=cfg["seed"],
    )
    res.extra["test_domain"] = df.domain.to_numpy()[te]
    out = RESULTS / "routerbench"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.pkl", "wb") as f:
        pickle.dump(res, f)
    df.drop(columns=["prompt"]).to_parquet(DATA_PROCESSED / "routerbench_table.parquet", index=False)
    print(
        f"done in {time.time() - start:.0f} s; strong={models[res.strong]} cheap={models[res.cheap]}; "
        f"split {len(tr)}/{len(va)}/{len(te)}"
    )


if __name__ == "__main__":
    main()
