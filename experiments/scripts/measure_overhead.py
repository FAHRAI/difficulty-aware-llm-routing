"""Routing overhead per request on CPU: text representation, difficulty or success prediction and rule R1.
Plug-in costs use the recorded input token counts.

Outputs: results/sprout_P6/overhead.{json,md}
"""

import json
import platform
import subprocess
import time

import numpy as np
import torch

from darouter.data.loading import load_config, load_sprout, pool_arrays
from darouter.difficulty.embedding import embed, load_encoder
from darouter.difficulty.estimators import DAR
from darouter.paths import RESULTS
from darouter.routing.predictors import KNNRouter, PerModelLR
from darouter.routing.rules import select_r1

THREADS = 4
WARMUP, MEASURED, BATCH = 50, 500, 256
TAU = 0.8


def cpu_name() -> str:
    if platform.system() == "Darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    return platform.processor() or platform.machine()


def main() -> None:
    torch.set_num_threads(THREADS)
    cfg = load_config()
    df, fs = load_sprout()
    split = df.split.to_numpy()
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    _, Y, _, CH = pool_arrays(df, cfg, "P6", tr)
    prompts = df.prompt.to_numpy()
    encoder = load_encoder("cpu")

    dar = {name: DAR(name).fit(fs, tr, Y[tr]) for name in ("HEUR", "TFIDF", "EMB")}
    knn = KNNRouter(cfg["seed"]).fit(fs.emb[tr], Y[tr])
    lr = PerModelLR(cfg["seed"]).fit(fs.emb[tr], Y[tr])
    predictors = {f"DAR-{k}": (lambda texts, r=r: r.proba_text(texts, encoder)) for k, r in dar.items()}
    predictors["KNN-EMB"] = lambda texts: knn.proba(embed(texts, encoder, batch_size=len(texts)))
    predictors["LR-EMB"] = lambda texts: lr.proba(embed(texts, encoder, batch_size=len(texts)))

    sample = np.random.default_rng(1).choice(te, WARMUP + MEASURED, replace=False)
    timing = {"cpu": cpu_name(), "torch_threads": THREADS}
    for name, predict in predictors.items():

        def route(idx, predict=predict):
            return select_r1(predict([prompts[i] for i in idx]), CH[idx], TAU)[0]

        for i in sample[:WARMUP]:
            route([i])
        single = []
        for i in sample[WARMUP:]:
            start = time.perf_counter()
            route([i])
            single.append(time.perf_counter() - start)
        start = time.perf_counter()
        route(sample[:BATCH])
        timing[name] = {
            "single_median_ms": 1000 * float(np.median(single)),
            "single_p95_ms": 1000 * float(np.quantile(single, 0.95)),
            f"batch{BATCH}_ms_per_request": 1000 * (time.perf_counter() - start) / BATCH,
        }

    out = RESULTS / "sprout_P6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "overhead.json").write_text(json.dumps(timing, indent=1))
    lines = [
        f"# Routing overhead ({timing['cpu']}, CPU, {THREADS} threads)\n",
        f"| router | single request, median ms | single request, p95 ms | batch {BATCH}, ms per request |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {v['single_median_ms']:.2f} | {v['single_p95_ms']:.2f} | {v[f'batch{BATCH}_ms_per_request']:.3f} |"
        for k, v in timing.items()
        if isinstance(v, dict)
    ]
    text = "\n".join(lines) + "\n"
    (out / "overhead.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
