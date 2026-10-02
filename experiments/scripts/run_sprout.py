"""Routing experiment on SPROUT.

  python experiments/scripts/run_sprout.py               # pool P6, official splits
  python experiments/scripts/run_sprout.py --pool P13    # all 13 models
  python experiments/scripts/run_sprout.py --lodo        # leave-one-domain-out

Outputs: results/sprout_<pool>[_lodo]/result*.pkl
"""

import argparse
import pickle
import time

import numpy as np

from darouter.data.loading import load_config, load_responses, load_sprout, pool_arrays
from darouter.paths import RESULTS
from darouter.routing.experiment import Partition, run

MID_MODEL = "wxai-llama-3-1-8b-instruct"  # middle stage of the three-model cascade


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="P6")
    parser.add_argument("--lodo", action="store_true", help="leave each domain out of training and tuning")
    args = parser.parse_args()

    cfg = load_config()
    df, fs = load_sprout()
    split = df.split.to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "validation", "test"))
    out = RESULTS / (f"sprout_{args.pool}" + ("_lodo" if args.lodo else ""))
    out.mkdir(parents=True, exist_ok=True)

    if not args.lodo:
        pool, Y, C, CH = pool_arrays(df, cfg, args.pool, tr)
        start = time.time()
        res = run(
            fs,
            Y,
            C,
            CH,
            Partition(tr, va, te),
            pool.models,
            cfg["targets_eps"],
            responses=load_responses(pool.models),
            prompts=df.prompt.tolist(),
            mid_model=MID_MODEL,
            seed=cfg["seed"],
        )
        res.extra["test_domain"] = df.domain.to_numpy()[te]
        res.extra["test_keys"] = df.key.to_numpy()[te]
        with open(out / "result.pkl", "wb") as f:
            pickle.dump(res, f)
        print(f"done in {time.time() - start:.0f} s; strong={pool.models[res.strong]} cheap={pool.models[res.cheap]}")
        return

    domain = df.domain.to_numpy()
    for d in sorted(set(domain)):
        keep_tr, keep_va, te_d = tr[domain[tr] != d], va[domain[va] != d], te[domain[te] == d]
        pool, Y, C, CH = pool_arrays(df, cfg, args.pool, keep_tr)
        start = time.time()
        res = run(
            fs,
            Y,
            C,
            CH,
            Partition(keep_tr, keep_va, te_d),
            pool.models,
            cfg["targets_eps"],
            methods="text",
            seed=cfg["seed"],
        )
        res.extra["test_domain"] = domain[te_d]
        with open(out / f"result_{d}.pkl", "wb") as f:
            pickle.dump(res, f)
        print(f"held out {d}: n_test={len(te_d)}, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
