"""Rule R1 with a fixed fallback model (the strongest or the cheapest) instead of the most probable model.

Requires results/sprout_P6/result.pkl. Outputs: results/sprout_P6/fixed_fallback.{json,md}
"""

import json
import pickle

import numpy as np

from darouter.data.loading import load_config, load_sprout, pool_arrays
from darouter.difficulty.estimators import DAR
from darouter.metrics.evaluation import outcome_of_choice, pick_operating_point
from darouter.metrics.report import Inference, to_json
from darouter.paths import RESULTS
from darouter.routing.predictors import KNNRouter, PerModelLR
from darouter.routing.rules import TAU_GRID, select_r1

LARGEST = "wxai-llama-3-405b-instruct"


def choose(p: np.ndarray, ch: np.ndarray, tau: float, fallback: int) -> tuple[np.ndarray, np.ndarray]:
    chosen, used_fallback = select_r1(p, ch, tau)
    return np.where(used_fallback, fallback, chosen), used_fallback


def main() -> None:
    cfg = load_config()
    df, fs = load_sprout()
    split = df.split.to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "validation", "test"))
    pool, Y, C, CH = pool_arrays(df, cfg, "P6", tr)
    run_dir = RESULTS / "sprout_P6"
    with open(run_dir / "result.pkl", "rb") as f:
        res = pickle.load(f)

    dar = DAR("EMB", seed=cfg["seed"]).fit(fs, tr, Y[tr])
    knn = KNNRouter(cfg["seed"]).fit(fs.emb[tr], Y[tr])
    lr = PerModelLR(cfg["seed"]).fit(fs.emb[tr], Y[tr])
    probas = {
        "DAR-EMB": (dar.proba(fs, va), dar.proba(fs, te)),
        "KNN-EMB": (knn.proba(fs.emb[va]), knn.proba(fs.emb[te])),
        "LR-EMB": (lr.proba(fs.emb[va]), lr.proba(fs.emb[te])),
    }

    added = []
    for name, (p_val, p_test) in probas.items():
        for label, fallback in (("STRONG", res.strong), ("CHEAP", res.cheap)):
            method = f"{name}/R1-fallback-{label}"
            sweep = [outcome_of_choice(choose(p_val, CH[va], t, fallback)[0], Y[va], C[va]) for t in TAU_GRID]
            val_acc = np.array([a.mean() for a, _ in sweep])
            val_cost = np.array([c.mean() for _, c in sweep])
            for eps, target in res.target_acc.items():
                tau = pick_operating_point(list(TAU_GRID), val_acc, val_cost, target)
                if tau is None:  # target unreachable: the strongest model is used instead
                    chosen, used = np.full(len(te), res.strong), np.zeros(len(te), bool)
                    chosen_val_acc, chosen_val_cost = Y[va, res.strong].mean(), C[va, res.strong].mean()
                else:
                    chosen, used = choose(p_test, CH[te], tau, fallback)
                    k = list(TAU_GRID).index(tau)
                    chosen_val_acc, chosen_val_cost = val_acc[k], val_cost[k]
                acc, cost = outcome_of_choice(chosen, Y[te], C[te])
                res.points[(method, eps)] = {
                    "knob": tau,
                    "feasible": tau is not None,
                    "val_acc": chosen_val_acc,
                    "val_cost": chosen_val_cost,
                    "shares": (np.bincount(chosen, minlength=len(pool.models)) / len(chosen)).tolist(),
                    "fallback_share": float(used.mean()),
                    "fallback_acc": float(acc[used].mean()) if used.any() else None,
                }
                res.outcomes[(method, eps)] = (acc, cost)
                added.append((method, eps))

    inf = Inference(res, res.extra["test_domain"], B=cfg["bootstrap"], seed=cfg["seed"])
    largest = pool.models.index(LARGEST)
    rows = [inf.summary(m, eps) for m, eps in added]
    lines = [
        "# Rule R1 with a fixed fallback model\n",
        "| method | eps | acc | cost reduction % [CI] | margin pp [CI] | fallback | share of 405b % | feasible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in rows:
        (r_lo, r_hi), (m_lo, m_hi) = s["cost_reduction_ci"], s["margin_ci"]
        lines.append(
            f"| {s['method']} | {s['eps']} | {s['acc']:.3f} | {100 * s['cost_reduction']:.1f} "
            f"[{100 * r_lo:.1f}, {100 * r_hi:.1f}] | {100 * s['margin']:+.2f} [{100 * m_lo:+.2f}, {100 * m_hi:+.2f}] "
            f"| {100 * s['fallback_share']:.1f} | {100 * s['shares'][largest]:.1f} | {s['feasible']} |"
        )
    (run_dir / "fixed_fallback.json").write_text(json.dumps(rows, indent=1, default=to_json))
    text = "\n".join(lines) + "\n"
    (run_dir / "fixed_fallback.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
