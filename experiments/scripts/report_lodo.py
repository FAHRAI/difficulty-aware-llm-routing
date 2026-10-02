"""Summary of the leave-one-domain-out runs: results/sprout_P6_lodo/lodo.md"""

import pickle

import numpy as np

from darouter.metrics.report import Inference, short
from darouter.paths import RESULTS

EPS = 0.02
METHODS = ["DAR-EMB/R1", "DAR-EMB/R2", "DAR-TFIDF/R2", "DAR-HEUR/R2", "LR-EMB/R2", "KNN-EMB/R2", "BEST-MIX"]


def main() -> None:
    run_dir = RESULTS / "sprout_P6_lodo"
    lines = [
        f"# Leave-one-domain-out, eps = {EPS} (held-out domain excluded from all fitting and tuning)\n",
        "| held-out domain | n | STRONG | STRONG acc | " + " | ".join(METHODS) + " |",
        "|---|---|---|---|" + "---|" * len(METHODS),
    ]
    pooled: dict[str, list] = {}
    for path in sorted(run_dir.glob("result_*.pkl")):
        with open(path, "rb") as f:
            res = pickle.load(f)
        inf = Inference(res, res.extra["test_domain"], B=1000)
        strong_acc, strong_cost = res.outcomes[(inf.strong, EPS)]
        cells = []
        for m in METHODS:
            s = inf.summary(m, EPS)
            lo, hi = s["margin_ci"]
            cells.append(
                f"{100 * s['cost_reduction']:.0f}% / {100 * s['margin']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            )
            acc, cost = res.outcomes[(m, EPS)]
            pooled.setdefault(m, []).append((acc, cost, strong_acc, strong_cost))
        lines.append(
            f"| {path.stem.removeprefix('result_')} | {len(strong_acc)} | {short(res.models[res.strong])} | "
            f"{strong_acc.mean():.3f} | " + " | ".join(cells) + " |"
        )
    lines += [
        "\nCells: cost reduction vs STRONG / margin in pp [95% CI].\n",
        "Pooled over all held-out test requests:\n",
    ]
    for m, parts in pooled.items():
        acc, cost, s_acc, s_cost = (np.concatenate([p[k] for p in parts]) for k in range(4))
        lines.append(
            f"- {m}: acc {acc.mean():.3f} vs STRONG {s_acc.mean():.3f}; "
            f"margin {100 * (acc.mean() - (1 - EPS) * s_acc.mean()):+.2f} pp; "
            f"cost reduction {100 * (1 - cost.mean() / s_cost.mean()):.1f}%"
        )
    text = "\n".join(lines) + "\n"
    (run_dir / "lodo.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
