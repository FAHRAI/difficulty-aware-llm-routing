"""Bootstrap summaries, tables and the quality–cost figure for a finished run.

  python experiments/scripts/report.py --run sprout_P6

Outputs in results/<run>: report.json, tables.md, fig_quality_cost.png
"""

import argparse
import json
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from darouter.metrics.report import Inference, fmt_ci, mixture_frontier, short, to_json  # noqa: E402
from darouter.paths import RESULTS  # noqa: E402

EPS = 0.02
TOLERANCE_ROWS = [
    "DAR-EMB/R1",
    "DAR-EMB/R2",
    "DAR-TFIDF/R1",
    "DAR-HEUR/R1",
    "LR-EMB/R1",
    "LR-EMB/R2",
    "KNN-EMB/R1",
    "KNN-EMB/R2",
    "BEST-MIX",
    "CASCADE-2",
    "CASCADE-3",
    "STATIC-CAT/R1",
    "DAR-CAT/R1",
    "DAR-EMB+CAT/R1",
    "DAR-EMB+CAT/R2",
    "LR-EMB+CAT/R1",
]
PAIRS = [("DAR-EMB/R1", o) for o in ("LR-EMB/R1", "KNN-EMB/R1", "BEST-MIX", "DAR-EMB/R2", "CASCADE-2", "CASCADE-3")] + [
    ("DAR-EMB/R2", "LR-EMB/R2"),
    ("DAR-EMB/R2", "BEST-MIX"),
    ("DAR-EMB+CAT/R1", "LR-EMB+CAT/R1"),
    ("DAR-EMB+CAT/R2", "LR-EMB+CAT/R2"),
    ("DAR-EMB/R1", "DAR-EMB[failfrac]/R1"),
]
DOMAIN_ROWS = ["DAR-EMB/R1", "DAR-EMB/R2", "LR-EMB/R1", "LR-EMB/R2", "BEST-MIX", "CASCADE-2"]
SHARE_ROWS = ["DAR-EMB/R1", "DAR-EMB/R2", "LR-EMB/R1", "LR-EMB/R2", "KNN-EMB/R1", "STATIC-CAT/R1", "ORACLE"]
CURVES = {
    "DAR-EMB/R1": ("C0", "-"),
    "DAR-EMB/R2": ("C0", ":"),
    "LR-EMB/R2": ("C1", ":"),
    "LR-EMB/R1": ("C1", "-"),
    "KNN-EMB/R2": ("C2", ":"),
    "CASCADE-2": ("C3", "-"),
}


def build_report(res, B: int) -> dict:
    domains = res.extra["test_domain"]
    inf = Inference(res, domains, B=B)
    eps_all = sorted(res.target_acc)
    fixed = [f"FIXED:{m}" for m in res.models]
    routers = sorted({m for m, _ in res.points if not m.startswith("FIXED:")})
    report = {
        "models": res.models,
        "strong": res.models[res.strong],
        "cheap": res.models[res.cheap],
        "target_acc_val": {str(k): v for k, v in res.target_acc.items()},
        "fit_seconds": res.fit_seconds,
        "extra": {k: v for k, v in res.extra.items() if k not in ("test_domain", "test_keys")},
        "summaries": [inf.summary(m, EPS) for m in fixed] + [inf.summary(m, e) for m in routers for e in eps_all],
        "paired": [],
        "formal": {},
        "per_domain": {},
    }
    for e in eps_all:
        report["formal"][str(e)] = inf.formal_family("DAR-EMB/R1", e)
        report["paired"] += [inf.paired(a, b, e) for a, b in PAIRS if (a, e) in res.outcomes and (b, e) in res.outcomes]
    report["per_domain"] = {m: inf.per_domain(m, EPS, domains) for m in DOMAIN_ROWS if (m, EPS) in res.outcomes}
    return report


def tables(res, report: dict, run: str) -> str:
    S = {(s["method"], s["eps"]): s for s in report["summaries"]}
    eps_all = sorted(res.target_acc)
    fixed = [f"FIXED:{m}" for m in res.models]
    routers = sorted({m for m, _ in res.points if not m.startswith("FIXED:") and m != "ORACLE"})
    lines = [
        f"# {run}\n",
        f"STRONG = {short(report['strong'])}, CHEAP = {short(report['cheap'])}; "
        f"test n = {len(res.extra['test_domain'])}\n",
        f"## Main table, eps = {EPS}\n",
        "| method | acc [95% CI] | cost $/1k q [CI] | cost reduction vs STRONG, % [CI] | margin, pp [CI] | "
        "one-sided LB | fallback | feasible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in fixed + routers + ["ORACLE"]:
        s = S.get((m, EPS))
        if s is None:
            continue
        fb = "" if s["fallback_share"] is None else f"{100 * s['fallback_share']:.1f}%"
        lines.append(
            f"| {short(m)} | {fmt_ci(s['acc'], *s['acc_ci'])} | {fmt_ci(s['cost_per_1k'], *s['cost_ci'], nd=4)} "
            f"| {fmt_ci(s['cost_reduction'], *s['cost_reduction_ci'], pct=True)} "
            f"| {fmt_ci(s['margin'], *s['margin_ci'], pct=True)} | {100 * s['margin_lower_one_sided_95']:.2f} "
            f"| {fb} | {s['feasible']} |"
        )
    lines += [
        "\n## Across tolerances (cost reduction %, margin pp)\n",
        "| method | " + " | ".join(f"eps={e}" for e in eps_all) + " |",
        "|---|" + "---|" * len(eps_all),
    ]
    for m in TOLERANCE_ROWS:
        if (m, eps_all[0]) in S:
            cells = [f"{100 * S[(m, e)]['cost_reduction']:.1f} / {100 * S[(m, e)]['margin']:+.2f}" for e in eps_all]
            lines.append(f"| {short(m)} | " + " | ".join(cells) + " |")
    lines += ["\n## Hypotheses for DAR-EMB/R1 (raw / Holm-adjusted p)\n"]
    for e, f in report["formal"].items():
        lines.append(f"- eps={e}: " + ", ".join(f"{k}: {f['raw_p'][k]:.4f} / {f['holm_p'][k]:.4f}" for k in f["raw_p"]))
    lines += [
        "\n## Paired contrasts (first minus second)\n",
        "| pair | eps | Δacc pp [CI] | Δcost $/1k [CI] | rel. cost |",
        "|---|---|---|---|---|",
    ]
    for p in report["paired"]:
        lines.append(
            f"| {p['pair']} | {p['eps']} | {fmt_ci(p['d_acc'], *p['d_acc_ci'], pct=True)} | "
            f"{fmt_ci(p['d_cost_per_1k'], *p['d_cost_ci'], nd=4)} | {100 * p['rel_cost']:+.1f}% |"
        )
    lines += [f"\n## Per domain at eps = {EPS}\n"]
    for m, rows in report["per_domain"].items():
        lines += [
            f"\n**{m}**\n",
            "| domain | n | acc | STRONG acc | margin pp | cost $/1k | STRONG $/1k | reduction % |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['domain']} | {r['n']} | {r['acc']:.3f} | {r['strong_acc']:.3f} | {100 * r['margin']:+.2f} | "
            f"{r['cost_per_1k']:.4f} | {r['strong_cost_per_1k']:.4f} | {100 * r['cost_reduction']:.1f} |"
            for r in rows
        ]
    lines += [
        f"\n## Routing shares at eps = {EPS}\n",
        "| method | " + " | ".join(short(m) for m in res.models) + " |",
        "|---|" + "---|" * len(res.models),
    ]
    for m in SHARE_ROWS:
        s = S.get((m, EPS))
        if s and s["shares"]:
            lines.append(f"| {m} | " + " | ".join(f"{100 * x:.1f}" for x in s["shares"]) + " |")
    return "\n".join(lines) + "\n"


def plot_curves(res, report: dict, path) -> None:
    S = {(s["method"], s["eps"]): s for s in report["summaries"]}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    points = []
    for m in res.models:
        s = S[(f"FIXED:{m}", EPS)]
        points.append((s["cost_per_1k"], 100 * s["acc"]))
        ax.scatter(*points[-1], marker="s", color="black", zorder=5, s=22)
        ax.annotate(short(m), points[-1], fontsize=7, xytext=(4, -9), textcoords="offset points")
    ax.plot(*zip(*mixture_frontier(points)), color="grey", ls="--", lw=1, label="best query-independent mixture")
    for name, (col, ls) in CURVES.items():
        if name in res.curves:
            _, test_acc, test_cost, _, _ = res.curves[name]
            o = np.argsort(test_cost)
            ax.plot(1000 * test_cost[o], 100 * test_acc[o], color=col, ls=ls, lw=1.3, label=name)
        s = S.get((name, EPS))
        if s:
            ax.scatter(s["cost_per_1k"], 100 * s["acc"], color=col, marker="o", edgecolor="black", zorder=6, s=36)
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per 1,000 queries, USD (log scale)")
    ax.set_ylabel("Success rate on test, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="sprout_P6", help="directory name under results/")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run_dir = RESULTS / args.run
    with open(run_dir / "result.pkl", "rb") as f:
        res = pickle.load(f)
    report = build_report(res, args.bootstrap)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, default=to_json))
    text = tables(res, report, args.run)
    (run_dir / "tables.md").write_text(text)
    plot_curves(res, report, run_dir / "fig_quality_cost.png")
    print(text[:4000])


if __name__ == "__main__":
    main()
