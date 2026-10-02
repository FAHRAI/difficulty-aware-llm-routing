"""Figures from finished runs: results/figures/{scheme,quality_cost,domains}.png

Requires results/sprout_P6 and results/routerbench with result.pkl and report.json (see report.py).
"""

import json
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from darouter.metrics.report import mixture_frontier  # noqa: E402
from darouter.paths import RESULTS  # noqa: E402

EPS = 0.02
OUT = RESULTS / "figures"
CURVES = {
    "DAR-EMB/R1": ("black", "-", "DAR-EMB, R1"),
    "DAR-EMB/R2": ("black", ":", "DAR-EMB, R2"),
    "LR-EMB/R2": ("tab:blue", "--", "LR-EMB, R2"),
    "KNN-EMB/R2": ("tab:green", "-.", "KNN-EMB, R2"),
    "CASCADE-2": ("tab:red", "-", "Cascade (2 models)"),
}
DOMAIN_BARS = [("DAR-EMB/R1", "black"), ("DAR-EMB/R2", "dimgrey"), ("LR-EMB/R2", "tab:blue"), ("BEST-MIX", "lightgrey")]


def scheme(path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fill, ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="black"))

    ax.text(1, 44, "Offline (historical response matrix)", fontsize=9, style="italic")
    box(1, 31, 20, 9, "Responses $Y_{im}$\nof $|M|$ models\n(train split)", "#eeeeee")
    box(27, 31, 19, 9, "Rasch fit:\nabilities $\\theta_m$,\ndifficulties $b_i$")
    box(52, 31, 20, 9, "Estimator\n$\\hat d(q)$ ← features\n(HEUR / TF-IDF / EMB)")
    box(78, 31, 21, 9, "Calibration\n$\\hat p_m=\\sigma(\\alpha_m+\\beta_m\\hat d)$\n(cross-fitted)")
    for x0 in (21.5, 46.5, 72.5):
        arrow(x0, 35.5, x0 + 5, 35.5)
    ax.text(1, 22, "Online (per query)", fontsize=9, style="italic")
    box(1, 6, 15, 10, "Query $q$", "#eeeeee")
    box(21, 6, 17, 10, "Difficulty\n$\\hat d(q)$")
    box(43, 6, 19, 10, "Success estimates\n$\\hat p_m(q)$, cost $\\hat C_m(q)$")
    box(67, 6, 15, 10, "Rule R1 / R2\n(threshold $\\tau$\nor weight $\\lambda$)")
    box(87, 6, 12, 10, "Selected\nmodel $m^*$", "#eeeeee")
    for x0 in (16.5, 38.5, 62.5, 82.5):
        arrow(x0, 11, x0 + 4, 11)
    arrow(62, 30.5, 30, 16.5)
    arrow(88, 30.5, 53, 16.5)
    ax.text(
        99,
        22,
        "validation split: choose $\\tau$ or $\\lambda$\nfor the target $(1-\\varepsilon)A_{strong}$",
        fontsize=8,
        ha="right",
        va="center",
        bbox={"fc": "white", "ec": "none", "pad": 1},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def load_run(run: str):
    with open(RESULTS / run / "result.pkl", "rb") as f:
        res = pickle.load(f)
    report = json.loads((RESULTS / run / "report.json").read_text())
    return res, {(s["method"], s["eps"]): s for s in report["summaries"]}, report


def quality_cost(path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    panels = [("sprout_P6", "a) SPROUT, open-weight pool", 60), ("routerbench", "b) RouterBench", 50)]
    for ax, (run, title, ymin) in zip(axes, panels):
        res, S, _ = load_run(run)
        points = [(S[(f"FIXED:{m}", EPS)]["cost_per_1k"], 100 * S[(f"FIXED:{m}", EPS)]["acc"]) for m in res.models]
        ax.scatter(*zip(*points), marker="s", color="grey", s=12, zorder=4)
        ax.plot(*zip(*mixture_frontier(points)), color="grey", lw=1, ls=(0, (4, 2)), label="best fixed mixture")
        for name, (color, style, label) in CURVES.items():
            if name not in res.curves:
                continue
            _, test_acc, test_cost, _, _ = res.curves[name]
            order = np.argsort(test_cost)
            ax.plot(1000 * test_cost[order], 100 * test_acc[order], color=color, ls=style, lw=1.1, label=label)
            s = S[(name, EPS)]
            ax.scatter(s["cost_per_1k"], 100 * s["acc"], color=color, edgecolor="black", s=22, zorder=6)
        strong = S[(f"FIXED:{res.models[res.strong]}", EPS)]
        ax.axhline(100 * (1 - EPS) * strong["acc"], color="grey", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Cost per 1,000 queries, USD (log)")
        ax.set_ylim(ymin, 88)
    axes[0].set_ylabel("Success rate on test, %")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def domains(path) -> None:
    _, _, report = load_run("sprout_P6")
    per_domain = report["per_domain"]
    labels = [f"{r['domain']}\n(n={r['n']})" for r in per_domain["DAR-EMB/R1"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for k, (method, color) in enumerate(DOMAIN_BARS):
        rows = per_domain[method]
        offset = x + (k - 1.5) * 0.2
        axes[0].bar(
            offset, [100 * r["cost_reduction"] for r in rows], 0.2, color=color, edgecolor="black", lw=0.4, label=method
        )
        axes[1].bar(offset, [100 * r["margin"] for r in rows], 0.2, color=color, edgecolor="black", lw=0.4)
    for ax, ylabel in zip(axes, ["Cost reduction vs. strongest, %", "Quality margin, p.p."]):
        ax.set_xticks(x, labels, fontsize=7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(-130, 130)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "mathtext.fontset": "stix",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    scheme(OUT / "scheme.png")
    quality_cost(OUT / "quality_cost.png")
    domains(OUT / "domains.png")
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
