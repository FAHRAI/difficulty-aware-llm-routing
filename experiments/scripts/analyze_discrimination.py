"""Diagnostics of a finished SPROUT run: discrimination of the success estimates, correlation of estimated and
outcome-derived difficulty, routing behaviour, outcome sensitivities and a duplicate-prompt audit.

  python experiments/scripts/analyze_discrimination.py --run sprout_P6

Outputs in results/<run>: discrimination.json, discrimination.md
"""

import argparse
import json
import pickle
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from darouter.data.loading import load_config, load_responses, pool_arrays
from darouter.difficulty.irt import item_difficulty
from darouter.metrics.evaluation import auroc, brier, reliability, stratified_bootstrap_indices
from darouter.metrics.report import short, to_json
from darouter.paths import DATA_PROCESSED, RESULTS
from darouter.routing.rules import select_r1, select_r2

EPS = 0.02
MID_MODEL = "wxai-llama-3-1-8b-instruct"
LARGEST, STRONGEST = "wxai-llama-3-405b-instruct", "wxai-llama-3-3-70b-instruct"
BEHAVIOUR_ROWS = ["DAR-EMB/R1", "DAR-EMB/R2", "LR-EMB/R1", "LR-EMB/R2", "KNN-EMB/R1", "DAR-TFIDF/R1", "DAR-HEUR/R1"]
SENSITIVITY_ROWS = ["DAR-EMB/R1", "DAR-EMB/R2", "LR-EMB/R1", "LR-EMB/R2", "KNN-EMB/R1", "DAR-TFIDF/R1"]
ANSWER_LETTER = re.compile(r"answer is \(?([A-J])\)?")


class Context:
    """Test-split arrays of a finished run, aligned with the stored success estimates."""

    def __init__(self, res, df: pd.DataFrame, cfg: dict, pool_name: str):
        split = df.split.to_numpy()
        self.tr, self.te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
        pool, Y, C, CH = pool_arrays(df, cfg, pool_name, self.tr)
        if pool.models != res.models:
            raise ValueError("model pool of the run does not match the configuration")
        self.res, self.df = res, df
        self.Y, self.C, self.CH = Y[self.te], C[self.te], CH[self.te]
        self.cbar = CH[self.tr][:, res.strong].mean()
        self.domain = df.domain.to_numpy()[self.te]
        self.boot = stratified_bootstrap_indices(self.domain, cfg["bootstrap"], cfg["seed"])

    def choice(self, method: str, eps: float = EPS) -> np.ndarray:
        """Model chosen for every test request by a method at its validation-selected operating point."""
        point = self.res.points[(method, eps)]
        if not point["feasible"]:
            return np.full(len(self.te), self.res.strong)
        base, rule = method.rsplit("/", 1)
        p = self.res.probas[base]
        if rule == "R1":
            return select_r1(p, self.CH, point["knob"])[0]
        return select_r2(p, self.CH, point["knob"], self.cbar)


def discrimination(ctx: Context) -> tuple[dict, list[str]]:
    res, Y, cheap = ctx.res, ctx.Y, ctx.res.cheap
    mid = res.models.index(MID_MODEL)
    domains = sorted(set(ctx.domain))
    out = {"auroc": {}, "auroc_paired_vs_DAR-EMB": {}}
    md = [
        "## AUROC for 'the cheapest model is sufficient' (test)\n",
        "| method | AUROC [95% CI] | AUROC llama-3-1-8b | " + " | ".join(domains) + " |",
        "|---|---|---|" + "---|" * len(domains),
    ]
    for name, p in res.probas.items():
        boot = [auroc(p[b, cheap], Y[b, cheap]) for b in ctx.boot]
        per_domain = {d: auroc(p[ctx.domain == d, cheap], Y[ctx.domain == d, cheap]) for d in domains}
        entry = {
            "cheap": auroc(p[:, cheap], Y[:, cheap]),
            "cheap_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "mid": auroc(p[:, mid], Y[:, mid]),
            "cheap_per_domain": per_domain,
            "brier_per_model": [brier(p[:, j], Y[:, j]) for j in range(len(res.models))],
            "reliability_pooled": reliability(p.ravel(), Y.ravel()),
        }
        out["auroc"][name] = entry
        lo, hi = entry["cheap_ci"]
        md.append(
            f"| {name} | {entry['cheap']:.3f} [{lo:.3f}, {hi:.3f}] | {entry['mid']:.3f} | "
            + " | ".join(f"{v:.3f}" for v in per_domain.values())
            + " |"
        )

    ref = res.probas["DAR-EMB"][:, cheap]
    md += ["\n## Paired AUROC difference vs DAR-EMB (stratified bootstrap)\n"]
    for name, p in res.probas.items():
        if name == "DAR-EMB":
            continue
        diffs = [auroc(p[b, cheap], Y[b, cheap]) - auroc(ref[b], Y[b, cheap]) for b in ctx.boot]
        point = auroc(p[:, cheap], Y[:, cheap]) - auroc(ref, Y[:, cheap])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        out["auroc_paired_vs_DAR-EMB"][name] = [point, lo, hi]
        md.append(f"- {name}: {point:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return out, md


def difficulty_correlation(ctx: Context) -> tuple[dict, list[str]]:
    res = ctx.res
    theta = np.array(res.extra["DAR-EMB:theta"])
    b_test = item_difficulty(ctx.Y, theta)
    out = {
        "spearman_vs_outcome_difficulty": {n: float(spearmanr(d, b_test).statistic) for n, d in res.difficulty.items()},
        "success_count_freq": np.bincount(ctx.Y.sum(axis=1), minlength=len(res.models) + 1).tolist(),
        "theta": dict(zip(res.models, theta.tolist())),
    }
    md = ["\n## Spearman correlation of estimated and outcome-derived difficulty (test)\n"]
    md += [f"- {n}: {v:.3f}" for n, v in out["spearman_vs_outcome_difficulty"].items()]
    md.append(f"\nNumber of pool models solving a request (0..{len(res.models)}): {out['success_count_freq']}")
    md.append("\nRasch abilities (train): " + ", ".join(f"{short(m)} {t:.2f}" for m, t in out["theta"].items()))
    return out, md


def routing_behaviour(ctx: Context) -> tuple[dict, list[str]]:
    res, Y, C = ctx.res, ctx.Y, ctx.C
    yc, ys = Y[:, res.cheap], Y[:, res.strong]
    largest, strongest = res.models.index(LARGEST), res.models.index(STRONGEST)
    counts = {
        "both_correct": int((yc & ys).sum()),
        "cheap_only": int((yc & (1 - ys)).sum()),
        "strong_only": int(((1 - yc) & ys).sum()),
        "both_wrong": int(((1 - yc) & (1 - ys)).sum()),
        "only_strong_solves": int(((Y.sum(axis=1) == 1) & (ys == 1)).sum()),
        "largest_solves_strongest_fails": int(((Y[:, largest] == 1) & (Y[:, strongest] == 0)).sum()),
        "only_largest_solves": int(((Y.sum(axis=1) == 1) & (Y[:, largest] == 1)).sum()),
    }
    rescue, cheap_ok, unsolved = (yc == 0) & (ys == 1), yc == 1, Y.sum(axis=1) == 0
    out = {
        "cheap_strong_counts": counts,
        "behaviour": {},
        "unsolved_share": float(unsolved.mean()),
        "unsolved_cost_share_strong": float(C[unsolved, res.strong].sum() / C[:, res.strong].sum()),
    }
    md = [
        "\n## Cheapest vs strongest model on test\n",
        ", ".join(f"{k}: {v}" for k, v in counts.items()),
        f"\n## Routing behaviour at eps = {EPS}\n",
        "| method | rescue requests solved % | cheap-sufficient kept on cheapest % | cost on unsolved requests % |",
        "|---|---|---|---|",
    ]
    for m in BEHAVIOUR_ROWS:
        if (m, EPS) not in res.points:
            continue
        chosen = ctx.choice(m)
        rows = np.arange(len(chosen))
        acc, cost = Y[rows, chosen], C[rows, chosen]
        b = {
            "rescue_solved": float(acc[rescue].mean()),
            "kept_on_cheap": float((chosen[cheap_ok] == res.cheap).mean()),
            "cost_share_unsolved": float(cost[unsolved].sum() / cost.sum()),
        }
        out["behaviour"][m] = b
        md.append(
            f"| {m} | {100 * b['rescue_solved']:.1f} | {100 * b['kept_on_cheap']:.1f} | "
            f"{100 * b['cost_share_unsolved']:.1f} |"
        )
    md.append(
        f"\nRequests no model solves: {100 * out['unsolved_share']:.1f} % of test; the strongest model spends "
        f"{100 * out['unsolved_cost_share_strong']:.1f} % of its cost on them."
    )
    return out, md


def exact_match(ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact-match correctness on MMLU-Pro test items (answer letter extracted by regular expression)."""
    mask = ctx.domain == "mmlu_pro"
    gold = ctx.df.golden_answer.to_numpy()[ctx.te][mask]
    responses = load_responses(ctx.res.models)
    em = np.zeros((mask.sum(), len(ctx.res.models)), dtype=int)
    extractable = np.zeros(len(ctx.res.models))
    for j in range(len(ctx.res.models)):
        letters = [(ANSWER_LETTER.findall(t or "") or [None])[-1] for t in responses[j][ctx.te][mask]]
        extractable[j] = np.mean([x is not None for x in letters])
        em[:, j] = [int(x is not None and x == g.strip()[0]) for x, g in zip(letters, gold)]
    return mask, em, extractable


def outcome_sensitivity(ctx: Context) -> tuple[dict, list[str]]:
    res = ctx.res
    score = ctx.df[[f"score__{m}" for m in res.models]].to_numpy()[ctx.te]
    mask, em, extractable = exact_match(ctx)
    judge = ctx.Y[mask]
    out = {
        "exact_match": {
            "extractable": dict(zip(res.models, extractable.tolist())),
            "agreement_with_judge": dict(zip(res.models, (em == judge).mean(axis=0).tolist())),
            "accuracy": dict(zip(res.models, em.mean(axis=0).tolist())),
        },
        "rescored": {},
    }
    strong_score, strong_em = score[:, res.strong].mean(), em[:, res.strong].mean()
    md = [
        f"\n## Policies at eps = {EPS} re-scored with other outcome definitions\n",
        "| method | mean judge score | strongest | exact match, MMLU-Pro | strongest |",
        "|---|---|---|---|---|",
    ]
    for m in SENSITIVITY_ROWS:
        if (m, EPS) not in res.points:
            continue
        chosen = ctx.choice(m)
        s, e = score[np.arange(len(chosen)), chosen].mean(), em[np.arange(mask.sum()), chosen[mask]].mean()
        out["rescored"][m] = {"judge_score": float(s), "exact_match_mmlu_pro": float(e)}
        md.append(f"| {m} | {s:.3f} | {strong_score:.3f} | {e:.3f} | {strong_em:.3f} |")
    w = np.array(res.points[("BEST-MIX", EPS)]["weights"])
    md.append(f"| BEST-MIX | {(score @ w).mean():.3f} | {strong_score:.3f} | {(em @ w).mean():.3f} | {strong_em:.3f} |")
    md.append(
        "\nJudge vs exact match on MMLU-Pro (agreement / extractable): "
        + ", ".join(
            f"{short(m)} {out['exact_match']['agreement_with_judge'][m]:.3f}/{out['exact_match']['extractable'][m]:.2f}"
            for m in res.models
        )
    )
    return out, md


def duplicate_audit(df: pd.DataFrame) -> tuple[dict, list[str]]:
    counts = df.prompt.value_counts()
    dups = df[df.prompt.isin(counts[counts > 1].index)]
    across = dups.groupby("prompt").split.nunique()
    train_prompts = set(df[df.split == "train"].prompt)
    out = {
        "rows_with_duplicate_prompt": int(len(dups)),
        "distinct_duplicated_prompts": int(len(across)),
        "prompts_across_splits": int((across > 1).sum()),
        "test_rows_with_prompt_in_train": int(df[df.split == "test"].prompt.isin(train_prompts).sum()),
    }
    return out, ["\n## Duplicate prompts\n", json.dumps(out)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="sprout_P6", help="directory name under results/")
    parser.add_argument("--pool", default="P6")
    args = parser.parse_args()
    run_dir = RESULTS / args.run
    with open(run_dir / "result.pkl", "rb") as f:
        res = pickle.load(f)
    cfg = load_config()
    df = pd.read_parquet(DATA_PROCESSED / "sprout.parquet")
    ctx = Context(res, df, cfg, args.pool)

    out, md = {}, [f"# Diagnostics — {args.run}\n"]
    for part in (
        discrimination(ctx),
        difficulty_correlation(ctx),
        routing_behaviour(ctx),
        outcome_sensitivity(ctx),
        duplicate_audit(df),
    ):
        out.update(part[0])
        md += part[1]
    (run_dir / "discrimination.json").write_text(json.dumps(out, indent=1, default=to_json))
    text = "\n".join(md) + "\n"
    (run_dir / "discrimination.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
