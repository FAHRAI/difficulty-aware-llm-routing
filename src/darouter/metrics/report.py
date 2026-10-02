"""Bootstrap summaries, paired contrasts and per-domain breakdowns of an experiment result."""

from __future__ import annotations

import numpy as np

from darouter.metrics.evaluation import boot_means, ci, holm, p_greater_than_zero, stratified_bootstrap_indices


def short(m: str) -> str:
    return (
        m.replace("wxai-", "")
        .replace("aws-", "")
        .replace("openai-", "")
        .replace("-instruct", "")
        .replace("-8k-max-tokens", "")
        .replace("-v01", "")
        .replace("-v1", "")
    )


class Inference:
    def __init__(self, res, strata: np.ndarray, B: int = 2000, seed: int = 20261001):
        self.res = res
        self.idx = stratified_bootstrap_indices(strata, B, seed)
        self.strong = f"FIXED:{res.models[res.strong]}"

    def summary(self, method: str, eps: float) -> dict:
        a, c = self.res.outcomes[(method, eps)]
        sa, sc = self.res.outcomes[(self.strong, eps)]
        ba, bc = boot_means(a, self.idx), boot_means(c, self.idx)
        bsa, bsc = boot_means(sa, self.idx), boot_means(sc, self.idx)
        red = 1 - bc / bsc
        margin = ba - (1 - eps) * bsa
        pt = self.res.points[(method, eps)]
        return {
            "method": method,
            "eps": eps,
            "feasible": pt["feasible"],
            "knob": pt["knob"],
            "acc": a.mean(),
            "acc_ci": ci(ba),
            "cost_per_1k": 1000 * c.mean(),
            "cost_ci": tuple(1000 * x for x in ci(bc)),
            "cost_reduction": 1 - c.mean() / sc.mean(),
            "cost_reduction_ci": ci(red),
            "margin": a.mean() - (1 - eps) * sa.mean(),
            "margin_ci": ci(margin),
            "margin_lower_one_sided_95": float(np.quantile(margin, 0.05)),
            "boundary_crossing_freq": float((margin < 0).mean()),
            "fallback_share": pt.get("fallback_share"),
            "fallback_acc": pt.get("fallback_acc"),
            "shares": pt.get("shares"),
            "plugin_cost_rel_error": pt.get("plugin_cost_rel_error"),
            "val_acc": pt["val_acc"],
            "val_cost_per_1k": 1000 * pt["val_cost"],
        }

    def paired(self, m1: str, m2: str, eps: float) -> dict:
        a1, c1 = self.res.outcomes[(m1, eps)]
        a2, c2 = self.res.outcomes[(m2, eps)]
        da, dc = boot_means(a1 - a2, self.idx), boot_means(c1 - c2, self.idx)
        return {
            "pair": f"{m1} - {m2}",
            "eps": eps,
            "d_acc": (a1 - a2).mean(),
            "d_acc_ci": ci(da),
            "d_cost_per_1k": 1000 * (c1 - c2).mean(),
            "d_cost_ci": tuple(1000 * x for x in ci(dc)),
            "rel_cost": c1.mean() / c2.mean() - 1,
        }

    def formal_family(self, method: str, eps: float) -> dict:
        a, c = self.res.outcomes[(method, eps)]
        sa, sc = self.res.outcomes[(self.strong, eps)]
        ma, mc = self.res.outcomes[("BEST-MIX", eps)]
        margin = boot_means(a - (1 - eps) * sa, self.idx)
        p = {
            "H1_quality_retained": p_greater_than_zero(margin),
            "H2_cheaper_than_strong": p_greater_than_zero(boot_means(sc - c, self.idx)),
            "H3_cheaper_than_best_mix": p_greater_than_zero(boot_means(mc - c, self.idx)),
        }
        return {"raw_p": p, "holm_p": holm(p)}

    def per_domain(self, method: str, eps: float, domains: np.ndarray) -> list[dict]:
        a, c = self.res.outcomes[(method, eps)]
        sa, sc = self.res.outcomes[(self.strong, eps)]
        rows = []
        for d in sorted(set(domains)):
            m = domains == d
            rows.append(
                {
                    "domain": d,
                    "n": int(m.sum()),
                    "acc": a[m].mean(),
                    "strong_acc": sa[m].mean(),
                    "margin": a[m].mean() - (1 - eps) * sa[m].mean(),
                    "cost_per_1k": 1000 * c[m].mean(),
                    "strong_cost_per_1k": 1000 * sc[m].mean(),
                    "cost_reduction": 1 - c[m].mean() / sc[m].mean(),
                }
            )
        return rows


def fmt_ci(x, lo, hi, pct=False, nd=3):
    if pct:
        return f"{100 * x:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def mixture_frontier(points: list[tuple[float, float]], steps: int = 50) -> list[tuple[float, float]]:
    """Upper concave hull of (cost, accuracy) points: the best random mixture of fixed models.

    Mixtures are linear in cost, so each hull segment is densified for plotting on a log-cost axis.
    """
    hull: list[tuple[float, float]] = []
    for p in sorted(points):
        while len(hull) >= 2 and (
            (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0])
        ):
            hull.pop()
        if not hull or p[1] > hull[-1][1]:
            hull.append(p)
    return [
        (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        for (x0, y0), (x1, y1) in zip(hull, hull[1:])
        for t in np.linspace(0, 1, steps)
    ]


def to_json(o):
    """json.dump default for numpy scalars and arrays."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
