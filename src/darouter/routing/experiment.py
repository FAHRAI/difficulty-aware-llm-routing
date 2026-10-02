"""One routing experiment on a train/validation/test partition.

Every method is reduced to per-query test outcomes (acc_q, cost_q) at each validation-selected operating point,
plus a full test sweep for descriptive curves. Bootstrap/inference happens in metrics.report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from darouter.difficulty.estimators import DAR, FeatureStore
from darouter.metrics.evaluation import outcome_of_choice, outcome_of_mix, pick_operating_point
from darouter.routing.baselines import Cascade, StaticCategory, best_mix_weights
from darouter.routing.predictors import KNNRouter, PerModelLR
from darouter.routing.rules import LAMBDA_GRID, TAU_GRID, select_r1, select_r2

TEXT_DAR = ("HEUR", "TFIDF", "EMB")
META_DAR = ("CAT", "EMB+CAT")


@dataclass
class Partition:
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray


@dataclass
class Result:
    models: list[str]
    strong: int
    cheap: int
    target_acc: dict = field(default_factory=dict)  # eps -> A* on validation
    points: dict = field(default_factory=dict)  # (method, eps) -> summary dict
    outcomes: dict = field(default_factory=dict)  # (method, eps) -> (acc_q, cost_q) on test
    curves: dict = field(default_factory=dict)  # method -> (knobs, test_acc, test_cost, val_acc, val_cost)
    probas: dict = field(default_factory=dict)  # method -> p̂ on test (n_te, M)
    difficulty: dict = field(default_factory=dict)  # method -> d̂ on test
    fit_seconds: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _sweep_r1(p, ch, Y, C):
    acc, cost, fb = [], [], []
    for t in TAU_GRID:
        choice, fallback = select_r1(p, ch, t)
        a, c = outcome_of_choice(choice, Y, C)
        acc.append(a.mean()), cost.append(c.mean()), fb.append(fallback.mean())
    return np.array(acc), np.array(cost)


def _sweep_r2(p, ch, Y, C, cbar):
    acc, cost = [], []
    for lam in LAMBDA_GRID:
        a, c = outcome_of_choice(select_r2(p, ch, lam, cbar), Y, C)
        acc.append(a.mean()), cost.append(c.mean())
    return np.array(acc), np.array(cost)


def run(
    fs: FeatureStore,
    Y: np.ndarray,
    C: np.ndarray,
    CH: np.ndarray,
    part: Partition,
    models: list[str],
    eps_grid,
    responses: dict | None = None,
    prompts: list[str] | None = None,
    mid_model: str | None = None,
    methods: str = "all",
    seed: int = 20261001,
) -> Result:
    tr, va, te = part.tr, part.va, part.te
    Ytr, Yva, Yte = Y[tr], Y[va], Y[te]
    Cva, Cte = C[va], C[te]
    CHva, CHte = CH[va], CH[te]
    strong = int(np.argmax(Yva.mean(axis=0)))
    cost_va = Cva.mean(axis=0)
    cheap = int(sorted(range(len(models)), key=lambda j: (cost_va[j], -Yva[:, j].mean()))[0])
    res = Result(models, strong, cheap)
    for eps in eps_grid:
        res.target_acc[eps] = (1 - eps) * Yva[:, strong].mean()

    def record(
        name, eps, knob, acc_q, cost_q, val_acc, val_cost, choice=None, fallback=None, ch_te=None, feasible=True
    ):
        info = {
            "knob": None if knob is None else float(knob),
            "feasible": feasible,
            "val_acc": float(val_acc),
            "val_cost": float(val_cost),
            "test_acc": float(acc_q.mean()),
            "test_cost": float(cost_q.mean()),
        }
        if choice is not None:
            info["shares"] = np.bincount(choice, minlength=len(models)).astype(float).tolist()
            info["shares"] = [s / len(choice) for s in info["shares"]]
        if fallback is not None:
            info["fallback_share"] = float(fallback.mean())
            info["fallback_acc"] = float(acc_q[fallback].mean()) if fallback.any() else None
        if ch_te is not None and choice is not None:
            planned = ch_te[np.arange(len(choice)), choice].mean()
            info["plugin_cost_rel_error"] = float(planned / cost_q.mean() - 1)
        res.points[(name, eps)] = info
        res.outcomes[(name, eps)] = (acc_q, cost_q)

    # Fixed models, uniform random, oracle
    for j, m in enumerate(models):
        a, c = Yte[:, j].astype(float), Cte[:, j]
        for eps in eps_grid:
            record(f"FIXED:{m}", eps, None, a, c, Yva[:, j].mean(), Cva[:, j].mean())
    w_u = np.full(len(models), 1 / len(models))
    for eps in eps_grid:
        a, c = outcome_of_mix(w_u, Yte, Cte)
        record("RANDOM-UNIFORM", eps, None, a, c, (Yva @ w_u).mean(), (Cva @ w_u).mean())
    big = np.where(Yte == 1, Cte, np.inf)
    oracle_choice = np.where(Yte.any(axis=1), big.argmin(axis=1), Cte.argmin(axis=1))
    a, c = outcome_of_choice(oracle_choice, Yte, Cte)
    for eps in eps_grid:
        record("ORACLE", eps, None, a, c, np.nan, np.nan, choice=oracle_choice)

    # BEST-MIX (LP on validation means)
    for eps in eps_grid:
        w = best_mix_weights(Yva.mean(axis=0), cost_va, res.target_acc[eps])
        feasible = w is not None
        if not feasible:
            w = np.eye(len(models))[strong]
        a, c = outcome_of_mix(w, Yte, Cte)
        record("BEST-MIX", eps, None, a, c, (Yva @ w).mean(), (Cva @ w).mean(), feasible=feasible)
        res.points[("BEST-MIX", eps)]["weights"] = w.tolist()

    # Probability-based routers
    cbar = CH[tr][:, strong].mean()
    p_methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add_dar(est, target="rasch", name=None):
        t0 = time.time()
        dar = DAR(est, target=target, seed=seed).fit(fs, tr, Ytr)
        name = name or f"DAR-{est}"
        res.fit_seconds[name] = time.time() - t0
        d_va, d_te = dar.difficulty(fs, va), dar.difficulty(fs, te)
        p_methods[name] = (dar.proba_from_d(d_va), dar.proba_from_d(d_te))
        res.difficulty[name] = d_te
        res.extra[f"{name}:calibration"] = dar.calibration_params().tolist()
        if target == "rasch":
            res.extra[f"{name}:theta"] = dar.theta_.tolist()
        return dar

    if methods in ("all", "text"):
        for est in TEXT_DAR:
            add_dar(est)
        add_dar("EMB", target="failfrac", name="DAR-EMB[failfrac]")
    if methods == "all":
        for est in META_DAR:
            add_dar(est)

    t0 = time.time()
    knn = KNNRouter(seed).fit(fs.emb[tr], Ytr)
    res.fit_seconds["KNN-EMB"] = time.time() - t0
    p_methods["KNN-EMB"] = (knn.proba(fs.emb[va]), knn.proba(fs.emb[te]))
    res.extra["KNN-EMB:k"] = knn.k_
    t0 = time.time()
    lr = PerModelLR(seed).fit(fs.emb[tr], Ytr)
    res.fit_seconds["LR-EMB"] = time.time() - t0
    p_methods["LR-EMB"] = (lr.proba(fs.emb[va]), lr.proba(fs.emb[te]))
    res.extra["LR-EMB:C"] = lr.C_
    if methods == "all":
        Xtr = np.hstack([fs.emb[tr], fs.onehot(tr)])
        t0 = time.time()
        lrc = PerModelLR(seed).fit(Xtr, Ytr)
        res.fit_seconds["LR-EMB+CAT"] = time.time() - t0
        p_methods["LR-EMB+CAT"] = (
            lrc.proba(np.hstack([fs.emb[va], fs.onehot(va)])),
            lrc.proba(np.hstack([fs.emb[te], fs.onehot(te)])),
        )
        sc = StaticCategory().fit(fs.category[tr], Ytr)
        p_methods["STATIC-CAT"] = (sc.proba(fs.category[va]), sc.proba(fs.category[te]))

    for name, (p_va, p_te) in p_methods.items():
        res.probas[name] = p_te
        va_acc, va_cost = _sweep_r1(p_va, CHva, Yva, Cva)
        te_acc, te_cost = _sweep_r1(p_te, CHte, Yte, Cte)
        res.curves[f"{name}/R1"] = (TAU_GRID, te_acc, te_cost, va_acc, va_cost)
        for eps in eps_grid:
            tau = pick_operating_point(list(TAU_GRID), va_acc, va_cost, res.target_acc[eps])
            if tau is None:
                a, c = Yte[:, strong].astype(float), Cte[:, strong]
                record(f"{name}/R1", eps, None, a, c, np.nan, np.nan, feasible=False)
                continue
            choice, fb = select_r1(p_te, CHte, tau)
            a, c = outcome_of_choice(choice, Yte, Cte)
            k = list(TAU_GRID).index(tau)
            record(f"{name}/R1", eps, tau, a, c, va_acc[k], va_cost[k], choice, fb, CHte)
        if name == "STATIC-CAT":
            continue
        va_acc, va_cost = _sweep_r2(p_va, CHva, Yva, Cva, cbar)
        te_acc, te_cost = _sweep_r2(p_te, CHte, Yte, Cte, cbar)
        res.curves[f"{name}/R2"] = (LAMBDA_GRID, te_acc, te_cost, va_acc, va_cost)
        for eps in eps_grid:
            lam = pick_operating_point(list(LAMBDA_GRID), va_acc, va_cost, res.target_acc[eps])
            if lam is None:
                a, c = Yte[:, strong].astype(float), Cte[:, strong]
                record(f"{name}/R2", eps, None, a, c, np.nan, np.nan, feasible=False)
                continue
            choice = select_r2(p_te, CHte, lam, cbar)
            a, c = outcome_of_choice(choice, Yte, Cte)
            k = list(LAMBDA_GRID).index(lam)
            record(f"{name}/R2", eps, lam, a, c, va_acc[k], va_cost[k], choice, None, CHte)

    # Cascades
    if responses is not None and methods == "all":
        mid = models.index(mid_model) if mid_model in models else None
        chains = {"CASCADE-2": [cheap, strong]}
        if mid is not None and mid not in (cheap, strong):
            chains["CASCADE-3"] = [cheap, mid, strong]
        ptr = [prompts[i] for i in tr]
        pva = [prompts[i] for i in va]
        pte = [prompts[i] for i in te]
        for name, chain in chains.items():
            t0 = time.time()
            cas = Cascade(chain).fit(ptr, {m: [responses[m][i] for i in tr] for m in chain[:-1]}, Ytr)
            res.fit_seconds[name] = time.time() - t0
            g_va = cas.scores(pva, {m: [responses[m][i] for i in va] for m in chain[:-1]})
            g_te = cas.scores(pte, {m: [responses[m][i] for i in te] for m in chain[:-1]})
            sweep_va = [cas.outcome(g_va, t, Yva, Cva) for t in TAU_GRID]
            sweep_te = [cas.outcome(g_te, t, Yte, Cte) for t in TAU_GRID]
            va_acc = np.array([a.mean() for a, _ in sweep_va])
            va_cost = np.array([c.mean() for _, c in sweep_va])
            res.curves[name] = (
                TAU_GRID,
                np.array([a.mean() for a, _ in sweep_te]),
                np.array([c.mean() for _, c in sweep_te]),
                va_acc,
                va_cost,
            )
            for eps in eps_grid:
                t = pick_operating_point(list(TAU_GRID), va_acc, va_cost, res.target_acc[eps])
                if t is None:
                    a, c = Yte[:, strong].astype(float), Cte[:, strong]
                    record(name, eps, None, a, c, np.nan, np.nan, feasible=False)
                    continue
                k = list(TAU_GRID).index(t)
                a, c = sweep_te[k]
                record(name, eps, t, a, c, va_acc[k], va_cost[k])
            res.extra[f"{name}:chain"] = [models[m] for m in chain]
    return res
