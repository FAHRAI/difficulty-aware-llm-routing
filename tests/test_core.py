import numpy as np
from scipy.special import expit

from darouter.difficulty.features import HEUR_NAMES, head_tail, heuristic_features
from darouter.difficulty.irt import fit_rasch, item_difficulty
from darouter.metrics.evaluation import (
    holm,
    outcome_of_choice,
    outcome_of_mix,
    pick_operating_point,
    stratified_bootstrap_indices,
)
from darouter.routing.baselines import Cascade, StaticCategory, best_mix_weights
from darouter.routing.rules import select_r1, select_r2


def test_rasch_recovers_ordering():
    rng = np.random.default_rng(0)
    theta = np.array([-1.5, -0.5, 0.0, 0.7, 1.5, 2.5])
    b = rng.normal(0, 1.5, 4000)
    Y = (rng.random((4000, 6)) < expit(theta[None] - b[:, None])).astype(int)
    th_hat, b_hat = fit_rasch(Y)
    assert np.all(np.diff(th_hat) > 0)
    assert np.corrcoef(b, b_hat)[0, 1] > 0.75


def test_item_difficulty_depends_only_on_success_count():
    theta = np.array([-1.0, 0.0, 1.0, 2.0])
    Y = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    b = item_difficulty(Y, theta)
    assert np.allclose(b[0], b[1]) and np.allclose(b[0], b[2])
    assert b[3] < b[0] < b[4]


def test_head_tail_and_features():
    long = "A" * 1500 + "QUESTION?" + "B" * 1500
    ht = head_tail(long)
    assert len(ht) == 2003 and ht.startswith("A") and ht.endswith("B")
    assert head_tail("short") == "short"
    assert len(heuristic_features("What is 2+2?\n(A) 3\n(B) 4")) == len(HEUR_NAMES)


def test_select_r1_cheapest_qualifying_and_fallback():
    p = np.array([[0.9, 0.95, 0.99], [0.2, 0.6, 0.7], [0.1, 0.2, 0.3]])
    ch = np.array([[1.0, 2.0, 3.0]] * 3)
    choice, fb = select_r1(p, ch, tau=0.5)
    assert choice.tolist() == [0, 1, 2] and fb.tolist() == [False, False, True]
    choice, fb = select_r1(p, ch, tau=0.0)
    assert choice.tolist() == [0, 0, 0]


def test_select_r1_respects_cost_not_index():
    p = np.array([[0.9, 0.9]])
    ch = np.array([[5.0, 1.0]])
    assert select_r1(p, ch, 0.5)[0].tolist() == [1]


def test_select_r2_tradeoff():
    p = np.array([[0.6, 0.9]])
    ch = np.array([[1.0, 10.0]])
    assert select_r2(p, ch, lam=0.0, cbar=10.0).tolist() == [1]
    assert select_r2(p, ch, lam=1.0, cbar=10.0).tolist() == [0]


def test_best_mix_lp():
    acc, cost = np.array([0.6, 0.7, 0.9]), np.array([1.0, 5.0, 10.0])
    w = best_mix_weights(acc, cost, 0.75)
    assert abs(w @ acc - 0.75) < 1e-6 and abs(w.sum() - 1) < 1e-9
    assert w @ cost <= 1.0 * 0.5 + 10.0 * 0.5 + 1e-9  # at least as cheap as the cheap/strong line
    assert best_mix_weights(acc, cost, 0.95) is None


def test_cascade_cost_accumulates():
    Y = np.array([[1, 1], [0, 1]])
    C = np.array([[1.0, 10.0], [1.0, 10.0]])
    c = Cascade([0, 1])
    acc, cost = c.outcome(np.array([[0.9], [0.1]]), 0.5, Y, C)
    assert acc.tolist() == [1, 1] and cost.tolist() == [1.0, 11.0]


def test_static_category_shrinkage():
    cats = np.array(["a"] * 2 + ["b"] * 100)
    Y = np.vstack([np.ones((2, 1)), np.zeros((100, 1))])
    sc = StaticCategory(k=20).fit(cats, Y)
    assert 0 < sc.proba(np.array(["a"]))[0, 0] < 0.2
    assert np.allclose(sc.proba(np.array(["unseen"])), Y.mean())


def test_outcomes_and_operating_point():
    Y = np.array([[0, 1], [1, 1]])
    C = np.array([[1.0, 4.0], [1.0, 4.0]])
    acc, cost = outcome_of_choice(np.array([1, 0]), Y, C)
    assert acc.tolist() == [1, 1] and cost.tolist() == [4, 1]
    acc, cost = outcome_of_mix(np.array([0.5, 0.5]), Y, C)
    assert np.allclose(acc, [0.5, 1.0]) and np.allclose(cost, [2.5, 2.5])
    assert pick_operating_point([0.1, 0.2, 0.3], [0.8, 0.9, 0.95], [1, 2, 3], 0.9) == 0.2
    assert pick_operating_point([0.1], [0.5], [1], 0.9) is None


def test_bootstrap_preserves_strata_and_holm():
    strata = np.array(["x"] * 3 + ["y"] * 7)
    idx = stratified_bootstrap_indices(strata, 50, 1)
    assert all((strata[row] == "x").sum() == 3 for row in idx)
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_mixture_frontier_is_upper_hull():
    from darouter.metrics.report import mixture_frontier

    points = [(1.0, 60.0), (2.0, 62.0), (5.0, 80.0), (10.0, 85.0), (20.0, 81.0)]
    frontier = mixture_frontier(points, steps=2)
    xs = [x for x, _ in frontier]
    assert (2.0, 62.0) not in frontier and 20.0 not in xs
    assert frontier[0] == (1.0, 60.0) and frontier[-1] == (10.0, 85.0)


def test_config_prices_cover_pools():
    from darouter.data.loading import load_config

    cfg = load_config()
    for name, models in cfg["pools"].items():
        if models != "all":
            assert set(models) <= set(cfg["prices"]), name
    assert 0 < cfg["success_threshold"] < 1
