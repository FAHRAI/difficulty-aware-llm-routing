"""Adding a model to the pool with few labelled requests.

Each pool model in turn is treated as new. DAR-EMB reuses the difficulty estimator fitted on the other models and
fits only the two link parameters of the new model; the alternative trains a logistic regression on the embedding.
Both use the same n labelled training requests (20 random draws per n) and are scored on the test split.

Outputs: results/sprout_P6/new_model.{json,md}
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from darouter.data.loading import load_config, load_sprout, pool_arrays
from darouter.difficulty.estimators import DAR
from darouter.metrics.evaluation import auroc
from darouter.metrics.report import short
from darouter.paths import RESULTS
from darouter.routing.predictors import PerModelLR

SAMPLE_SIZES = (50, 100, 200, 500, 1000)
REPEATS = 20


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate_new_model(fs, tr, te, Y, j: int) -> dict:
    others = [k for k in range(Y.shape[1]) if k != j]
    dar = DAR("EMB").fit(fs, tr, Y[tr][:, others])
    d_train, d_test = dar.oof_, dar.difficulty(fs, te)
    y_train, y_test = Y[tr, j], Y[te, j]
    rows = {}
    for n in SAMPLE_SIZES:
        scores = {"DAR": {"logloss": [], "auroc": []}, "LR": {"logloss": [], "auroc": []}}
        for r in range(REPEATS):
            pick = np.random.default_rng(1000 * n + r).choice(len(tr), n, replace=False)
            if y_train[pick].min() == y_train[pick].max():
                continue
            p_dar = LogisticRegression(C=1e6).fit(d_train[pick, None], y_train[pick]).predict_proba(d_test[:, None])
            p_lr = (
                LogisticRegression(C=1.0, max_iter=2000).fit(fs.emb[tr][pick], y_train[pick]).predict_proba(fs.emb[te])
            )
            for name, p in (("DAR", p_dar[:, 1]), ("LR", p_lr[:, 1])):
                scores[name]["logloss"].append(log_loss(p, y_test))
                scores[name]["auroc"].append(auroc(p, y_test))
        rows[n] = {
            name: {
                f"{metric}_{stat}": float(fn(values))
                for metric, values in s.items()
                for stat, fn in (("mean", np.mean), ("sd", np.std))
            }
            | {"draws": len(s["logloss"])}
            for name, s in scores.items()
        }
    full = PerModelLR().fit(fs.emb[tr], Y[tr][:, [j]]).proba(fs.emb[te])[:, 0]
    rows["full_train_LR_auroc"] = auroc(full, y_test)
    return rows


def main() -> None:
    cfg = load_config()
    df, fs = load_sprout()
    split = df.split.to_numpy()
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    pool, Y, _, _ = pool_arrays(df, cfg, "P6", tr)
    result = {m: evaluate_new_model(fs, tr, te, Y, j) for j, m in enumerate(pool.models)}

    out = RESULTS / "sprout_P6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "new_model.json").write_text(json.dumps(result, indent=1))
    lines = [
        f"# New model calibrated from n labelled requests (test log-loss / AUROC, mean of {REPEATS} draws)\n",
        "| new model | " + " | ".join(f"n={n}: DAR vs LR" for n in SAMPLE_SIZES) + " | LR, full train AUROC |",
        "|---|" + "---|" * (len(SAMPLE_SIZES) + 1),
    ]
    for m, rows in result.items():
        cells = [
            f"{rows[n]['DAR']['logloss_mean']:.3f}/{rows[n]['DAR']['auroc_mean']:.3f} vs "
            f"{rows[n]['LR']['logloss_mean']:.3f}/{rows[n]['LR']['auroc_mean']:.3f}"
            for n in SAMPLE_SIZES
        ]
        lines.append(f"| {short(m)} | " + " | ".join(cells) + f" | {rows['full_train_LR_auroc']:.3f} |")
    text = "\n".join(lines) + "\n"
    (out / "new_model.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
