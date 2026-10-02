"""Difficulty estimators d̂(q) and the cross-fitted difficulty-aware router (DAR).

Estimators work on row indices into a shared FeatureStore so that representations are computed once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold

from darouter.difficulty.embedding import embed
from darouter.difficulty.features import head_tail, heuristic_matrix
from darouter.difficulty.irt import fit_rasch

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


@dataclass
class FeatureStore:
    text: list[str]  # head+tail router input
    heur: np.ndarray  # (N, 16)
    emb: np.ndarray  # (N, 384)
    category: np.ndarray  # (N,) str
    categories: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.categories = sorted(set(self.category))

    def onehot(self, idx: np.ndarray) -> np.ndarray:
        pos = {c: j for j, c in enumerate(self.categories)}
        out = np.zeros((len(idx), len(self.categories)))
        out[np.arange(len(idx)), [pos[c] for c in self.category[idx]]] = 1.0
        return out


class Estimator:
    uses_metadata = False

    def fit(self, fs: FeatureStore, idx: np.ndarray, y: np.ndarray) -> Estimator:
        raise NotImplementedError

    def predict(self, fs: FeatureStore, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_text(self, prompts, encoder=None) -> np.ndarray:
        """Difficulty of raw prompts, computing the representation from text (used in deployment)."""
        raise NotImplementedError(f"{type(self).__name__} needs metadata, not only text")


class Heur(Estimator):
    def fit(self, fs, idx, y):
        self.m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
        self.m.fit(fs.heur[idx], y)
        return self

    def predict(self, fs, idx):
        return self.m.predict(fs.heur[idx])

    def predict_text(self, prompts, encoder=None):
        return self.m.predict(heuristic_matrix(prompts))


class Tfidf(Estimator):
    """Alpha chosen on an inner 20 % hold-out of the training portion (GCV would need a dense n x n matrix)."""

    def fit(self, fs, idx, y):
        self.v = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, min_df=2)
        X = self.v.fit_transform([fs.text[i] for i in idx])
        rng = np.random.default_rng(0)
        hold = rng.random(len(idx)) < 0.2
        err = {
            a: np.mean((Ridge(alpha=a).fit(X[~hold], y[~hold]).predict(X[hold]) - y[hold]) ** 2) for a in RIDGE_ALPHAS
        }
        self.alpha_ = min(err, key=err.get)
        self.m = Ridge(alpha=self.alpha_).fit(X, y)
        return self

    def predict(self, fs, idx):
        return self.m.predict(self.v.transform([fs.text[i] for i in idx]))

    def predict_text(self, prompts, encoder=None):
        return self.m.predict(self.v.transform([head_tail(t) for t in prompts]))


class Emb(Estimator):
    def fit(self, fs, idx, y):
        self.m = RidgeCV(alphas=RIDGE_ALPHAS).fit(fs.emb[idx], y)
        return self

    def predict(self, fs, idx):
        return self.m.predict(fs.emb[idx])

    def predict_text(self, prompts, encoder=None):
        return self.m.predict(embed(prompts, encoder, batch_size=max(len(prompts), 1)))


class Cat(Estimator):
    uses_metadata = True

    def fit(self, fs, idx, y):
        cats = fs.category[idx]
        self.mean = {c: y[cats == c].mean() for c in set(cats)}
        self.default = float(y.mean())
        return self

    def predict(self, fs, idx):
        return np.array([self.mean.get(c, self.default) for c in fs.category[idx]])


class EmbCat(Estimator):
    uses_metadata = True

    def fit(self, fs, idx, y):
        self.m = RidgeCV(alphas=RIDGE_ALPHAS).fit(np.hstack([fs.emb[idx], fs.onehot(idx)]), y)
        return self

    def predict(self, fs, idx):
        return self.m.predict(np.hstack([fs.emb[idx], fs.onehot(idx)]))


ESTIMATORS = {"HEUR": Heur, "TFIDF": Tfidf, "EMB": Emb, "CAT": Cat, "EMB+CAT": EmbCat}


def difficulty_target(Y: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray | None]:
    """'rasch' -> (b, theta); 'failfrac' -> (1 - s/|M|, None)."""
    if kind == "rasch":
        theta, b = fit_rasch(Y)
        return b, theta
    if kind == "failfrac":
        return 1.0 - Y.mean(axis=1), None
    raise ValueError(kind)


class DAR:
    """Difficulty-aware success model: p̂_m(q) = sigmoid(alpha_m + beta_m * d̂(q)), cross-fitted on train."""

    def __init__(self, estimator: str, target: str = "rasch", n_folds: int = 5, seed: int = 20261001):
        self.estimator, self.target, self.n_folds, self.seed = estimator, target, n_folds, seed

    def fit(self, fs: FeatureStore, idx: np.ndarray, Y: np.ndarray) -> DAR:
        oof = np.empty(len(idx))
        for tr, ho in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(idx):
            d_tr, _ = difficulty_target(Y[tr], self.target)
            est = ESTIMATORS[self.estimator]().fit(fs, idx[tr], d_tr)
            oof[ho] = est.predict(fs, idx[ho])
        self.oof_ = oof
        self.calib_ = [LogisticRegression(C=1e6).fit(oof[:, None], Y[:, j]) for j in range(Y.shape[1])]
        d_all, self.theta_ = difficulty_target(Y, self.target)
        self.d_train_ = d_all
        self.est_ = ESTIMATORS[self.estimator]().fit(fs, idx, d_all)
        return self

    def difficulty(self, fs: FeatureStore, idx: np.ndarray) -> np.ndarray:
        return self.est_.predict(fs, idx)

    def proba_from_d(self, d: np.ndarray) -> np.ndarray:
        return np.column_stack([c.predict_proba(d[:, None])[:, 1] for c in self.calib_])

    def proba(self, fs: FeatureStore, idx: np.ndarray) -> np.ndarray:
        return self.proba_from_d(self.difficulty(fs, idx))

    def proba_text(self, prompts, encoder=None) -> np.ndarray:
        return self.proba_from_d(self.est_.predict_text(prompts, encoder))

    def calibration_params(self) -> np.ndarray:
        return np.array([[c.intercept_[0], c.coef_[0, 0]] for c in self.calib_])
