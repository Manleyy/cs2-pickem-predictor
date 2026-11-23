"""Match model: logistic regression on team-minus-opponent features.

Each match is added twice (A beats B with y=1, and B vs A with y=0) and there is
no intercept, so P(A beats B) = 1 - P(B beats A) by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .features import elo_expect


def symmetrize(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feats].to_numpy(float)
    return np.vstack([X, -X]), np.r_[np.ones(len(X)), np.zeros(len(X))]


@dataclass
class MatchModel:
    features: list[str]
    coef: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, C: float = 1.0) -> "MatchModel":
        X, y = symmetrize(df, self.features)
        self.scale = list(np.where(X.std(0) > 0, X.std(0), 1.0))
        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=1000)
        lr.fit(X / np.array(self.scale), y)
        self.coef = list(lr.coef_[0])
        return self

    def logit(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return (X / np.array(self.scale)) @ np.array(self.coef)

    def predict(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2, default=float))

    @classmethod
    def load(cls, path: Path) -> "MatchModel":
        return cls(**json.loads(Path(path).read_text()))

    def describe(self) -> str:
        lines = ["feature        coef/std  (per 1 unit of raw feature)"]
        for f, c, s in zip(self.features, self.coef, self.scale):
            lines.append(f"  {f:<12} {c:+.3f}    {c / s:+.3f}")
        return "\n".join(lines)


def scores(y_true, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(log_loss=round(log_loss(y_true, p, labels=[0, 1]), 4),
                brier=round(brier_score_loss(y_true, p), 4),
                accuracy=round(float(((p > 0.5) == (y_true == 1)).mean()), 4),
                n=int(len(y_true) // 2))


def evaluate(df: pd.DataFrame, test_from, feature_sets: dict[str, list[str]], C=1.0) -> pd.DataFrame:
    """Time split: fit before `test_from`, score after. Includes a no-fit
    baseline using Valve's own Elo formula on VRS points."""
    tr = df[df.date < pd.Timestamp(test_from)]
    te = df[df.date >= pd.Timestamp(test_from)]
    rows = []
    _, y = symmetrize(te, ["vrs_diff"])
    p0 = elo_expect(te.vrs_diff.to_numpy() * 100)
    rows.append(dict(model="VRS points, Valve formula (no fit)", **scores(y, np.r_[p0, 1 - p0])))
    for name, feats in feature_sets.items():
        ok_tr, ok_te = tr.dropna(subset=feats), te.dropna(subset=feats)
        if len(ok_tr) < 200 or len(ok_te) < 50:
            continue
        m = MatchModel(feats).fit(ok_tr, C)
        Xte, yte = symmetrize(ok_te, feats)
        rows.append(dict(model=name, **scores(yte, m.predict(Xte))))
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, model: MatchModel, bins=10) -> pd.DataFrame:
    X, y = symmetrize(df.dropna(subset=model.features), model.features)
    p = model.predict(X)
    b = pd.cut(p, np.linspace(0, 1, bins + 1))
    return pd.DataFrame(dict(p=p, y=y, bin=b)).groupby("bin", observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3)


# --------------------------------------------------------- series formats ---
def bo3_from_map(p):
    p = np.asarray(p, float)
    return p * p * (3 - 2 * p)


def map_from_bo3(q, iters=60):
    """Invert q = p^2(3-2p) by bisection (monotone on [0,1])."""
    q = np.asarray(q, float)
    lo, hi = np.zeros_like(q), np.ones_like(q)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_high = bo3_from_map(mid) > q
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return (lo + hi) / 2


def stretch(p, factor: float):
    """Scale log-odds: factor > 1 makes favourites stronger."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 + np.exp(-factor * np.log(p / (1 - p))))
