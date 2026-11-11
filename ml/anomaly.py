"""Rider behaviour anomaly detection (Isolation Forest).

Each rider is summarised by a few behaviour features computed from their
history *before* a given moment (`as_of`), so we never peek at the future:

  idle_ratio               idle minutes / total delivery minutes
  long_stop_rate           share of deliveries with a 10+ minute idle stop
  median_speed_kmh         typical speed while moving
  claims_per_week          how often they file claims
  claim_to_delivery_ratio  claims / deliveries

An Isolation Forest learns what "normal" riders look like. Its raw score is
turned into a percentile among training riders, so the output reads simply:
0.97 = "more unusual than 97% of riders". Riders with too little history get
a neutral 0.5 instead of a noisy guess.

Note: unusual is not the same as dishonest. This is one signal among several,
and it is never enough on its own to reject a claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RIDER_FEATURES = ["idle_ratio", "long_stop_rate", "median_speed_kmh", "claims_per_week", "claim_to_delivery_ratio"]
LONG_STOP_MINUTES = 10.0
MIN_DELIVERIES = 10


def rider_feature_table(frame: pd.DataFrame, claims: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Behaviour features per rider using only deliveries and claims before `as_of`."""
    d = frame[frame["picked_up_at"] < as_of]
    c = claims[claims["submitted_at"] < as_of]
    g = d.groupby("rider_id")
    feats = pd.DataFrame({
        "n_deliveries": g.size(),
        "idle_ratio": g["idle_minutes"].sum() / g["actual_minutes"].sum(),
        "long_stop_rate": g["idle_minutes"].apply(lambda s: float((s >= LONG_STOP_MINUTES).mean())),
        "median_speed_kmh": g["avg_speed_kmh"].median(),
        "first_seen": g["picked_up_at"].min(),
    })
    n_claims = c.groupby("rider_id").size().reindex(feats.index, fill_value=0)
    weeks = ((pd.Timestamp(as_of) - feats["first_seen"]).dt.total_seconds() / (7 * 86400)).clip(lower=1.0)
    feats["n_claims"] = n_claims
    feats["claims_per_week"] = n_claims / weeks
    feats["claim_to_delivery_ratio"] = n_claims / feats["n_deliveries"]
    return feats.drop(columns="first_seen")


@dataclass
class RiderAnomalyModel:
    forest: IsolationForest | None = None
    train_scores: np.ndarray | None = None      # sorted raw scores of training riders
    medians: pd.Series | None = None            # for explanations
    iqrs: pd.Series | None = None

    def fit(self, feats: pd.DataFrame) -> RiderAnomalyModel:
        X = feats.loc[feats["n_deliveries"] >= MIN_DELIVERIES, RIDER_FEATURES]
        self.forest = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=1)
        self.forest.fit(X)
        self.train_scores = np.sort(self._raw(X))
        self.medians = X.median()
        q75, q25 = X.quantile(0.75), X.quantile(0.25)
        self.iqrs = (q75 - q25).replace(0, 1e-6)
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        # sklearn: lower score_samples = more abnormal. Flip so higher = more abnormal.
        return -self.forest.score_samples(X[RIDER_FEATURES])

    def score(self, feats: pd.DataFrame) -> pd.Series:
        """Anomaly percentile (0-1) per rider; 0.5 when history is too short."""
        out = pd.Series(0.5, index=feats.index, name="rider_anomaly")
        ok = feats["n_deliveries"] >= MIN_DELIVERIES
        if ok.any():
            raw = self._raw(feats.loc[ok])
            out.loc[ok] = np.searchsorted(self.train_scores, raw, side="right") / len(self.train_scores)
        return out

    def explain(self, row: pd.Series, top: int = 2) -> list[str]:
        """Plain-English list of the rider's most unusual features."""
        z = ((row[RIDER_FEATURES] - self.medians) / self.iqrs).astype(float)
        notes = []
        for feat in z.abs().sort_values(ascending=False).index[:top]:
            if abs(z[feat]) < 1.5:
                break
            direction = "higher" if z[feat] > 0 else "lower"
            notes.append(f"{feat.replace('_', ' ')} {row[feat]:.2f} is much {direction} than typical ({self.medians[feat]:.2f})")
        return notes

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> RiderAnomalyModel:
        return joblib.load(path)
