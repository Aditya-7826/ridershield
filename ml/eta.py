"""ETA regression: how long *should* this delivery take?

The model predicts delivery time from distance, zone, hour, weekday, vehicle,
rain, traffic and disruption flags. We use it in two ways:

* predict(df)        expected time under the conditions that actually happened
* predict_clear(df)  expected time for the same trip on a normal day
                     (no rain, no flood/closure, the zone's usual traffic for
                     that hour)

From these we get, for any delivery:

  condition_delay  = predict - predict_clear   minutes the conditions explain
  residual         = actual  - predict         minutes nothing external explains

A genuine weather delay has a large condition_delay and a small residual.
A rider-caused delay has a large residual on a normal day.

The objective is absolute error, so the model predicts the *median* time.
That keeps a few very long tea breaks from inflating everyone's ETA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ml.features import ETA_FEATURES, eta_matrix

XGB_PARAMS = dict(
    objective="reg:absoluteerror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    n_jobs=2,             # gentle on a low-RAM laptop
    random_state=42,
)


@dataclass
class EtaModel:
    model: XGBRegressor | None = None
    # median traffic per (zone_id, hour) in the training data = "a normal day"
    typical_traffic: pd.DataFrame | None = None
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""

    # ------------------------------------------------------------------ training
    def fit(self, train: pd.DataFrame) -> EtaModel:
        """Fit on the training frame; the last 10% (by time) is used for early stopping."""
        cut = int(len(train) * 0.9)
        fit_df, val_df = train.iloc[:cut], train.iloc[cut:]
        self.model = XGBRegressor(**XGB_PARAMS)
        self.model.fit(
            eta_matrix(fit_df), fit_df["actual_minutes"],
            eval_set=[(eta_matrix(val_df), val_df["actual_minutes"])],
            verbose=False,
        )
        self.typical_traffic = (
            train.groupby(["zone_id", "hour"])["traffic_level"].median().rename("typical_traffic").reset_index()
        )
        self.trained_at = datetime.now().isoformat(timespec="seconds")
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(eta_matrix(df))

    def clear_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same trips, but on a normal day for that zone and hour."""
        clear = df[ETA_FEATURES].copy()
        typical = clear[["zone_id", "hour"]].merge(self.typical_traffic, on=["zone_id", "hour"], how="left")
        clear["traffic_level"] = typical["typical_traffic"].fillna(df["traffic_level"].median()).to_numpy()
        clear["rain_mm"] = 0.0
        clear["is_flood"] = 0
        clear["is_road_closure"] = 0
        return clear

    def predict_clear(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict(self.clear_conditions(df))

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-delivery ETA breakdown used by the risk score and the agents."""
        expected = self.predict(df)
        clear = self.predict_clear(df)
        out = pd.DataFrame(index=df.index)
        out["eta_minutes"] = expected
        out["eta_clear_minutes"] = clear
        out["condition_delay"] = np.maximum(expected - clear, 0.0)
        if "actual_minutes" in df:
            out["residual"] = df["actual_minutes"].to_numpy() - expected
            out["delay_vs_clear"] = df["actual_minutes"].to_numpy() - clear
        return out

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def score(y_true, y_pred) -> dict:
        err = np.asarray(y_true) - np.asarray(y_pred)
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "median_ae": float(np.median(np.abs(err))),
        }

    def feature_importance(self) -> pd.Series:
        """Total gain per feature: how much each feature reduced the error."""
        gain = self.model.get_booster().get_score(importance_type="total_gain")
        s = pd.Series({f: gain.get(f, 0.0) for f in ETA_FEATURES})
        return (s / s.sum()).sort_values(ascending=False)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> EtaModel:
        return joblib.load(path)
