"""Zone disruption detector.

Idea: one rider being late proves little. But if MOST deliveries in the same
zone and time window are late, something external (rain, flood, a jam, a
closure) is almost certainly going on. That is strong evidence for a claim.

How it works
------------
1. A delivery counts as "delayed" when it took clearly longer than the ETA
   model's normal-day estimate (eta_clear):
       actual - eta_clear >= max(8 min, 30% of eta_clear)
2. For a claim, look at all OTHER deliveries in the same zone during the hour
   before, the hour of and the hour after the delivery (3 hourly buckets).
3. Compare the delayed share in that window with the zone's usual share for
   that hour of day (learned on training data). Some zones are always a bit
   slow at 7 pm; we only care about unusual slowdowns.
4. Flag a disruption when there are enough deliveries to judge, the delayed
   share is high, and it is well above normal.

disruption_score (0-1) = lift above normal, scaled to the room left above the
baseline, and shrunk when there are only a few deliveries to go on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DELAY_MIN_MINUTES = 8.0
DELAY_FRACTION = 0.30


def is_delayed(actual, eta_clear) -> np.ndarray:
    actual, eta_clear = np.asarray(actual, dtype=float), np.asarray(eta_clear, dtype=float)
    return (actual - eta_clear) >= np.maximum(DELAY_MIN_MINUTES, DELAY_FRACTION * eta_clear)


@dataclass
class DisruptionResult:
    zone_id: int
    window_start: datetime
    window_end: datetime
    n_deliveries: int
    n_delayed: int
    delayed_share: float
    baseline_share: float
    disruption_score: float
    disrupted: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window_start"], d["window_end"] = self.window_start.isoformat(), self.window_end.isoformat()
        return d


@dataclass
class ZoneDisruptionDetector:
    min_deliveries: int = 4     # fewer than this -> not enough evidence either way
    min_share: float = 0.5      # at least half the deliveries delayed...
    min_lift: float = 0.25      # ...and at least 25 points above normal
    prior_strength: float = 20  # smoothing for zone-hours with little history
    baseline: pd.DataFrame | None = None
    global_share: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame) -> ZoneDisruptionDetector:
        """Learn the normal delayed share per (zone, hour of day).

        `frame` needs zone_id, hour and a boolean `delayed` column.
        Small groups are pulled toward the global share (Bayesian smoothing).
        """
        self.global_share = float(frame["delayed"].mean())
        g = frame.groupby(["zone_id", "hour"])["delayed"].agg(["sum", "count"]).reset_index()
        k = self.prior_strength
        g["baseline_share"] = (g["sum"] + k * self.global_share) / (g["count"] + k)
        self.baseline = g[["zone_id", "hour", "baseline_share"]]
        return self

    def baseline_for(self, zone_id: int, hour: int) -> float:
        row = self.baseline[(self.baseline["zone_id"] == zone_id) & (self.baseline["hour"] == hour)]
        return float(row["baseline_share"].iloc[0]) if len(row) else self.global_share

    # ------------------------------------------------------------------ scoring
    def score_counts(self, n: np.ndarray, n_delayed: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised core: returns (share, score, disrupted)."""
        n, n_delayed, baseline = (np.asarray(a, dtype=float) for a in (n, n_delayed, baseline))
        share = np.divide(n_delayed, n, out=np.zeros_like(n), where=n > 0)
        lift = share - baseline
        score = np.clip(lift / np.maximum(1 - baseline, 1e-6), 0, 1) * (n / (n + 2))
        disrupted = (n >= self.min_deliveries) & (share >= self.min_share) & (lift >= self.min_lift)
        return share, np.round(score, 3), disrupted

    def assess(self, frame: pd.DataFrame, zone_id: int, at: datetime, exclude_rider_id: int | None = None) -> DisruptionResult:
        """Assess one zone around one moment (e.g. a claimed delivery).

        `frame` must contain zone_id, picked_up_at, rider_id and `delayed`.
        The claimant's own deliveries are excluded so a rider cannot be their
        own evidence.
        """
        hour_start = pd.Timestamp(at).floor("h")
        start, end = hour_start - timedelta(hours=1), hour_start + timedelta(hours=2)
        w = frame[(frame["zone_id"] == zone_id) & (frame["picked_up_at"] >= start) & (frame["picked_up_at"] < end)]
        if exclude_rider_id is not None:
            w = w[w["rider_id"] != exclude_rider_id]
        n, n_delayed = len(w), int(w["delayed"].sum())
        base = self.baseline_for(zone_id, hour_start.hour)
        share, score, disrupted = self.score_counts([n], [n_delayed], [base])
        return DisruptionResult(
            zone_id=int(zone_id), window_start=start.to_pydatetime(), window_end=end.to_pydatetime(),
            n_deliveries=n, n_delayed=n_delayed, delayed_share=round(float(share[0]), 3),
            baseline_share=round(base, 3), disruption_score=float(score[0]), disrupted=bool(disrupted[0]),
        )

    def score_zone_hours(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score every zone-hour at once (same 3-hour window as `assess`).

        Used for evaluation. Does not exclude any rider.
        """
        f = frame.assign(hour_start=frame["picked_up_at"].dt.floor("h"))
        counts = f.groupby(["zone_id", "hour_start"])["delayed"].agg(["sum", "count"])
        out = []
        for zone_id, g in counts.groupby(level="zone_id"):
            g = g.droplevel("zone_id")
            full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"), fill_value=0)
            rolled = full.rolling(3, center=True, min_periods=1).sum()
            rolled = rolled.loc[g.index]
            out.append(pd.DataFrame({
                "zone_id": zone_id, "hour_start": rolled.index,
                "n": rolled["count"].to_numpy(), "n_delayed": rolled["sum"].to_numpy(),
            }))
        res = pd.concat(out, ignore_index=True)
        res["hour"] = res["hour_start"].dt.hour
        res = res.merge(self.baseline, on=["zone_id", "hour"], how="left")
        res["baseline_share"] = res["baseline_share"].fillna(self.global_share)
        res["share"], res["score"], res["disrupted"] = self.score_counts(res["n"], res["n_delayed"], res["baseline_share"])
        return res

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ZoneDisruptionDetector:
        return joblib.load(path)
