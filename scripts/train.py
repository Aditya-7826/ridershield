"""Train RiderShield's models and write their model cards.

Usage:
    python scripts/train.py --model eta     # XGBoost ETA regression
    python scripts/train.py --model zone    # zone disruption baselines (needs the ETA model)
    python scripts/train.py --model anomaly # Isolation Forest on rider behaviour
    python scripts/train.py --model all

Data comes from PostgreSQL (run scripts/seed_db.py first). The split is by
time: training = before ml.config.TRAIN_CUTOFF, test = after.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.session import get_engine  # noqa: E402
from ml.config import ARTIFACTS_DIR, TRAIN_CUTOFF  # noqa: E402
from ml.eta import XGB_PARAMS, EtaModel  # noqa: E402
from ml.features import ETA_FEATURES, load_delivery_frame  # noqa: E402
from ml.model_card import md_table, write_model_card  # noqa: E402
from ml.zone_disruption import ZoneDisruptionDetector, is_delayed  # noqa: E402
from ml.anomaly import MIN_DELIVERIES, RIDER_FEATURES, RiderAnomalyModel, rider_feature_table  # noqa: E402

ETA_PATH = ARTIFACTS_DIR / "eta_model.joblib"
ZONE_PATH = ARTIFACTS_DIR / "zone_detector.joblib"
ANOMALY_PATH = ARTIFACTS_DIR / "rider_anomaly.joblib"
MAJOR_EVENTS = ["flood", "heavy_rain", "traffic_jam", "road_closure"]


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["picked_up_at"] < TRAIN_CUTOFF]
    test = frame[frame["picked_up_at"] >= TRAIN_CUTOFF]
    return train, test


# --------------------------------------------------------------------------- ETA


def train_eta(frame: pd.DataFrame) -> EtaModel:
    train, test = split(frame)
    print(f"ETA model: {len(train):,} train / {len(test):,} test deliveries (cutoff {TRAIN_CUTOFF:%Y-%m-%d})")
    t0 = time.perf_counter()
    eta = EtaModel().fit(train)
    print(f"  trained in {time.perf_counter() - t0:.1f}s, best iteration {eta.model.best_iteration}")

    pred = eta.predict(test)
    # Naive baseline: median training time for the same distance (0.5 km bins).
    bins = lambda d: (d // 0.5).astype(int)  # noqa: E731
    by_dist = train.groupby(bins(train["distance_km"]))["actual_minutes"].median()
    naive = bins(test["distance_km"]).map(by_dist).fillna(train["actual_minutes"].median())

    metrics = {
        "xgboost": EtaModel.score(test["actual_minutes"], pred),
        "baseline_distance_median": EtaModel.score(test["actual_minutes"], naive),
    }
    by_event = (
        test.assign(abs_err=np.abs(test["actual_minutes"] - pred))
        .groupby("event_type")
        .agg(deliveries=("abs_err", "size"), mae=("abs_err", "mean"), avg_actual=("actual_minutes", "mean"))
        .reset_index().sort_values("deliveries", ascending=False)
    )
    importance = eta.feature_importance()
    eta.metrics = {**metrics, "n_train": len(train), "n_test": len(test)}
    eta.save(ETA_PATH)
    (ARTIFACTS_DIR / "eta_metrics.json").write_text(json.dumps(eta.metrics, indent=2), encoding="utf-8")

    metric_df = pd.DataFrame(
        [{"model": k, "MAE (min)": v["mae"], "RMSE (min)": v["rmse"], "median AE (min)": v["median_ae"]}
         for k, v in metrics.items()]
    )
    imp_df = importance.rename("share of total gain").reset_index().rename(columns={"index": "feature"})

    print("\n== ETA test metrics ==")
    print(metric_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n== Test MAE by condition ==")
    print(by_event.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n== Feature importance (gain) ==")
    print(imp_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    card = write_model_card("eta", "ETA regression (XGBoost)", {
        "Purpose": "Predict how long a delivery should take, under actual conditions and on a normal day. "
                   "The gap between the two is the delay the conditions explain; actual minus predicted is the "
                   "residual nothing external explains.",
        "Training data": f"Synthetic deliveries from `data/generator.py` (seed 42). Time-based split: "
                         f"{len(train):,} deliveries before {TRAIN_CUTOFF:%Y-%m-%d} for training "
                         f"(last 10% used for early stopping), {len(test):,} after for testing.",
        "Features": ", ".join(f"`{f}`" for f in ETA_FEATURES) + ". `zone_id` and `vehicle_type` are categorical.",
        "Model": f"`XGBRegressor` with objective `{XGB_PARAMS['objective']}` (predicts the median, robust to long "
                 f"idle stops), max_depth {XGB_PARAMS['max_depth']}, learning rate {XGB_PARAMS['learning_rate']}, "
                 f"early stopping (best iteration {eta.model.best_iteration}).",
        "Test metrics": md_table(metric_df),
        "Error by condition": md_table(by_event),
        "Feature importance": md_table(imp_df),
        "Limitations": "Trained on synthetic data, so the metrics show the pipeline works, not real-world accuracy. "
                       "Conditions are hourly per zone; real weather varies street by street. The generator has "
                       "no weekday effect, which is why `weekday` gets almost no importance.",
    })
    print(f"\nSaved {ETA_PATH.relative_to(ARTIFACTS_DIR.parents[1])} and {Path(card).name}")
    return eta


# --------------------------------------------------------------------------- zone disruption


def train_zone(frame: pd.DataFrame) -> ZoneDisruptionDetector:
    if not ETA_PATH.exists():
        sys.exit("ETA model not found: run `python scripts/train.py --model eta` first.")
    eta = EtaModel.load(ETA_PATH)
    frame = frame.assign(eta_clear=eta.predict_clear(frame))
    frame["delayed"] = is_delayed(frame["actual_minutes"], frame["eta_clear"])
    train, test = split(frame)

    det = ZoneDisruptionDetector().fit(train)
    scored = det.score_zone_hours(test)
    # Label each zone-hour with what was happening in that hour.
    hour_events = test.assign(hour_start=test["picked_up_at"].dt.floor("h")).groupby(["zone_id", "hour_start"])["event_type"].first()
    scored = scored.merge(hour_events.reset_index(), on=["zone_id", "hour_start"], how="left")
    judged = scored[scored["n"] >= det.min_deliveries]

    by_event = (
        judged.groupby("event_type")
        .agg(zone_hours=("disrupted", "size"), flagged=("disrupted", "sum"), avg_score=("score", "mean"))
        .reset_index()
    )
    by_event["flag_rate"] = by_event["flagged"] / by_event["zone_hours"]
    major = judged["event_type"].isin(MAJOR_EVENTS)
    summary = {
        "detection_rate_major_events": float(judged.loc[major, "disrupted"].mean()),
        "false_alarm_rate_normal_hours": float(judged.loc[judged["event_type"] == "none", "disrupted"].mean()),
        "precision_flagged_is_any_event": float((judged.loc[judged["disrupted"], "event_type"] != "none").mean()),
        "zone_hours_judged": int(len(judged)),
        "global_delayed_share_train": det.global_share,
    }
    det.save(ZONE_PATH)
    (ARTIFACTS_DIR / "zone_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n== Zone disruption detector (test zone-hours with >= {det.min_deliveries} deliveries in window) ==")
    print(by_event.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    card = write_model_card("zone_disruption", "Zone disruption detector", {
        "Purpose": "Flag zones and times where many deliveries are delayed together, which is strong evidence "
                   "that a delay had an external cause.",
        "Method": f"A delivery is *delayed* if actual − normal-day ETA ≥ max(8 min, 30% of ETA). For a window of "
                  f"three hourly buckets around the delivery (claimant excluded), compare the delayed share with "
                  f"the zone's usual share for that hour of day (smoothed toward the global share, k={det.prior_strength}). "
                  f"Disrupted when deliveries ≥ {det.min_deliveries}, share ≥ {det.min_share} and share − baseline "
                  f"≥ {det.min_lift}.",
        "Training data": f"Baselines learned on deliveries before {TRAIN_CUTOFF:%Y-%m-%d}; global delayed share "
                         f"{det.global_share:.3f}.",
        "Test results by condition": md_table(by_event),
        "Summary": "\n".join(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in summary.items()),
        "Limitations": "Quiet hours have too few deliveries to judge (they return 'not disrupted', not 'clear'). "
                       "Light rain is not meant to trigger it. Traffic jams are detected less often: in the data "
                       "only about a quarter of deliveries during a jam pass the delay threshold, so jam claims "
                       "rely more on the ETA model's condition-explained delay and the recorded traffic level. "
                       "A narrower 1-hour window was tried; it judged ~40% fewer zone-hours for a small gain on jams.",
    })
    print(f"\nSaved {ZONE_PATH.relative_to(ARTIFACTS_DIR.parents[1])} and {Path(card).name}")
    return det


# --------------------------------------------------------------------------- rider anomaly


def train_anomaly(frame: pd.DataFrame) -> RiderAnomalyModel:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        claims = pd.read_sql(text("SELECT id, rider_id, submitted_at FROM claims"), conn, parse_dates=["submitted_at"])
    feats = rider_feature_table(frame, claims, TRAIN_CUTOFF)
    model = RiderAnomalyModel().fit(feats)
    feats["anomaly_pct"] = model.score(feats)
    model.save(ANOMALY_PATH)

    used = feats[feats["n_deliveries"] >= MIN_DELIVERIES]
    summary = used[RIDER_FEATURES + ["anomaly_pct"]].describe(percentiles=[0.5, 0.9]).T[["mean", "50%", "90%", "max"]]
    summary = summary.reset_index().rename(columns={"index": "feature", "50%": "median", "90%": "p90"})
    top = used.sort_values("anomaly_pct", ascending=False).head(5)
    top_notes = [f"rider {rid}: pct {r['anomaly_pct']:.2f}; " + "; ".join(model.explain(r) or ["no single extreme feature"])
                 for rid, r in top.iterrows()]

    print(f"\n== Rider anomaly model: {len(used)} riders with >= {MIN_DELIVERIES} deliveries before {TRAIN_CUTOFF:%Y-%m-%d} ==")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Most unusual riders:")
    for line in top_notes:
        print("  " + line)

    card = write_model_card("rider_anomaly", "Rider behaviour anomaly detector (Isolation Forest)", {
        "Purpose": "Score how unusual a rider's behaviour is compared with other riders. One input to the risk "
                   "score; never used alone to reject a claim.",
        "Features": "\n".join(f"- `{f}`" for f in RIDER_FEATURES),
        "Training data": f"{len(used)} riders with at least {MIN_DELIVERIES} deliveries before "
                         f"{TRAIN_CUTOFF:%Y-%m-%d}. Riders with less history get a neutral 0.5.",
        "Model": "`IsolationForest(n_estimators=200, contamination='auto', random_state=42)`. The raw score is "
                 "converted to a percentile among training riders (0.97 = more unusual than 97% of riders).",
        "Feature summary (training riders)": md_table(summary),
        "Most unusual training riders": "\n".join(f"- {n}" for n in top_notes),
        "Limitations and fairness": "Unusual is not dishonest: new riders, riders in slow zones or riders on "
                                    "slower vehicles can look different for innocent reasons. That is why the "
                                    "weight is modest and borderline cases go to a human.",
    })
    print(f"\nSaved {ANOMALY_PATH.relative_to(ARTIFACTS_DIR.parents[1])} and {Path(card).name}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["eta", "zone", "anomaly", "all"], required=True)
    args = parser.parse_args()

    frame = load_delivery_frame(get_engine())
    if frame.empty:
        sys.exit("No deliveries in the database: run `python scripts/seed_db.py` first.")
    if args.model in ("eta", "all"):
        train_eta(frame)
    if args.model in ("zone", "all"):
        train_zone(frame)
    if args.model in ("anomaly", "all"):
        train_anomaly(frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
