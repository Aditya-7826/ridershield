"""Gather evidence for claims and turn it into risk-score inputs.

One function, `claim_evidence_inputs`, builds the inputs for a single claim.
The batch helper `score_claims` loops over many claims with exactly the same
code, so the risk report, the agents and the evaluation all agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import Engine

from ml.anomaly import RiderAnomalyModel, rider_feature_table
from ml.config import ARTIFACTS_DIR
from ml.eta import EtaModel
from ml.features import load_delivery_frame
from ml.photo_checks import check_photo
from ml.risk import ClaimInputs, RiskConfig, combine, compute_signals
from ml.zone_disruption import ZoneDisruptionDetector, is_delayed

ETA_PATH = ARTIFACTS_DIR / "eta_model.joblib"
ZONE_PATH = ARTIFACTS_DIR / "zone_detector.joblib"
ANOMALY_PATH = ARTIFACTS_DIR / "rider_anomaly.joblib"


@dataclass
class Models:
    eta: EtaModel
    zone: ZoneDisruptionDetector
    anomaly: RiderAnomalyModel
    cfg: RiskConfig

    @classmethod
    def load(cls) -> Models:
        for p in (ETA_PATH, ZONE_PATH, ANOMALY_PATH):
            if not p.exists():
                raise FileNotFoundError(f"{p.name} not found: run `python scripts/train.py --model all`")
        return cls(EtaModel.load(ETA_PATH), ZoneDisruptionDetector.load(ZONE_PATH),
                   RiderAnomalyModel.load(ANOMALY_PATH), RiskConfig.load())


def load_tables(engine: Engine) -> dict[str, pd.DataFrame]:
    with engine.connect() as conn:
        claims = pd.read_sql("SELECT * FROM claims ORDER BY submitted_at", conn, parse_dates=["submitted_at"])
        evidence = pd.read_sql("SELECT * FROM claim_evidence", conn, parse_dates=["captured_at"])
        zones = pd.read_sql("SELECT * FROM zones", conn)
    return {"frame": load_delivery_frame(engine), "claims": claims, "evidence": evidence, "zones": zones}


def prepare_frame(frame: pd.DataFrame, eta: EtaModel) -> pd.DataFrame:
    """Add ETA-based columns to the delivery frame (once, for all claims)."""
    out = frame.copy()
    exp = eta.explain(out)
    out = pd.concat([out, exp], axis=1)
    out["delayed"] = is_delayed(out["actual_minutes"], out["eta_clear_minutes"])
    return out


def claim_evidence_inputs(
    claim: pd.Series | dict,
    delivery: pd.Series,
    frame: pd.DataFrame,
    models: Models,
    rider_anomaly: float,
    rider_claim_times: pd.Series,
    zone_center: tuple[float, float],
    photo: dict | None,
    previous_hashes: list[str],
) -> tuple[ClaimInputs, dict]:
    """Return the risk inputs for one claim plus a dict of human-readable evidence."""
    submitted: datetime = pd.Timestamp(claim["submitted_at"]).to_pydatetime()
    zone = models.zone.assess(frame, int(delivery["zone_id"]), delivery["picked_up_at"], exclude_rider_id=int(claim["rider_id"]))
    recent = rider_claim_times[(rider_claim_times < submitted) & (rider_claim_times >= submitted - timedelta(days=30))]

    photo_check = None
    if photo is not None:
        start = pd.Timestamp(delivery["picked_up_at"]).to_pydatetime()
        end = start + timedelta(minutes=float(delivery["actual_minutes"]))
        captured = photo.get("captured_at")
        captured = None if captured is None or pd.isna(captured) else pd.Timestamp(captured).to_pydatetime()
        gps = None
        if photo.get("gps_lat") is not None and not pd.isna(photo.get("gps_lat")):
            gps = (float(photo["gps_lat"]), float(photo["gps_lon"]))
        photo_check = check_photo(photo.get("phash"), captured, gps, start, end, zone_center, previous_hashes)

    inputs = ClaimInputs(
        claimed_delay=float(claim["claimed_delay_minutes"]),
        observed_delay=float(delivery["delay_vs_clear"]),
        condition_delay=float(delivery["condition_delay"]),
        restaurant_wait=float(delivery["restaurant_wait_minutes"]),
        zone_disruption_score=zone.disruption_score,
        zone_judgeable=zone.n_deliveries >= models.zone.min_deliveries,
        rider_anomaly=float(rider_anomaly),
        claims_last_30d=int(len(recent)),
        image_risk=None if photo_check is None else photo_check.risk,
    )
    evidence = {
        "delivery": {
            "id": int(delivery["id"]), "zone_id": int(delivery["zone_id"]),
            "picked_up_at": pd.Timestamp(delivery["picked_up_at"]).isoformat(),
            "distance_km": float(delivery["distance_km"]), "actual_minutes": float(delivery["actual_minutes"]),
            "idle_minutes": float(delivery["idle_minutes"]),
            "restaurant_wait_minutes": float(delivery["restaurant_wait_minutes"]),
            "eta_minutes": round(float(delivery["eta_minutes"]), 1),
            "eta_clear_minutes": round(float(delivery["eta_clear_minutes"]), 1),
        },
        "conditions": {
            "event_type": delivery["event_type"], "rain_mm": float(delivery["rain_mm"]),
            "traffic_level": float(delivery["traffic_level"]), "severity": float(delivery["severity"]),
        },
        "zone_disruption": zone.to_dict(),
        "photo": None if photo_check is None else photo_check.to_dict(),
        "claims_last_30d": int(len(recent)),
    }
    return inputs, evidence


def score_claims(
    claims: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    models: Models,
    rider_features_as_of: datetime,
) -> pd.DataFrame:
    """Score many claims. Rider behaviour features are a snapshot at `rider_features_as_of`."""
    frame = prepare_frame(tables["frame"], models.eta)
    by_delivery = frame.set_index("id", drop=False)
    all_claims = tables["claims"]
    feats = rider_feature_table(frame, all_claims, rider_features_as_of)
    anomaly = models.anomaly.score(feats)
    zones = tables["zones"].set_index("id")
    ev = tables["evidence"].merge(all_claims[["id", "submitted_at"]], left_on="claim_id", right_on="id", suffixes=("", "_c"))
    ev = ev.sort_values("submitted_at")
    claim_times = {rid: g["submitted_at"] for rid, g in all_claims.groupby("rider_id")}

    rows = []
    for _, c in claims.sort_values("submitted_at").iterrows():
        d = by_delivery.loc[int(c["delivery_id"])]
        photo_row = ev[ev["claim_id"] == c["id"]]
        photo = photo_row.iloc[0].to_dict() if len(photo_row) else None
        prev = ev.loc[(ev["submitted_at"] < c["submitted_at"]) & ev["phash"].notna(), "phash"].tolist()
        inputs, _ = claim_evidence_inputs(
            c, d, frame, models, anomaly.get(int(c["rider_id"]), 0.5),
            claim_times.get(int(c["rider_id"]), pd.Series(dtype="datetime64[ns]")),
            (float(zones.at[int(d["zone_id"]), "lat"]), float(zones.at[int(d["zone_id"]), "lon"])),
            photo, prev,
        )
        signals = compute_signals(inputs, models.cfg)
        result = combine(signals, models.cfg)
        rows.append({"claim_id": int(c["id"]), "true_label": c.get("true_label"),
                     "risk_score": result.score, **asdict(inputs), **{f"sig_{k}": v for k, v in signals.items()}})
    return pd.DataFrame(rows)


def decide(score: float, cfg: RiskConfig) -> str:
    """Map a risk score to a decision using the configured thresholds."""
    if score < cfg.thresholds["auto_approve_below"]:
        return "auto_approve"
    if score >= cfg.thresholds["auto_reject_at_or_above"]:
        return "auto_reject"
    return "human_review"
