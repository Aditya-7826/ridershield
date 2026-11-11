"""Evidence agent: gather facts about the claim from PostgreSQL (no LLM).

Collects the delivery record, the zone's conditions, how other deliveries in
the zone were doing at the same time, the rider's history and any photo, and
converts them into the risk-score inputs using the same code as the batch
tools (ml.claim_scoring.claim_evidence_inputs).
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta

import pandas as pd

from agents.deps import Deps
from agents.schemas import ClaimState
from db import queries
from ml.anomaly import rider_feature_table
from ml.claim_scoring import claim_evidence_inputs, prepare_frame
from ml.features import build_delivery_frame


def evidence_node(state: ClaimState, deps: Deps) -> dict:
    delivery_id = state.get("delivery_id")
    if delivery_id is None:
        return {"evidence": {"available": False, "reason": "no delivery matched this claim"}}

    engine, models = deps.engine, deps.models
    submitted = pd.Timestamp(state["submitted_at"]).to_pydatetime()
    rider_id = int(state["rider_id"])

    target = queries.get_delivery(engine, delivery_id)
    zone_id = int(target.iloc[0]["zone_id"])
    hour_start = pd.Timestamp(target.iloc[0]["picked_up_at"]).floor("h").to_pydatetime()
    window, conditions = queries.zone_window(engine, zone_id, hour_start)
    riders = queries.riders_vehicle(engine, window["rider_id"].unique().tolist())
    frame = prepare_frame(build_delivery_frame(window, riders, conditions), models.eta)
    delivery = frame[frame["id"] == delivery_id].iloc[0]

    hist_deliveries, hist_claims = queries.rider_history(engine, rider_id, submitted)
    feats = rider_feature_table(hist_deliveries, hist_claims, submitted)
    if rider_id in feats.index:
        anomaly = float(models.anomaly.score(feats.loc[[rider_id]]).iloc[0])
        rider_notes = models.anomaly.explain(feats.loc[rider_id])
        rider_stats = {k: round(float(v), 3) for k, v in feats.loc[rider_id].items()}
    else:
        anomaly, rider_notes, rider_stats = 0.5, ["new rider with no history"], {}

    zone_name, lat, lon = queries.zone_center(engine, zone_id)
    photo = queries.claim_photo(engine, int(state["claim_id"]))
    prev_hashes = queries.previous_photo_hashes(engine, submitted, int(state["claim_id"]))

    claimed = state.get("intake", {}).get("claimed_delay_minutes")
    claim = {
        "rider_id": rider_id, "submitted_at": submitted,
        # If the rider did not state a delay, assume they claim what actually happened.
        "claimed_delay_minutes": claimed if claimed is not None else max(float(delivery["delay_vs_clear"]), 0.0),
    }
    inputs, facts = claim_evidence_inputs(
        claim, delivery, frame, models, anomaly, hist_claims["submitted_at"], (lat, lon), photo, prev_hashes,
    )
    claims_7d = int((hist_claims["submitted_at"] >= submitted - timedelta(days=7)).sum())

    evidence = {
        "available": True,
        "inputs": asdict(inputs),
        **facts,
        "zone_name": zone_name,
        "zone_center": [lat, lon],
        "rider": {"anomaly_percentile": round(anomaly, 3), "notes": rider_notes, "stats": rider_stats,
                  "claims_last_7d": claims_7d},
        "duplicate_claims": queries.other_claims_for_delivery(engine, delivery_id, int(state["claim_id"])),
        "photo_record": None if photo is None else {
            "file_path": photo.get("file_path"), "phash": photo.get("phash"),
            "captured_at": None if pd.isna(photo.get("captured_at")) else str(photo.get("captured_at")),
        },
        "policy": {k: (v.item() if hasattr(v, "item") else v) for k, v in queries.active_policy(engine).items()},
    }
    return {"evidence": evidence}
