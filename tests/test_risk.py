"""Edge-case tests for the risk score, photo checks and rider anomaly model."""
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.anomaly import MIN_DELIVERIES, RIDER_FEATURES, RiderAnomalyModel
from ml.claim_scoring import decide
from ml.photo_checks import RISK_MISMATCH, RISK_NO_EXIF, RISK_REUSED, check_photo, hamming
from ml.risk import SIGNALS, ClaimInputs, RiskConfig, combine, compute_signals, score_claim


@pytest.fixture(scope="module")
def cfg():
    return RiskConfig.load()


GENUINE = ClaimInputs(
    claimed_delay=30, observed_delay=31, condition_delay=28, restaurant_wait=3,
    zone_disruption_score=0.9, zone_judgeable=True, rider_anomaly=0.3, claims_last_30d=0, image_risk=0.0,
)


# ----------------------------------------------------------------------------- config


def test_config_weights_sum_to_one(cfg):
    assert sum(cfg.weights[s] for s in SIGNALS) == pytest.approx(1.0)


def test_config_rejects_bad_thresholds(cfg):
    bad = RiskConfig(cfg.weights, cfg.params, {"auto_approve_below": 0.8, "auto_reject_at_or_above": 0.5})
    with pytest.raises(ValueError):
        bad.validate()


def test_config_rejects_missing_weight(cfg):
    weights = {k: v for k, v in cfg.weights.items() if k != "image_check"}
    with pytest.raises(ValueError):
        RiskConfig(weights, cfg.params, cfg.thresholds).validate()


# ----------------------------------------------------------------------------- score


def test_genuine_flood_claim_scores_low(cfg):
    assert score_claim(GENUINE, cfg).score < cfg.thresholds["auto_approve_below"]


def test_all_signals_zero_gives_zero(cfg):
    r = combine({s: 0.0 for s in SIGNALS}, cfg)
    assert r.score == 0.0


def test_all_signals_one_gives_one(cfg):
    r = combine({s: 1.0 for s in SIGNALS}, cfg)
    assert r.score == pytest.approx(1.0)


def test_missing_photo_is_not_penalised(cfg):
    sig = {s: 0.2 for s in SIGNALS}
    with_photo = combine(sig, cfg).score
    sig["image_check"] = None
    without = combine(sig, cfg)
    assert without.score == pytest.approx(with_photo)       # same evidence -> same score
    assert without.breakdown["image_check"]["used"] is False
    assert sum(b["weight"] for b in without.breakdown.values()) == pytest.approx(1.0, abs=1e-3)


def test_breakdown_contributions_add_up(cfg):
    r = score_claim(replace(GENUINE, rider_anomaly=0.95, claims_last_30d=4), cfg)
    assert sum(b["contribution"] for b in r.breakdown.values()) == pytest.approx(r.score, abs=1e-3)


def test_signals_stay_in_range_for_extreme_inputs(cfg):
    weird = ClaimInputs(claimed_delay=0, observed_delay=-40, condition_delay=500, restaurant_wait=0,
                        zone_disruption_score=1.5, zone_judgeable=True, rider_anomaly=-1, claims_last_30d=999)
    sig = compute_signals(weird, cfg)
    assert all(v is None or 0.0 <= v <= 1.0 for v in sig.values())


def test_zone_without_enough_deliveries_is_neutral(cfg):
    sig = compute_signals(replace(GENUINE, zone_judgeable=False, zone_disruption_score=0.0), cfg)
    assert sig["zone_disruption"] == cfg.params["zone_neutral_value"]


def test_unexplained_delay_raises_risk(cfg):
    # Real 30-minute delay on a clear day: nothing external explains it.
    rider_caused = replace(GENUINE, condition_delay=0, zone_disruption_score=0.0)
    assert compute_signals(rider_caused, cfg)["condition_support"] > 0.8
    assert score_claim(rider_caused, cfg).score > score_claim(GENUINE, cfg).score


def test_restaurant_wait_counts_as_external(cfg):
    restaurant = replace(GENUINE, condition_delay=0, restaurant_wait=34)
    assert compute_signals(restaurant, cfg)["condition_support"] == 0.0


def test_inflated_claim_raises_inflation_signal(cfg):
    sig = compute_signals(replace(GENUINE, claimed_delay=40, observed_delay=2), cfg)
    assert sig["claim_inflation"] > 0.8


def test_honest_small_overstatement_is_tolerated(cfg):
    sig = compute_signals(replace(GENUINE, claimed_delay=34, observed_delay=31), cfg)
    assert sig["claim_inflation"] == 0.0


def test_decide_boundaries(cfg):
    lo, hi = cfg.thresholds["auto_approve_below"], cfg.thresholds["auto_reject_at_or_above"]
    assert decide(lo - 0.001, cfg) == "auto_approve"
    assert decide(hi, cfg) == "auto_reject"
    if hi > lo:
        assert decide(lo, cfg) == "human_review"


# ----------------------------------------------------------------------------- photo checks

START = datetime(2026, 7, 1, 19, 0)
END = START + timedelta(minutes=40)
CENTER = (12.9352, 77.6245)


def test_hamming():
    assert hamming("ffff", "fffe") == 1
    assert hamming("0000000000000000", "ffffffffffffffff") == 64


def test_reused_photo_flagged():
    prev = ["a1b2c3d4e5f60718"]
    near_copy = f"{int(prev[0], 16) ^ 0b101:016x}"   # 2 bits flipped
    c = check_photo(near_copy, START, CENTER, START, END, CENTER, prev)
    assert c.reused and c.risk == RISK_REUSED and c.closest_hamming == 2


def test_different_photo_not_flagged():
    c = check_photo("0123456789abcdef", START + timedelta(minutes=10), CENTER, START, END, CENTER, ["fedcba9876543210"])
    assert not c.reused and c.risk == 0.0 and c.exif_status == "ok"


def test_old_photo_is_time_mismatch():
    c = check_photo("0123456789abcdef", START - timedelta(days=30), CENTER, START, END, CENTER, [])
    assert c.exif_status == "time_mismatch" and c.risk == RISK_MISMATCH


def test_far_away_photo_is_location_mismatch():
    c = check_photo("0123456789abcdef", START, (13.10, 77.59), START, END, CENTER, [])
    assert c.exif_status == "location_mismatch" and c.risk == RISK_MISMATCH


def test_missing_exif():
    c = check_photo("0123456789abcdef", None, None, START, END, CENTER, [])
    assert c.exif_status == "missing" and c.risk == RISK_NO_EXIF


def test_first_photo_has_nothing_to_compare():
    c = check_photo("0123456789abcdef", START, CENTER, START, END, CENTER, [])
    assert c.closest_hamming is None and not c.reused


# ----------------------------------------------------------------------------- anomaly


def _riders(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "n_deliveries": rng.integers(20, 200, n),
        "idle_ratio": rng.normal(0.05, 0.01, n).clip(0),
        "long_stop_rate": rng.normal(0.03, 0.01, n).clip(0),
        "median_speed_kmh": rng.normal(12, 1, n),
        "claims_per_week": rng.normal(0.2, 0.05, n).clip(0),
        "claim_to_delivery_ratio": rng.normal(0.02, 0.005, n).clip(0),
    }, index=pd.RangeIndex(1, n + 1, name="rider_id"))


def test_anomaly_outlier_scores_high():
    feats = _riders()
    model = RiderAnomalyModel().fit(feats)
    odd = feats.iloc[[0]].copy()
    odd[["idle_ratio", "long_stop_rate", "claims_per_week"]] = [0.35, 0.4, 3.0]
    assert model.score(odd).iloc[0] > 0.95
    assert model.score(feats).median() == pytest.approx(0.5, abs=0.1)
    assert any("idle ratio" in n or "claims per week" in n for n in model.explain(odd.iloc[0]))


def test_anomaly_short_history_is_neutral():
    feats = _riders()
    model = RiderAnomalyModel().fit(feats)
    newbie = feats.iloc[[0]].assign(n_deliveries=MIN_DELIVERIES - 1, claims_per_week=5.0)
    assert model.score(newbie).iloc[0] == 0.5
    assert set(RIDER_FEATURES) <= set(feats.columns)
