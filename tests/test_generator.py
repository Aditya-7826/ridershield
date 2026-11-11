"""Tests for the synthetic data generator (small config so they run fast)."""
import pandas as pd
import pytest

from data.generator import GeneratorConfig, generate

SMALL = GeneratorConfig(seed=1, n_riders=120, n_deliveries=8_000, n_days=30)


@pytest.fixture(scope="module")
def data():
    return generate(SMALL)


def test_same_seed_gives_identical_data(data):
    again = generate(SMALL)
    pd.testing.assert_frame_equal(data.deliveries, again.deliveries)
    pd.testing.assert_frame_equal(data.claims, again.claims)


def test_different_seed_gives_different_data(data):
    other = generate(GeneratorConfig(seed=2, n_riders=120, n_deliveries=8_000, n_days=30))
    assert not data.deliveries["actual_minutes"].equals(other.deliveries["actual_minutes"])


def test_sizes_match_config(data):
    assert len(data.riders) == SMALL.n_riders
    assert len(data.deliveries) == SMALL.n_deliveries
    assert len(data.conditions) == len(data.zones) * SMALL.n_days * 24


def test_all_three_labels_present(data):
    assert set(data.claims["true_label"]) == {"genuine_external", "rider_caused", "fraudulent"}


def test_foreign_keys_are_consistent(data):
    assert data.claims["delivery_id"].isin(data.deliveries["id"]).all()
    assert data.claims["rider_id"].isin(data.riders["id"]).all()
    assert data.claim_evidence["claim_id"].isin(data.claims["id"]).all()
    assert data.deliveries["zone_id"].isin(data.zones["id"]).all()
    # a claim's rider is the rider who made that delivery
    merged = data.claims.merge(data.deliveries, left_on="delivery_id", right_on="id", suffixes=("", "_d"))
    assert (merged["rider_id"] == merged["rider_id_d"]).all()


def test_at_most_one_claim_per_delivery(data):
    assert data.claims["delivery_id"].is_unique


def test_disruptions_slow_down_deliveries(data):
    d = data.deliveries.assign(hour_start=data.deliveries["picked_up_at"].dt.floor("h"))
    m = d.merge(data.conditions[["zone_id", "hour_start", "event_type"]], on=["zone_id", "hour_start"])
    by_event = m.groupby("event_type")["actual_minutes"].mean()
    assert by_event["heavy_rain"] > by_event["none"] + 10
    assert by_event["traffic_jam"] > by_event["none"]


def test_genuine_claims_happen_in_worse_conditions_than_fraud(data):
    c = data.claims.merge(data.deliveries[["id", "zone_id", "picked_up_at"]], left_on="delivery_id", right_on="id")
    c["hour_start"] = c["picked_up_at"].dt.floor("h")
    c = c.merge(data.conditions[["zone_id", "hour_start", "severity"]], on=["zone_id", "hour_start"])
    sev = c.groupby("true_label")["severity"].mean()
    assert sev["genuine_external"] > sev["fraudulent"]


def test_rider_caused_claims_have_more_idle_time(data):
    c = data.claims.merge(data.deliveries[["id", "idle_minutes"]], left_on="delivery_id", right_on="id")
    idle = c.groupby("true_label")["idle_minutes"].median()
    assert idle["rider_caused"] > idle["genuine_external"] + 5


def test_fraudsters_reuse_photos(data):
    ev = data.claim_evidence.merge(data.claims[["id", "true_label"]], left_on="claim_id", right_on="id")
    fraud_hashes = ev.loc[ev["true_label"] == "fraudulent", "phash"]
    other_hashes = ev.loc[ev["true_label"] != "fraudulent", "phash"]
    # Reused photos are near-duplicates: each copy flips up to 3 bits of the
    # original hash, so two copies are at most 6 bits apart.
    ints = [int(h, 16) for h in fraud_hashes]
    close = sum(any(bin(a ^ b).count("1") <= 6 for j, b in enumerate(ints) if j != i) for i, a in enumerate(ints))
    assert close >= 2
    # Photos on non-fraud claims are all different from each other.
    honest = [int(h, 16) for h in other_hashes]
    assert min(bin(a ^ b).count("1") for i, a in enumerate(honest) for b in honest[i + 1:]) > 6


def test_hidden_profile_not_in_rider_table(data):
    assert "profile" not in data.riders.columns
    assert "true_label" not in data.deliveries.columns
