"""Tests for the ETA model and the zone disruption detector (no database needed)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data.generator import GeneratorConfig, generate
from ml.eta import EtaModel
from ml.features import build_delivery_frame
from ml.zone_disruption import ZoneDisruptionDetector, is_delayed


@pytest.fixture(scope="module")
def frame():
    d = generate(GeneratorConfig(seed=3, n_riders=150, n_deliveries=12_000, n_days=40))
    return build_delivery_frame(d.deliveries, d.riders, d.conditions)


@pytest.fixture(scope="module")
def eta(frame):
    return EtaModel().fit(frame.iloc[:9_000])


# ----------------------------------------------------------------------------- ETA


def test_eta_beats_overall_median(frame, eta):
    test = frame.iloc[9_000:]
    model_mae = EtaModel.score(test["actual_minutes"], eta.predict(test))["mae"]
    naive_mae = EtaModel.score(test["actual_minutes"], np.full(len(test), frame["actual_minutes"].median()))["mae"]
    assert model_mae < 0.7 * naive_mae


def test_clear_day_is_faster_than_flood(frame, eta):
    flood = frame[frame["event_type"] == "flood"]
    assert len(flood) > 20
    out = eta.explain(flood)
    assert (out["condition_delay"] > 0).mean() > 0.9
    assert out["condition_delay"].mean() > 15


def test_clear_conditions_remove_weather_only(frame, eta):
    clear = eta.clear_conditions(frame.head(50))
    assert (clear["rain_mm"] == 0).all() and (clear["is_flood"] == 0).all()
    assert (clear["distance_km"].to_numpy() == frame.head(50)["distance_km"].to_numpy()).all()


def test_single_row_prediction_works(frame, eta):
    row = frame.iloc[[0]]
    assert eta.predict(row).shape == (1,)
    assert eta.predict(row)[0] > 0


def test_save_and_load_roundtrip(frame, eta, tmp_path):
    path = tmp_path / "eta.joblib"
    eta.save(path)
    again = EtaModel.load(path)
    np.testing.assert_allclose(eta.predict(frame.head(20)), again.predict(frame.head(20)))


# ----------------------------------------------------------------------------- zone disruption


def test_is_delayed_threshold():
    # 8-minute floor for short trips, 30% for long ones
    assert list(is_delayed([27, 28, 60, 64], [20, 20, 50, 50])) == [False, True, False, False]
    assert is_delayed([66], [50])[0]


@pytest.fixture
def detector():
    det = ZoneDisruptionDetector()
    det.baseline = pd.DataFrame({"zone_id": [1], "hour": [19], "baseline_share": [0.2]})
    det.global_share = 0.15
    return det


def test_too_few_deliveries_is_never_disrupted(detector):
    _, _, disrupted = detector.score_counts([3], [3], [0.1])
    assert not disrupted[0]


def test_high_share_well_above_baseline_is_disrupted(detector):
    share, score, disrupted = detector.score_counts([10], [8], [0.15])
    assert disrupted[0] and share[0] == 0.8 and score[0] > 0.6


def test_high_share_in_normally_slow_zone_is_not_disrupted(detector):
    _, _, disrupted = detector.score_counts([10], [6], [0.5])  # 60% vs a usual 50%
    assert not disrupted[0]


def test_score_is_zero_when_below_baseline(detector):
    _, score, _ = detector.score_counts([10], [1], [0.3])
    assert score[0] == 0


def test_assess_excludes_claimant(detector):
    at = datetime(2026, 6, 1, 19, 20)
    rows = [{"zone_id": 1, "picked_up_at": datetime(2026, 6, 1, 19, m), "rider_id": 99 if m < 10 else m, "delayed": True}
            for m in range(0, 60, 5)]
    frame = pd.DataFrame(rows)
    with_claimant = detector.assess(frame, 1, at)
    without = detector.assess(frame, 1, at, exclude_rider_id=99)
    assert with_claimant.n_deliveries == 12 and without.n_deliveries == 10
    assert without.disrupted and without.baseline_share == 0.2


def test_fit_smooths_small_groups():
    frame = pd.DataFrame({"zone_id": [1] * 100 + [2] * 2, "hour": [10] * 102,
                          "delayed": [False] * 90 + [True] * 10 + [True, True]})
    det = ZoneDisruptionDetector(prior_strength=20).fit(frame)
    z2 = det.baseline_for(2, 10)
    assert 0.1 < z2 < 0.3          # 2/2 delayed is pulled strongly toward the global share
    assert det.baseline_for(5, 3) == det.global_share
