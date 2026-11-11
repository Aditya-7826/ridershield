"""Vision agent, photo reuse and the human-in-the-loop review flow."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from sqlalchemy import text

from agents.schemas import IntakeResult, VisionResult
from agents.vision import content_risk
from ml.photo_checks import REUSE_MAX_HAMMING, hamming
from ml.photo_io import compute_phash, read_exif, write_exif
from tests.conftest import FakeLLM, requires_db, requires_models

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


# ----------------------------------------------------------------------------- unit


def test_fixture_copy_is_near_duplicate_and_others_are_not():
    orig = compute_phash(FIXTURES / "flooded_road.jpg")
    assert hamming(orig, compute_phash(FIXTURES / "flooded_road_copy.jpg")) <= REUSE_MAX_HAMMING
    for other in ("traffic_jam.jpg", "clear_road.jpg"):
        assert hamming(orig, compute_phash(FIXTURES / other)) > 3 * REUSE_MAX_HAMMING


def test_exif_round_trip(tmp_path):
    when = pd.Timestamp("2026-08-21 18:05:00").to_pydatetime()
    out = tmp_path / "x.jpg"
    write_exif(Image.open(FIXTURES / "traffic_jam.jpg"), out, when, (12.97, 77.64))
    captured, gps = read_exif(out)
    assert captured == when
    assert gps == pytest.approx((12.97, 77.64), abs=1e-4)


def test_missing_exif_reads_as_none():
    assert read_exif(FIXTURES / "flooded_road_copy.jpg") == (None, None)


@pytest.mark.parametrize("matches,conf,shows,expected", [
    (True, 0.9, "flooded_road", 0.0),
    (False, 0.3, "other", 0.3),          # unsure -> mild
    (False, 0.9, "other", 0.85),         # confident mismatch
    (False, 0.9, "unclear", 0.3),
])
def test_content_risk(matches, conf, shows, expected):
    r = VisionResult(shows=shows, matches_claim=matches, confidence=conf, description="x")
    assert content_risk(r) == pytest.approx(expected)


# ----------------------------------------------------------------------------- integration


def _flood_deliveries(engine, n=2):
    """Unclaimed held-out flood deliveries by different, low-claim riders, oldest first."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.rider_id, d.picked_up_at, d.actual_minutes, z.lat, z.lon FROM deliveries d
            JOIN conditions c ON c.zone_id = d.zone_id AND c.hour_start = date_trunc('hour', d.picked_up_at)
            JOIN zones z ON z.id = d.zone_id
            LEFT JOIN claims cl ON cl.delivery_id = d.id
            WHERE c.event_type = 'flood' AND d.idle_minutes < 3 AND cl.id IS NULL AND d.picked_up_at >= '2026-08-10'
              AND (SELECT COUNT(*) FROM claims x WHERE x.rider_id = d.rider_id) <= 2
            ORDER BY d.picked_up_at
        """)).all()
    picked, riders = [], set()
    for r in rows:
        if r.rider_id not in riders:
            picked.append(r)
            riders.add(r.rider_id)
        if len(picked) == n:
            return picked
    pytest.skip("not enough suitable flood deliveries")


@pytest.fixture
def env(tmp_path):
    from db.session import get_engine
    engine = get_engine()
    before = {"claims": None}
    with engine.connect() as conn:
        before["claims"] = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM claims")).scalar_one()
    yield engine, tmp_path
    with engine.begin() as conn:
        new = "(SELECT id FROM claims WHERE id > :m AND true_label IS NULL)"
        for table in ("reviews", "payouts", "decisions", "claim_evidence"):
            conn.execute(text(f"DELETE FROM {table} WHERE claim_id IN {new}"), {"m": before["claims"]})
        conn.execute(text("DELETE FROM claims WHERE id > :m AND true_label IS NULL"), {"m": before["claims"]})


def _make_deps(engine, llm, tmp_path):
    from agents.deps import Deps
    from agents.graph import build_graph, open_checkpointer
    from ml.claim_scoring import Models
    deps = Deps(engine=engine, llm=llm, models=Models.load())
    return deps, build_graph(deps, checkpointer=open_checkpointer(tmp_path / "cp.sqlite"))


def _submit(engine, row, claim_text, photo, upload_dir):
    from agents.service import attach_photo, create_claim
    submitted = pd.Timestamp(row.picked_up_at) + timedelta(minutes=float(row.actual_minutes) + 15)
    claim_id = create_claim(engine, row.rider_id, claim_text, submitted.to_pydatetime(), delivery_id=row.id)
    attach_photo(engine, claim_id, photo, upload_dir)
    return claim_id


@requires_db
@requires_models
def test_genuine_photo_passes_metadata_and_vision(env):
    from agents.graph import process_claim
    engine, tmp = env
    (a,) = _flood_deliveries(engine, 1)
    photo = tmp / "a.jpg"   # same picture, EXIF set to the delivery time and zone
    write_exif(Image.open(FIXTURES / "flooded_road.jpg"), photo,
               (pd.Timestamp(a.picked_up_at) + timedelta(minutes=10)).to_pydatetime(), (a.lat, a.lon))
    claim_id = _submit(engine, a, f"Road flooded, order {a.id} 20 min late", photo, tmp / "up")
    llm = FakeLLM(intake=IntakeResult(reason_category="flooding", summary="flood"),
                  vision=VisionResult(shows="flooded_road", matches_claim=True, confidence=0.9, description="flooded street"),
                  explanation="The claim was approved because the road was flooded.")
    deps, graph = _make_deps(engine, llm, tmp)
    state = process_claim(graph, deps, claim_id)
    v = state["vision"]
    assert v["metadata"]["exif_status"] == "ok" and not v["metadata"]["reused"]
    assert v["content_checked"] and v["image_risk"] == 0.0
    assert "VisionResult" in llm.calls
    with engine.connect() as conn:
        stored = conn.execute(text("SELECT vision_result FROM claim_evidence WHERE claim_id = :c"), {"c": claim_id}).scalar_one()
    assert stored["content_checked"] is True


@requires_db
@requires_models
def test_reused_photo_flagged_paused_survives_restart_and_resumes(env):
    """Claim A submits a photo; claim B (another rider) reuses a resized copy.

    B is processed in a SEPARATE process that exits while B waits for review.
    A fresh graph + checkpointer connection then resumes it (a server restart).
    """
    from agents.graph import build_graph, is_paused_for_review, open_checkpointer, process_claim, resume_review
    from agents.deps import Deps
    from agents.llm import OfflineLLM
    from ml.claim_scoring import Models

    engine, tmp = env
    a, b = _flood_deliveries(engine, 2)
    photo_a = tmp / "a.jpg"
    write_exif(Image.open(FIXTURES / "flooded_road.jpg"), photo_a,
               (pd.Timestamp(a.picked_up_at) + timedelta(minutes=10)).to_pydatetime(), (a.lat, a.lon))
    claim_a = _submit(engine, a, f"Road flooded, order {a.id} 20 min late", photo_a, tmp / "up")
    deps_a, graph_a = _make_deps(engine, OfflineLLM(), tmp)
    process_claim(graph_a, deps_a, claim_a)

    # --- process 1: submit claim B with the copied photo, then exit while paused
    cp_path = tmp / "restart_cp.sqlite"
    env_vars = {**os.environ, "CHECKPOINT_PATH": str(cp_path), "UPLOAD_DIR": str(tmp / "up"), "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run(
        [sys.executable, "scripts/submit_claim.py", "--rider", str(b.rider_id), "--delivery-id", str(b.id),
         "--text", f"Waterlogging, order {b.id} was 20 min late", "--photo", str(FIXTURES / "flooded_road_copy.jpg"),
         "--no-llm"],
        cwd=ROOT, env=env_vars, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    claim_b = int(re.search(r"Created claim (\d+)", out.stdout).group(1))
    assert "reused=True" in out.stdout
    assert "PAUSED for human review" in out.stdout

    with engine.connect() as conn:
        status = conn.execute(text("SELECT status FROM claims WHERE id = :c"), {"c": claim_b}).scalar_one()
        rules = conn.execute(text("SELECT breakdown->'rules' FROM decisions WHERE claim_id = :c"), {"c": claim_b}).scalar_one()
    assert status == "in_review"
    assert any("previously submitted photo" in r for r in rules)

    # --- "after restart": brand-new graph and checkpointer connection
    deps = Deps(engine=engine, llm=OfflineLLM(), models=Models.load())
    graph = build_graph(deps, checkpointer=open_checkpointer(cp_path))
    assert is_paused_for_review(graph, claim_b)
    state = resume_review(graph, claim_b, "approve", "asha", "delay is real, photo was reused")
    assert state["status"] == "approved" and state["review"]["reviewer"] == "asha"
    assert not is_paused_for_review(graph, claim_b)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT status FROM claims WHERE id = :c"), {"c": claim_b}).scalar_one() == "approved"
        review = conn.execute(text("SELECT reviewer, verdict FROM reviews WHERE claim_id = :c"), {"c": claim_b}).one()
        payouts = conn.execute(text("SELECT COUNT(*) FROM payouts WHERE claim_id = :c"), {"c": claim_b}).scalar_one()
    assert tuple(review) == ("asha", "approve")
    assert payouts == (1 if state["payout"] else 0)


@requires_db
@requires_models
def test_review_reject_and_double_resume_is_refused(env):
    from agents.graph import process_claim, resume_review
    from agents.llm import OfflineLLM

    engine, tmp = env
    a, b = _flood_deliveries(engine, 2)
    deps, graph = _make_deps(engine, OfflineLLM(), tmp)
    photo_a = tmp / "a.jpg"
    write_exif(Image.open(FIXTURES / "flooded_road.jpg"), photo_a,
               (pd.Timestamp(a.picked_up_at) + timedelta(minutes=10)).to_pydatetime(), (a.lat, a.lon))
    process_claim(graph, deps, _submit(engine, a, f"Flood, order {a.id} 20 min late", photo_a, tmp / "up"))
    claim_b = _submit(engine, b, f"Flood, order {b.id} 20 min late", FIXTURES / "flooded_road_copy.jpg", tmp / "up")
    state = process_claim(graph, deps, claim_b)
    assert state["status"] == "in_review"

    state = resume_review(graph, claim_b, "reject", "ravi", "same photo as another rider")
    assert state["status"] == "rejected" and state["payout"] is None
    with pytest.raises(ValueError):
        resume_review(graph, claim_b, "approve", "ravi")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM payouts WHERE claim_id = :c"), {"c": claim_b}).scalar_one() == 0


def test_invalid_review_payload_is_rejected():
    from pydantic import ValidationError

    from agents.review import ReviewVerdict
    with pytest.raises(ValidationError):
        ReviewVerdict.model_validate({"verdict": "maybe", "reviewer": "x"})
    with pytest.raises(ValidationError):
        ReviewVerdict.model_validate({"verdict": "approve", "reviewer": ""})
