"""Tests for payout policy, intake, decision rules and the full agent graph."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import text

from agents.decision import _contradicts, apply_rules, template_explanation
from agents.intake import _number_in_text, fallback_intake
from agents.schemas import IntakeResult
from ml.risk import RiskConfig
from policy.payout import Policy, calculate_payout
from tests.conftest import FakeLLM, requires_db, requires_models

POLICY = Policy(id=1, name="test", base_payout_inr=30, per_minute_inr=2, max_payout_inr=150,
                min_delay_minutes=10, max_claims_per_week=5)


# ----------------------------------------------------------------------------- payout


def test_payout_pays_claimed_when_smaller():
    p = calculate_payout(POLICY, claimed_minutes=25, observed_minutes=126)
    assert p.payable_minutes == 25 and p.amount_inr == 60   # 30 + 2 * 15


def test_payout_never_pays_inflation_beyond_tolerance():
    p = calculate_payout(POLICY, claimed_minutes=45, observed_minutes=12, tolerance_minutes=5)
    assert p.payable_minutes == 17 and p.amount_inr == 44   # 30 + 2 * 7


def test_payout_below_minimum_is_zero():
    assert calculate_payout(POLICY, claimed_minutes=8, observed_minutes=30).amount_inr == 0


def test_payout_is_capped():
    p = calculate_payout(POLICY, claimed_minutes=None, observed_minutes=200)
    assert p.amount_inr == 150 and "capped" in p.formula


def test_payout_negative_observed_delay():
    assert calculate_payout(POLICY, claimed_minutes=20, observed_minutes=-15, tolerance_minutes=5).amount_inr == 0


# ----------------------------------------------------------------------------- intake


@pytest.mark.parametrize("text,delivery,minutes,reason", [
    ("Road near Bellandur was flooded, order #45210 was 40 min late", 45210, 40, "flooding"),
    ("Sir road near BTM water logging very bad. Order no 43099 got late around 25 minutes", 43099, 25, "flooding"),
    ("Massive traffic jam, delivery 512 delayed by 18 minutes", 512, 18, "traffic_jam"),
    ("Barricades on the usual route, long detour. 24 minute delay on order 45510.", 45510, 24, "road_closure"),
    ("Order 77 was a 30-min delay due to rain", 77, 30, "heavy_rain"),
    ("Restaurant took very long to give the order. Waited a lot.", None, None, "restaurant_delay"),
    ("Police had blocked the road, long detour", None, None, "road_closure"),
    ("Pouring rain today", None, None, "heavy_rain"),
    ("something happened", None, None, "other"),
])
def test_fallback_intake(text, delivery, minutes, reason):
    r = fallback_intake(text)
    assert r.delivery_id == delivery
    assert r.claimed_delay_minutes == minutes
    assert r.reason_category == reason


def test_number_in_text_is_exact():
    assert _number_in_text(43099, "order 43099 late")
    assert not _number_in_text(4309, "order 43099 late")


# ----------------------------------------------------------------------------- decision rules


def _state(score, claims_7d=0, reused=False, duplicate=0, available=True):
    ev = {"available": available, "duplicate_claims": duplicate, "rider": {"claims_last_7d": claims_7d},
          "policy": {"max_claims_per_week": 5}, "photo": {"reused": reused, "reasons": []}}
    return {"evidence": ev, "risk": {"score": score}}


@pytest.fixture(scope="module")
def deps():
    return SimpleNamespace(models=SimpleNamespace(cfg=RiskConfig.load()))


def test_rule_low_score_approves(deps):
    assert apply_rules(_state(0.1), deps) == ("auto_approve", [])


def test_rule_high_score_rejects(deps):
    assert apply_rules(_state(0.9), deps)[0] == "auto_reject"


def test_rule_reused_photo_blocks_auto_approve(deps):
    decision, rules = apply_rules(_state(0.1, reused=True), deps)
    assert decision == "human_review" and "previously submitted photo" in rules[0]


def test_rule_weekly_cap_blocks_auto_approve(deps):
    assert apply_rules(_state(0.1, claims_7d=5), deps)[0] == "human_review"


def test_rule_duplicate_rejects_even_with_low_score(deps):
    assert apply_rules(_state(0.05, duplicate=1), deps)[0] == "auto_reject"


def test_rule_no_delivery_goes_to_human(deps):
    assert apply_rules(_state(0.0, available=False), deps)[0] == "human_review"


def test_contradiction_check():
    assert _contradicts("Your claim was rejected.", "auto_approve")
    assert _contradicts("The claim has been approved.", "auto_reject")
    assert not _contradicts("The claim was not approved because...", "auto_reject")


def test_template_explanation_mentions_payout_on_approval():
    facts = ["Decision: approved.", "Risk score 0.1.", "Conditions in X: flood.", "The delivery took 50 min.",
             "Payout: Rs 60."]
    out = template_explanation("auto_approve", facts)
    assert "flood" in out and "Rs 60" in out


# ----------------------------------------------------------------------------- full graph


@pytest.fixture
def graph_setup(tmp_path):
    from agents.deps import Deps
    from agents.graph import build_graph, open_checkpointer
    from db.session import get_engine
    from ml.claim_scoring import Models

    engine = get_engine()
    created: list[int] = []

    def make(llm):
        deps = Deps(engine=engine, llm=llm, models=Models.load())
        return deps, build_graph(deps, checkpointer=open_checkpointer(tmp_path / "cp.sqlite"))

    yield engine, make, created
    # Undo only what the tests wrote. Seeded (labelled) claims keep their evidence;
    # only claims created by the tests (true_label IS NULL) are deleted.
    with engine.begin() as conn:
        for cid in created:
            conn.execute(text("DELETE FROM payouts WHERE claim_id = :c"), {"c": cid})
            conn.execute(text("DELETE FROM decisions WHERE claim_id = :c"), {"c": cid})
            conn.execute(text("DELETE FROM claim_evidence WHERE claim_id = :c AND claim_id IN "
                              "(SELECT id FROM claims WHERE true_label IS NULL)"), {"c": cid})
            conn.execute(text("DELETE FROM claims WHERE id = :c AND true_label IS NULL"), {"c": cid})
            conn.execute(text("UPDATE claims SET status = 'pending' WHERE id = :c"), {"c": cid})


def _unclaimed_flood_delivery(engine):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT d.id, d.rider_id FROM deliveries d
            JOIN conditions c ON c.zone_id = d.zone_id AND c.hour_start = date_trunc('hour', d.picked_up_at)
            LEFT JOIN claims cl ON cl.delivery_id = d.id
            WHERE c.event_type = 'flood' AND d.idle_minutes < 3 AND cl.id IS NULL
            ORDER BY d.id DESC LIMIT 1
        """)).one()


@requires_db
@requires_models
def test_graph_new_claim_end_to_end(graph_setup):
    from datetime import timedelta

    import pandas as pd

    from agents.graph import process_claim
    from agents.service import create_claim
    from db import queries

    engine, make, created = graph_setup
    delivery_id, rider_id = _unclaimed_flood_delivery(engine)
    d = queries.get_delivery(engine, delivery_id).iloc[0]
    submitted = pd.Timestamp(d["picked_up_at"]) + timedelta(minutes=float(d["actual_minutes"]) + 15)
    claim_id = create_claim(engine, int(rider_id), f"Flooded road, order {delivery_id} was 20 min late",
                            submitted.to_pydatetime())
    created.append(claim_id)

    llm = FakeLLM(intake=IntakeResult(delivery_id=None, reason_category="flooding", summary="flood"),
                  explanation="The claim was approved because the road was flooded.")
    deps, graph = make(llm)
    state = process_claim(graph, deps, claim_id)

    assert state["intake"]["resolved_delivery_id"] == delivery_id      # filled from the text by regex
    assert state["intake"]["claimed_delay_minutes"] == 20
    assert state["risk"]["available"] and 0 <= state["risk"]["score"] <= 1
    assert state["decision"]["decision"] in {"auto_approve", "auto_reject", "human_review"}
    assert state["decision"]["explanation"]
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, delivery_id, claimed_delay_minutes FROM claims WHERE id = :c"),
                           {"c": claim_id}).one()
        n_dec = conn.execute(text("SELECT COUNT(*) FROM decisions WHERE claim_id = :c"), {"c": claim_id}).scalar_one()
    assert row.status == state["status"] and row.delivery_id == delivery_id and row.claimed_delay_minutes == 20
    assert n_dec == 1


@requires_db
@requires_models
def test_graph_survives_llm_outage(graph_setup):
    from agents.graph import process_claim

    engine, make, created = graph_setup
    with engine.connect() as conn:
        claim_id = conn.execute(text("SELECT id FROM claims WHERE true_label = 'fraudulent' ORDER BY id DESC LIMIT 1")).scalar_one()
    created.append(claim_id)
    deps, graph = make(FakeLLM(fail=True))
    state = process_claim(graph, deps, claim_id)
    assert state["intake"]["source"] == "fallback"
    assert state["decision"]["explanation_source"] == "template"
    assert state["decision"]["explanation"].startswith("Decision:")
    assert state["status"] in {"auto_approved", "auto_rejected", "in_review"}
    assert len(state["errors"]) == 2


@requires_db
@requires_models
def test_graph_duplicate_claim_rejected(graph_setup):
    from agents.graph import process_claim
    from agents.service import create_claim

    engine, make, created = graph_setup
    with engine.connect() as conn:
        c = conn.execute(text("SELECT rider_id, delivery_id, submitted_at FROM claims WHERE true_label = 'genuine_external' "
                              "ORDER BY id DESC LIMIT 1")).one()
    claim_id = create_claim(engine, c.rider_id, "same delay again", c.submitted_at, delivery_id=c.delivery_id)
    created.append(claim_id)
    deps, graph = make(FakeLLM(explanation="The claim was rejected as a duplicate."))
    state = process_claim(graph, deps, claim_id)
    assert state["decision"]["decision"] == "auto_reject"
    assert "already exists" in state["decision"]["rules"][0]


@requires_db
@requires_models
def test_checkpointer_stores_state(graph_setup):
    from agents.graph import process_claim, thread_config

    engine, make, created = graph_setup
    with engine.connect() as conn:
        claim_id = conn.execute(text("SELECT id FROM claims WHERE true_label = 'rider_caused' ORDER BY id DESC LIMIT 1")).scalar_one()
    created.append(claim_id)
    deps, graph = make(FakeLLM())
    process_claim(graph, deps, claim_id)
    snap = graph.get_state(thread_config(claim_id))
    assert snap.values["claim_id"] == claim_id and snap.values["status"]
    assert not snap.next          # finished: nothing left to run
