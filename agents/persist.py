"""Persist agent output: decision row, claim status, payout (no LLM)."""
from __future__ import annotations

import json

from sqlalchemy import text

from agents.deps import Deps
from agents.schemas import ClaimState

STATUS = {"auto_approve": "auto_approved", "auto_reject": "auto_rejected", "human_review": "in_review"}


def persist_decision(state: ClaimState, deps: Deps) -> dict:
    d, risk, intake = state["decision"], state.get("risk", {}), state.get("intake", {})
    claim_id = int(state["claim_id"])
    record = {
        "score": risk.get("score"), "signals": risk.get("breakdown"), "rules": d["rules"], "facts": d["facts"],
        "facts_cited": d["facts_cited"], "explanation_source": d["explanation_source"],
        "intake": intake, "vision": state.get("vision"), "proposed_payout": state.get("proposed_payout"),
        "errors": state.get("errors", []),
    }
    status = STATUS[d["decision"]]
    if deps.dry_run:
        return {"status": status}
    with deps.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO decisions (claim_id, decision, risk_score, breakdown, explanation)
            VALUES (:c, :d, :s, CAST(:b AS JSONB), :e)
        """), {"c": claim_id, "d": d["decision"], "s": risk.get("score"), "b": json.dumps(record, default=str),
               "e": d["explanation"]})
        conn.execute(text("UPDATE claims SET status = :s WHERE id = :c"), {"s": status, "c": claim_id})
        # Claims submitted through the app start with placeholders; fill them from intake.
        # Seeded (labelled) claims are left untouched so evaluation data never changes.
        if intake:
            conn.execute(text("""
                UPDATE claims SET delivery_id = COALESCE(:d, delivery_id), reason_category = :r,
                       claimed_delay_minutes = COALESCE(:m, claimed_delay_minutes)
                WHERE id = :c AND true_label IS NULL
            """), {"d": state.get("delivery_id"), "r": intake["reason_category"],
                   "m": intake.get("claimed_delay_minutes"), "c": claim_id})
        if state.get("payout") and state["payout"]["amount_inr"] > 0:
            write_payout(conn, claim_id, state["payout"]["amount_inr"], int(state["evidence"]["policy"]["id"]))
    return {"status": status}


def write_payout(conn, claim_id: int, amount: float, policy_id: int) -> None:
    conn.execute(text("""
        INSERT INTO payouts (claim_id, policy_id, amount_inr) VALUES (:c, :p, :a)
        ON CONFLICT (claim_id) DO NOTHING
    """), {"c": claim_id, "p": policy_id, "a": amount})
