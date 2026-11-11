"""Human-in-the-loop review.

When the decision agent says `human_review`, the graph reaches `review_node`,
which calls LangGraph's `interrupt()`. Execution stops and the full state is
saved by the checkpointer (SQLite file), so the case survives a server
restart. A reviewer later resumes the thread with
`Command(resume={"verdict": "approve"|"reject", "reviewer": ..., "notes": ...})`
and the node continues from the interrupt with that answer.

Note: on resume LangGraph re-runs the node from the top, so all database
writes happen AFTER the interrupt() call.
"""
from __future__ import annotations

from typing import Literal

from langgraph.types import interrupt
from pydantic import BaseModel, Field
from sqlalchemy import text

from agents.deps import Deps
from agents.persist import write_payout
from agents.schemas import ClaimState


class ReviewVerdict(BaseModel):
    verdict: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=64)
    notes: str | None = None


def needs_review(state: ClaimState) -> str:
    return "review" if state["decision"]["decision"] == "human_review" else "done"


def review_node(state: ClaimState, deps: Deps) -> dict:
    answer = interrupt({
        "claim_id": state["claim_id"],
        "risk_score": state.get("risk", {}).get("score"),
        "rules": state["decision"]["rules"],
        "explanation": state["decision"]["explanation"],
        "proposed_payout": state.get("proposed_payout"),
        "question": "Approve or reject this claim?",
    })
    verdict = ReviewVerdict.model_validate(answer)
    claim_id = int(state["claim_id"])
    status = "approved" if verdict.verdict == "approve" else "rejected"
    proposed = state.get("proposed_payout")
    payout = proposed if verdict.verdict == "approve" and proposed and proposed["amount_inr"] > 0 else None
    with deps.engine.begin() as conn:
        conn.execute(text("INSERT INTO reviews (claim_id, reviewer, verdict, notes) VALUES (:c, :r, :v, :n)"),
                     {"c": claim_id, "r": verdict.reviewer, "v": verdict.verdict, "n": verdict.notes})
        conn.execute(text("UPDATE claims SET status = :s WHERE id = :c"), {"s": status, "c": claim_id})
        if payout:
            write_payout(conn, claim_id, payout["amount_inr"], int(state["evidence"]["policy"]["id"]))
    return {"review": verdict.model_dump(), "status": status, "payout": payout}
