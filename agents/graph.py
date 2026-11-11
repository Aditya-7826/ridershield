"""The RiderShield claim graph (LangGraph StateGraph).

    START -> intake -> evidence -> vision -> risk -> decision -> persist
                                                                   |
                                     human_review? --yes--> review (interrupt) -> END
                                                   --no---> END

Every claim runs in its own thread (thread_id = "claim-<id>"). A SQLite
checkpointer saves the state after each step, so a claim paused for human
review survives a server restart and is resumed with `resume_review`.
"""
from __future__ import annotations

import sqlite3
from functools import partial
from pathlib import Path

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.config import CHECKPOINT_PATH
from agents.decision import decision_node
from agents.deps import Deps
from agents.evidence import evidence_node
from agents.intake import intake_node
from agents.persist import persist_decision
from agents.review import needs_review, review_node
from agents.risk_agent import risk_node
from agents.schemas import ClaimState
from agents.vision import vision_node
from db import queries


def build_graph(deps: Deps, checkpointer=None):
    g = StateGraph(ClaimState)
    g.add_node("intake", partial(intake_node, deps=deps))
    g.add_node("evidence", partial(evidence_node, deps=deps))
    g.add_node("vision", partial(vision_node, deps=deps))
    g.add_node("risk", partial(risk_node, deps=deps))
    g.add_node("decision", partial(decision_node, deps=deps))
    g.add_node("persist", partial(persist_decision, deps=deps))
    g.add_node("review", partial(review_node, deps=deps))
    g.add_edge(START, "intake")
    g.add_edge("intake", "evidence")
    g.add_edge("evidence", "vision")
    g.add_edge("vision", "risk")
    g.add_edge("risk", "decision")
    g.add_edge("decision", "persist")
    g.add_conditional_edges("persist", needs_review, {"review": "review", "done": END})
    g.add_edge("review", END)
    return g.compile(checkpointer=checkpointer)


def open_checkpointer(path: Path = CHECKPOINT_PATH) -> SqliteSaver:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def thread_config(claim_id: int) -> dict:
    return {"configurable": {"thread_id": f"claim-{claim_id}"}}


def initial_state(deps: Deps, claim_id: int) -> ClaimState:
    claim = queries.get_claim(deps.engine, claim_id)
    if claim is None:
        raise ValueError(f"claim {claim_id} not found")
    photo = queries.claim_photo(deps.engine, claim_id)
    delivery_id = claim.get("delivery_id")
    return ClaimState(
        claim_id=int(claim_id),
        rider_id=int(claim["rider_id"]),
        text=str(claim["text"]),
        submitted_at=pd.Timestamp(claim["submitted_at"]).isoformat(),
        delivery_id=None if delivery_id is None or pd.isna(delivery_id) else int(delivery_id),
        photo_path=None if photo is None else photo.get("file_path"),
        errors=[],
    )


def process_claim(graph, deps: Deps, claim_id: int) -> ClaimState:
    """Run a stored claim through the graph. Returns the state (paused if under review)."""
    graph.invoke(initial_state(deps, claim_id), config=thread_config(claim_id))
    return graph.get_state(thread_config(claim_id)).values


def is_paused_for_review(graph, claim_id: int) -> bool:
    return graph.get_state(thread_config(claim_id)).next == ("review",)


def resume_review(graph, claim_id: int, verdict: str, reviewer: str, notes: str | None = None) -> ClaimState:
    """Resume a paused claim with the reviewer's verdict."""
    if not is_paused_for_review(graph, claim_id):
        raise ValueError(f"claim {claim_id} is not waiting for review")
    graph.invoke(Command(resume={"verdict": verdict, "reviewer": reviewer, "notes": notes}),
                 config=thread_config(claim_id))
    return graph.get_state(thread_config(claim_id)).values
