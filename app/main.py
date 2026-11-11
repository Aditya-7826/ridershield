"""RiderShield HTTP API (FastAPI).

Endpoints
  POST /claims                    submit a claim (form fields + optional photo) -> 202, processed in background
  GET  /claims/{id}               status, decision, risk breakdown, explanation, evidence, payout
  GET  /claims/{id}/photo         the uploaded photo (for the reviewer)
  GET  /riders/{id}/deliveries    a rider's recent deliveries (to pick which one the claim is about)
  GET  /review/queue              claims waiting for a human
  POST /review/{id}               approve / reject a paused claim
  GET  /stats                     automation rate, approval rate, payouts
  GET  /health                    database, Ollama and models

Run:  uvicorn app.main:create_app --factory --port 8000
      (USE_LLM=0 to run without Ollama, using deterministic fallbacks)
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import text

from agents import config
from agents.deps import Deps
from agents.graph import build_graph, is_paused_for_review, open_checkpointer, thread_config
from agents.llm import OfflineLLM, OllamaClient
from agents.service import attach_photo, create_claim
from app.processing import ClaimProcessor
from db import queries
from db.session import get_engine
from ml.claim_scoring import Models

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_PHOTO_BYTES = 8 * 1024 * 1024


class ReviewIn(BaseModel):
    verdict: str
    reviewer: str
    notes: str | None = None


def _rows(engine, sql: str, **params) -> list[dict]:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), params)]


def create_app(deps: Deps | None = None, checkpoint_path: Path | None = None) -> FastAPI:
    """Build the app. Tests pass their own deps (fake LLM) and checkpoint file."""
    if deps is None:
        use_llm = os.getenv("USE_LLM", "1") != "0"
        deps = Deps(engine=get_engine(), llm=OllamaClient() if use_llm else OfflineLLM(), models=Models.load())
    graph = build_graph(deps, checkpointer=open_checkpointer(checkpoint_path or config.CHECKPOINT_PATH))
    processor = ClaimProcessor(deps, graph)
    engine = deps.engine

    app = FastAPI(title="RiderShield API", version="1.0",
                  description="AI micro-insurance for gig delivery riders (local models only).")
    app.state.processor = processor

    # ------------------------------------------------------------------ claims
    @app.post("/claims", status_code=202)
    async def submit_claim(
        background: BackgroundTasks,
        rider_id: int = Form(...),
        text_: str = Form(..., alias="text", min_length=5, max_length=2000),
        delivery_id: int | None = Form(None),
        submitted_at: datetime | None = Form(None),
        photo: UploadFile | None = File(None),
    ):
        if not _rows(engine, "SELECT id FROM riders WHERE id = :r", r=rider_id):
            raise HTTPException(404, f"rider {rider_id} not found")
        if submitted_at is None:
            if delivery_id is not None:
                d = queries.get_delivery(engine, delivery_id)
                if d.empty:
                    raise HTTPException(404, f"delivery {delivery_id} not found")
                # The synthetic data is historical, so "now" is taken as shortly after the delivery.
                submitted_at = (pd.Timestamp(d.iloc[0]["picked_up_at"])
                                + timedelta(minutes=float(d.iloc[0]["actual_minutes"]) + 20)).to_pydatetime()
            else:
                submitted_at = datetime.now()
        photo_bytes = None
        if photo is not None and photo.filename:
            if photo.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(415, "photo must be JPEG, PNG or WebP")
            photo_bytes = await photo.read()
            if len(photo_bytes) > MAX_PHOTO_BYTES:
                raise HTTPException(413, "photo is larger than 8 MB")

        claim_id = create_claim(engine, rider_id, text_, submitted_at, delivery_id)
        if photo_bytes:
            suffix = Path(photo.filename).suffix.lower() or ".jpg"
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"upload{suffix}"
                path.write_bytes(photo_bytes)
                attach_photo(engine, claim_id, path)
        processor.enqueue(claim_id)
        background.add_task(processor.run, claim_id)
        return {"claim_id": claim_id, "status": "pending", "job": "queued"}

    @app.get("/claims/{claim_id}")
    def get_claim(claim_id: int):
        claim = _rows(engine, "SELECT id, rider_id, delivery_id, submitted_at, text, reason_category, "
                              "claimed_delay_minutes, status FROM claims WHERE id = :c", c=claim_id)
        if not claim:
            raise HTTPException(404, f"claim {claim_id} not found")
        decision = _rows(engine, "SELECT decision, risk_score, breakdown, explanation, created_at FROM decisions "
                                 "WHERE claim_id = :c ORDER BY id DESC LIMIT 1", c=claim_id)
        review = _rows(engine, "SELECT reviewer, verdict, notes, created_at FROM reviews WHERE claim_id = :c "
                               "ORDER BY id DESC LIMIT 1", c=claim_id)
        payout = _rows(engine, "SELECT amount_inr, created_at FROM payouts WHERE claim_id = :c", c=claim_id)
        photo = queries.claim_photo(engine, claim_id)
        state = graph.get_state(thread_config(claim_id)).values or {}
        return jsonable_encoder({
            "claim": claim[0],
            "job": processor.jobs.get(claim_id),
            "decision": decision[0] if decision else None,
            "risk": state.get("risk"),
            "evidence": {k: v for k, v in (state.get("evidence") or {}).items() if k != "inputs"} or None,
            "vision": state.get("vision"),
            "intake": state.get("intake"),
            "review": review[0] if review else None,
            "payout": payout[0] if payout else None,
            "proposed_payout": state.get("proposed_payout"),
            "has_photo": photo is not None and bool(photo.get("file_path")),
            "awaiting_review": is_paused_for_review(graph, claim_id),
        })

    @app.get("/claims/{claim_id}/photo")
    def get_photo(claim_id: int):
        photo = queries.claim_photo(engine, claim_id)
        if not photo or not photo.get("file_path") or not Path(photo["file_path"]).exists():
            raise HTTPException(404, "no photo for this claim")
        return FileResponse(photo["file_path"])

    @app.get("/riders/{rider_id}/deliveries")
    def rider_deliveries(rider_id: int, limit: int = 15):
        rows = _rows(engine, """
            SELECT d.id, d.picked_up_at, z.name AS zone, d.distance_km, d.actual_minutes,
                   (SELECT COUNT(*) FROM claims c WHERE c.delivery_id = d.id) AS claims
            FROM deliveries d JOIN zones z ON z.id = d.zone_id
            WHERE d.rider_id = :r ORDER BY d.picked_up_at DESC LIMIT :n
        """, r=rider_id, n=min(max(limit, 1), 100))
        return jsonable_encoder(rows)

    # ------------------------------------------------------------------ review
    @app.get("/review/queue")
    def review_queue():
        rows = _rows(engine, """
            SELECT c.id AS claim_id, c.rider_id, c.submitted_at, c.text, d.risk_score, d.explanation,
                   d.breakdown->'rules' AS rules, d.breakdown->'proposed_payout' AS proposed_payout
            FROM claims c
            JOIN LATERAL (SELECT * FROM decisions x WHERE x.claim_id = c.id ORDER BY x.id DESC LIMIT 1) d ON TRUE
            WHERE c.status = 'in_review' ORDER BY c.submitted_at
        """)
        for r in rows:
            r["awaiting_review"] = is_paused_for_review(graph, r["claim_id"])
        return jsonable_encoder(rows)

    @app.post("/review/{claim_id}")
    def review(claim_id: int, body: ReviewIn):
        if body.verdict not in ("approve", "reject"):
            raise HTTPException(422, "verdict must be 'approve' or 'reject'")
        if not is_paused_for_review(graph, claim_id):
            raise HTTPException(409, f"claim {claim_id} is not waiting for review")
        state = processor.review(claim_id, body.verdict, body.reviewer, body.notes)
        return jsonable_encoder({"claim_id": claim_id, "status": state["status"], "review": state["review"],
                                 "payout": state.get("payout")})

    # ------------------------------------------------------------------ stats / health
    @app.get("/stats")
    def stats():
        s = _rows(engine, """
            WITH latest AS (
                SELECT DISTINCT ON (claim_id) claim_id, decision FROM decisions ORDER BY claim_id, id DESC
            )
            SELECT COUNT(*) AS decided,
                   COUNT(*) FILTER (WHERE l.decision = 'auto_approve') AS auto_approved,
                   COUNT(*) FILTER (WHERE l.decision = 'auto_reject')  AS auto_rejected,
                   COUNT(*) FILTER (WHERE l.decision = 'human_review') AS sent_to_review,
                   COUNT(*) FILTER (WHERE c.status = 'in_review')      AS waiting_for_review,
                   COUNT(*) FILTER (WHERE c.status IN ('auto_approved', 'approved')) AS approved_total,
                   COUNT(*) FILTER (WHERE c.status IN ('auto_rejected', 'rejected')) AS rejected_total
            FROM latest l JOIN claims c ON c.id = l.claim_id
        """)[0]
        p = _rows(engine, "SELECT COUNT(*) AS n, COALESCE(SUM(amount_inr), 0) AS total, "
                          "COALESCE(AVG(amount_inr), 0) AS avg FROM payouts")[0]
        decided = s["decided"] or 0
        final = (s["approved_total"] or 0) + (s["rejected_total"] or 0)
        return {
            **s,
            "automation_rate": round((s["auto_approved"] + s["auto_rejected"]) / decided, 3) if decided else None,
            "approval_rate": round(s["approved_total"] / final, 3) if final else None,
            "payouts": {"count": p["n"], "total_inr": float(p["total"]), "average_inr": round(float(p["avg"]), 1)},
            "claims_total": _rows(engine, "SELECT COUNT(*) AS n FROM claims")[0]["n"],
        }

    @app.get("/health")
    def health():
        out = {"database": False, "ollama": False, "models_loaded": True,
               "text_model": config.TEXT_MODEL, "vision_model": config.VISION_MODEL}
        try:
            out["database"] = bool(_rows(engine, "SELECT 1 AS ok"))
        except Exception:
            pass
        try:
            out["ollama"] = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3).status_code == 200
        except httpx.HTTPError:
            pass
        out["status"] = "ok" if out["database"] else "degraded"
        return out

    return app
