"""Vision agent: does the attached photo support the claim?

Two layers:
1. Metadata (already computed by the evidence agent, ml.photo_checks):
   perceptual-hash reuse, EXIF time and EXIF GPS against the delivery.
2. Content: a small local vision-language model looks at the photo and says
   what it shows and whether that plausibly matches the claimed reason.

To stay within a low-RAM laptop, the text model is unloaded before the
vision model is used, and the vision model is unloaded right after.

image_risk = max(metadata risk, content risk), where content risk is 0 when
the photo matches, 0.3 when the model is unsure, and 0.4-0.9 (by confidence)
when it clearly does not match.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from agents.config import TEXT_MODEL, VISION_MODEL
from agents.deps import Deps
from agents.llm import LLMError
from agents.schemas import ClaimState, VisionResult

SYSTEM = """You check photos attached to delivery-delay insurance claims.
Describe what the photo shows and decide if it plausibly shows the situation the rider claims.
shows must be one of: flooded_road, heavy_rain, traffic_jam, accident, closed_road, restaurant, other, unclear.
Be strict: a dry empty road does not support a flooding or traffic claim."""

CLAIM_WORDS = {
    "heavy_rain": "heavy rain", "flooding": "a flooded or waterlogged road", "traffic_jam": "a traffic jam",
    "road_closure": "a closed or blocked road", "restaurant_delay": "a busy restaurant / waiting for food",
    "other": "a delivery problem",
}
MIN_CONFIDENCE = 0.5


def content_risk(result: VisionResult) -> float:
    if result.matches_claim:
        return 0.0
    if result.confidence < MIN_CONFIDENCE or result.shows == "unclear":
        return 0.3
    return round(0.4 + 0.5 * result.confidence, 2)


def vision_node(state: ClaimState, deps: Deps) -> dict:
    ev = state.get("evidence", {})
    meta = ev.get("photo") if ev.get("available") else None
    if meta is None:
        return {"vision": None}
    vision = {"metadata": meta, "content_checked": False, "image_risk": meta["risk"]}
    errors = list(state.get("errors", []))
    path = state.get("photo_path")
    if deps.use_vision and path and Path(path).exists():
        reason = state.get("intake", {}).get("reason_category", "other")
        prompt = f"The rider says the delay was caused by {CLAIM_WORDS.get(reason, reason)}. Does this photo show that?"
        deps.llm.unload(TEXT_MODEL)              # free RAM before loading the vision model
        try:
            res = deps.llm.chat_json(VISION_MODEL, SYSTEM, prompt, VisionResult, images=[Path(path)])
            risk = content_risk(res)
            vision.update(content_checked=True, **res.model_dump(), content_risk=risk,
                          image_risk=max(meta["risk"], risk))
        except LLMError as exc:
            errors.append(f"vision: {exc}; used metadata checks only")
        finally:
            deps.llm.unload(VISION_MODEL)
    if not deps.dry_run:
        with deps.engine.begin() as conn:
            conn.execute(text("UPDATE claim_evidence SET vision_result = CAST(:v AS JSONB) "
                              "WHERE claim_id = :c AND kind = 'photo'"),
                         {"v": json.dumps(vision, default=str), "c": int(state["claim_id"])})
    return {"vision": vision, "errors": errors}
