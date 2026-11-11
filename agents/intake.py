"""Intake agent: rider's free text -> structured claim fields.

The LLM extracts the order number, reason, delay and time. Deterministic code
then resolves the delivery and guards against mistakes:
  * an order number the LLM "found" that is not in the text is discarded,
  * the delivery must belong to the claiming rider,
  * with no usable number we take the rider's latest delivery before the claim.
If the LLM fails twice, a keyword/regex fallback produces the same fields.
"""
from __future__ import annotations

import re

import pandas as pd

from agents.config import TEXT_MODEL
from agents.deps import Deps
from agents.llm import LLMError
from agents.schemas import ClaimState, IntakeResult
from db import queries

SYSTEM = """You read delay claims from food delivery riders in India and extract fields as JSON.
reason_category must be one of: heavy_rain, flooding, traffic_jam, road_closure, restaurant_delay, other.
- flooding: waterlogged/flooded roads or underpasses. heavy_rain: rain itself slowed them.
- traffic_jam: jams, heavy traffic, signals. road_closure: blocked/closed roads, barricades, detours.
- restaurant_delay: food not ready, waiting at the restaurant.
delivery_id: the order or delivery number ONLY if the text contains it, otherwise null.
claimed_delay_minutes: the delay in minutes ONLY if stated, otherwise null.
incident_time: HH:MM only if a time is stated, otherwise null.
summary: one short neutral sentence."""

KEYWORDS = [
    ("flooding", ["flood", "waterlog", "water log", "water-log", "water up", "water till", "underpass", "knee"]),
    ("road_closure", ["closed", "closure", "blocked", "barricade", "detour", "police"]),
    ("restaurant_delay", ["restaurant", "not ready", "food was", "pickup", "waited"]),
    ("traffic_jam", ["traffic", "jam", "signal", "stuck"]),
    ("heavy_rain", ["rain", "pouring", "storm", "slippery"]),
]


def fallback_intake(text: str) -> IntakeResult:
    """Keyword/regex extraction used when the LLM is unavailable or fails."""
    low = text.lower()
    m_id = re.search(r"(?:order|delivery)\s*(?:#|no\.?|number)?\s*#?(\d{1,7})", low)
    m_min = re.search(r"(\d{1,3})\s*(?:min|mins|minutes)\b", low)
    m_time = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", low)
    reason = next((cat for cat, words in KEYWORDS if any(w in low for w in words)), "other")
    return IntakeResult(
        delivery_id=int(m_id.group(1)) if m_id else None,
        reason_category=reason,
        claimed_delay_minutes=float(m_min.group(1)) if m_min else None,
        incident_time=f"{int(m_time.group(1)):02d}:{m_time.group(2)}" if m_time else None,
        summary=" ".join(text.split())[:160],
    )


def _number_in_text(n: int, text: str) -> bool:
    return re.search(rf"(?<!\d){n}(?!\d)", text) is not None


def intake_node(state: ClaimState, deps: Deps) -> dict:
    text = state["text"]
    errors = list(state.get("errors", []))
    try:
        result = deps.llm.chat_json(TEXT_MODEL, SYSTEM, text, IntakeResult)
        source = "llm"
        if result.delivery_id is not None and not _number_in_text(result.delivery_id, text):
            result.delivery_id = None          # LLM invented a number
        # Small models sometimes leave explicit numbers out (seen with qwen2.5:3b:
        # "Order no 43099 ... around 25 minutes" -> both null). Numbers written in
        # the text are extracted reliably by regex, so fill any gaps from it.
        rx = fallback_intake(text)
        filled = []
        for field in ("delivery_id", "claimed_delay_minutes", "incident_time"):
            if getattr(result, field) is None and getattr(rx, field) is not None:
                setattr(result, field, getattr(rx, field))
                filled.append(field)
        if filled:
            source = "llm+regex:" + ",".join(filled)
    except LLMError as exc:
        errors.append(f"intake: {exc}; used keyword fallback")
        result, source = fallback_intake(text), "fallback"

    rider_id = int(state["rider_id"])
    submitted = pd.Timestamp(state["submitted_at"]).to_pydatetime()
    resolution, delivery_id = "not_found", None
    for candidate, how in ((state.get("delivery_id"), "given"), (result.delivery_id, "from_text")):
        if candidate is None:
            continue
        d = queries.get_delivery(deps.engine, int(candidate))
        if d.empty:
            continue
        if int(d.iloc[0]["rider_id"]) != rider_id:
            resolution = "belongs_to_other_rider"
            continue
        delivery_id, resolution = int(candidate), how
        break
    if delivery_id is None and resolution != "belongs_to_other_rider":
        delivery_id = queries.latest_delivery_before(deps.engine, rider_id, submitted)
        resolution = "latest_delivery" if delivery_id else "not_found"

    intake = {**result.model_dump(), "source": source, "resolved_delivery_id": delivery_id, "resolution": resolution}
    return {"intake": intake, "delivery_id": delivery_id, "errors": errors}
