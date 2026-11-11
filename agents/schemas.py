"""Pydantic schemas for LLM outputs and the graph state."""
from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

ReasonCategory = Literal["heavy_rain", "flooding", "traffic_jam", "road_closure", "restaurant_delay", "other"]


class IntakeResult(BaseModel):
    """What the intake agent extracts from the rider's free text."""
    delivery_id: int | None = Field(None, description="order/delivery number if the rider mentions one, else null")
    reason_category: ReasonCategory = Field(description="main cause the rider blames")
    claimed_delay_minutes: float | None = Field(None, ge=0, le=600, description="delay in minutes if stated, else null")
    incident_time: str | None = Field(None, description="time of the incident as HH:MM if stated, else null")
    summary: str = Field(description="one short sentence restating the claim")


class Explanation(BaseModel):
    """Decision explanation written by the LLM."""
    explanation: str = Field(description="2-4 plain-English sentences for the rider and reviewer")
    facts_cited: list[int] = Field(description="numbers of the facts the explanation relies on")


class VisionResult(BaseModel):
    """What the vision model says about the attached photo."""
    shows: Literal["flooded_road", "heavy_rain", "traffic_jam", "accident", "closed_road", "restaurant", "other", "unclear"]
    matches_claim: bool = Field(description="does the photo plausibly show the claimed situation?")
    confidence: float = Field(ge=0, le=1)
    description: str = Field(description="one short sentence describing the photo")


class ClaimState(TypedDict, total=False):
    """Everything the graph knows about a claim. Kept JSON-friendly for the checkpointer."""
    claim_id: int
    rider_id: int
    text: str
    submitted_at: str
    delivery_id: int | None
    photo_path: str | None
    intake: dict
    evidence: dict
    vision: dict | None
    risk: dict
    decision: dict
    payout: dict | None
    proposed_payout: dict | None
    review: dict | None
    status: str
    errors: list[str]
