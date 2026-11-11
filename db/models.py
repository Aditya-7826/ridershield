"""SQLAlchemy ORM models for RiderShield.

Table overview
--------------
zones          city areas; some are flood-prone
riders         delivery riders
policies       payout rules (amounts are applied by policy/ code, never the LLM)
conditions     one row per zone per hour: rain, traffic and any disruption event
deliveries     individual deliveries with timing and GPS-derived idle time
claims         delay claims filed by riders (+ ground-truth label for evaluation)
claim_evidence photos and other attachments for a claim
decisions      automated decision, risk score and its breakdown
reviews        human reviewer verdicts for claims sent to review
payouts        money paid on approved claims
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Shared vocabularies. Kept as plain strings + CHECK constraints so the
# migrations stay simple and the values are readable in SQL.
EVENT_TYPES = ("none", "rain", "heavy_rain", "flood", "traffic_jam", "road_closure")
REASON_CATEGORIES = ("heavy_rain", "flooding", "traffic_jam", "road_closure", "restaurant_delay", "other")
CLAIM_LABELS = ("genuine_external", "rider_caused", "fraudulent")
CLAIM_STATUSES = ("pending", "auto_approved", "auto_rejected", "in_review", "approved", "rejected")
DECISIONS = ("auto_approve", "auto_reject", "human_review")


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


class Base(DeclarativeBase):
    pass


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    flood_prone: Mapped[bool] = mapped_column(Boolean, default=False)
    base_traffic: Mapped[float] = mapped_column(Float, comment="typical congestion offset, 0-1")


class Rider(Base):
    __tablename__ = "riders"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    home_zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"))
    vehicle_type: Mapped[str] = mapped_column(String(16))
    joined_at: Mapped[datetime] = mapped_column(DateTime)
    rating: Mapped[float] = mapped_column(Float)


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    base_payout_inr: Mapped[float] = mapped_column(Float)
    per_minute_inr: Mapped[float] = mapped_column(Float)
    max_payout_inr: Mapped[float] = mapped_column(Float)
    min_delay_minutes: Mapped[int] = mapped_column(Integer)
    max_claims_per_week: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Condition(Base):
    __tablename__ = "conditions"
    __table_args__ = (
        UniqueConstraint("zone_id", "hour_start", name="uq_conditions_zone_hour"),
        CheckConstraint(_in("event_type", EVENT_TYPES), name="ck_conditions_event_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"))
    hour_start: Mapped[datetime] = mapped_column(DateTime)
    rain_mm: Mapped[float] = mapped_column(Float, comment="rainfall in this hour")
    traffic_level: Mapped[float] = mapped_column(Float, comment="0 = empty roads, 1 = gridlock")
    event_type: Mapped[str] = mapped_column(String(16), default="none")
    severity: Mapped[float] = mapped_column(Float, default=0.0, comment="0-1")


class Delivery(Base):
    __tablename__ = "deliveries"
    __table_args__ = (Index("ix_deliveries_zone_time", "zone_id", "picked_up_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    rider_id: Mapped[int] = mapped_column(ForeignKey("riders.id"), index=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id"))
    picked_up_at: Mapped[datetime] = mapped_column(DateTime)
    distance_km: Mapped[float] = mapped_column(Float)
    restaurant_wait_minutes: Mapped[float] = mapped_column(Float)
    idle_minutes: Mapped[float] = mapped_column(Float, comment="long GPS stops not at pickup/drop")
    avg_speed_kmh: Mapped[float] = mapped_column(Float, comment="speed while moving")
    actual_minutes: Mapped[float] = mapped_column(Float, comment="order accepted to delivered")


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint(_in("reason_category", REASON_CATEGORIES), name="ck_claims_reason"),
        CheckConstraint(f"true_label IS NULL OR {_in('true_label', CLAIM_LABELS)}", name="ck_claims_label"),
        CheckConstraint(_in("status", CLAIM_STATUSES), name="ck_claims_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    delivery_id: Mapped[int | None] = mapped_column(ForeignKey("deliveries.id"), index=True)
    rider_id: Mapped[int] = mapped_column(ForeignKey("riders.id"), index=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime)
    text: Mapped[str] = mapped_column(Text, comment="rider's free-text claim")
    reason_category: Mapped[str] = mapped_column(String(24))
    claimed_delay_minutes: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # Ground truth from the synthetic generator. Used ONLY for evaluation,
    # never as a model feature. NULL for claims submitted through the app.
    true_label: Mapped[str | None] = mapped_column(String(24))

    evidence: Mapped[list[ClaimEvidence]] = relationship(back_populates="claim")


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="photo")
    file_path: Mapped[str | None] = mapped_column(String(255))
    phash: Mapped[str | None] = mapped_column(String(32), index=True, comment="perceptual hash, hex")
    captured_at: Mapped[datetime | None] = mapped_column(DateTime, comment="EXIF timestamp if present")
    gps_lat: Mapped[float | None] = mapped_column(Float)
    gps_lon: Mapped[float | None] = mapped_column(Float)
    vision_result: Mapped[dict | None] = mapped_column(JSONB)

    claim: Mapped[Claim] = relationship(back_populates="evidence")


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (CheckConstraint(_in("decision", DECISIONS), name="ck_decisions_decision"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True)
    decision: Mapped[str] = mapped_column(String(16))
    risk_score: Mapped[float | None] = mapped_column(Float, comment="NULL when no evidence was available")
    breakdown: Mapped[dict] = mapped_column(JSONB, comment="per-signal contributions to the risk score")
    explanation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (CheckConstraint("verdict IN ('approve', 'reject')", name="ck_reviews_verdict"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(8))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Payout(Base):
    __tablename__ = "payouts"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), unique=True)
    policy_id: Mapped[int] = mapped_column(ForeignKey("policies.id"))
    amount_inr: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
