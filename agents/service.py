"""Create new claims (used by the CLI and, later, the API)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Engine, text


def create_claim(engine: Engine, rider_id: int, claim_text: str, submitted_at: datetime,
                 delivery_id: int | None = None) -> int:
    """Insert a claim with placeholder fields; the intake agent fills them in.

    true_label stays NULL: only synthetic claims have ground truth.
    """
    with engine.begin() as conn:
        claim_id = conn.execute(text("""
            INSERT INTO claims (delivery_id, rider_id, submitted_at, text, reason_category,
                                claimed_delay_minutes, status, true_label)
            VALUES (:d, :r, :t, :x, 'other', 0, 'pending', NULL)
            RETURNING id
        """), {"d": delivery_id, "r": rider_id, "t": submitted_at, "x": claim_text}).scalar_one()
    return int(claim_id)
