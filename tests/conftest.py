"""Shared fixtures: database availability, trained models and a fake LLM."""
from __future__ import annotations

import pytest

from agents.llm import LLMError
from agents.schemas import Explanation, IntakeResult, VisionResult
from ml.claim_scoring import ANOMALY_PATH, ETA_PATH, ZONE_PATH


class FakeLLM:
    """Returns canned objects per schema; set `fail=True` to simulate an outage."""

    def __init__(self, intake: IntakeResult | None = None, explanation: str = "The claim was processed.",
                 vision: VisionResult | None = None, fail: bool = False):
        self.intake = intake
        self.explanation = explanation
        self.vision = vision
        self.fail = fail
        self.calls: list[str] = []

    def chat_json(self, model, system, user, schema, images=None):
        self.calls.append(schema.__name__)
        if self.fail:
            raise LLMError("fake outage")
        if schema is IntakeResult:
            return self.intake or IntakeResult(reason_category="heavy_rain", summary="rain delay")
        if schema is Explanation:
            return Explanation(explanation=self.explanation, facts_cited=[1, 2])
        if schema is VisionResult:
            if self.vision is None:
                raise LLMError("no vision result configured")
            return self.vision
        raise AssertionError(f"unexpected schema {schema}")

    def unload(self, model):
        pass


def _db_available() -> bool:
    try:
        from sqlalchemy import text

        from db.session import get_engine
        with get_engine().connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM deliveries")).scalar_one() > 0
    except Exception:
        return False


requires_db = pytest.mark.skipif(not _db_available(), reason="PostgreSQL with seeded data not available")
requires_models = pytest.mark.skipif(
    not all(p.exists() for p in (ETA_PATH, ZONE_PATH, ANOMALY_PATH)),
    reason="trained models not found (run scripts/train.py --model all)",
)
