"""API tests with FastAPI's TestClient, a fake LLM and the real database."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text

from agents.schemas import IntakeResult, VisionResult
from ml.photo_io import write_exif
from tests.conftest import FakeLLM, requires_db, requires_models
from tests.test_vision_review import _flood_deliveries

FIXTURES = Path(__file__).resolve().parent / "fixtures"
pytestmark = [requires_db, requires_models]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from agents import config
    from agents.deps import Deps
    from app.main import create_app
    from db.session import get_engine
    from ml.claim_scoring import Models

    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    engine = get_engine()
    with engine.connect() as conn:
        max_id = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM claims")).scalar_one()
    llm = FakeLLM(intake=IntakeResult(reason_category="flooding", summary="flood"),
                  vision=VisionResult(shows="flooded_road", matches_claim=True, confidence=0.9, description="flooded road"),
                  explanation="The claim was processed using the listed facts.")
    deps = Deps(engine=engine, llm=llm, models=Models.load())
    with TestClient(create_app(deps, checkpoint_path=tmp_path / "cp.sqlite")) as c:
        yield c, engine
    with engine.begin() as conn:
        new = "(SELECT id FROM claims WHERE id > :m AND true_label IS NULL)"
        for table in ("reviews", "payouts", "decisions", "claim_evidence"):
            conn.execute(text(f"DELETE FROM {table} WHERE claim_id IN {new}"), {"m": max_id})
        conn.execute(text("DELETE FROM claims WHERE id > :m AND true_label IS NULL"), {"m": max_id})


def _exif_photo(tmp_path: Path, row) -> Path:
    p = tmp_path / f"photo_{row.id}.jpg"
    write_exif(Image.open(FIXTURES / "flooded_road.jpg"), p,
               (pd.Timestamp(row.picked_up_at) + timedelta(minutes=10)).to_pydatetime(), (row.lat, row.lon))
    return p


def test_health(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["database"] is True and body["status"] == "ok"


def test_root_redirects_to_docs(client):
    c, _ = client
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_submit_with_photo_then_reused_photo_goes_to_review_and_is_approved(client, tmp_path):
    c, engine = client
    a, b = _flood_deliveries(engine, 2)

    # rider A: genuine claim with a photo
    with open(_exif_photo(tmp_path, a), "rb") as f:
        r = c.post("/claims", data={"rider_id": a.rider_id, "delivery_id": a.id,
                                    "text": f"Road flooded, order {a.id} was 20 min late"},
                   files={"photo": ("flood.jpg", f, "image/jpeg")})
    assert r.status_code == 202
    claim_a = r.json()["claim_id"]
    body = c.get(f"/claims/{claim_a}").json()
    assert body["job"] == "done"
    assert body["decision"] is not None and body["risk"]["available"]
    assert body["has_photo"] and body["vision"]["content_checked"]
    assert c.get(f"/claims/{claim_a}/photo").status_code == 200

    # rider B: re-uses a resized copy of the same photo
    with open(FIXTURES / "flooded_road_copy.jpg", "rb") as f:
        r = c.post("/claims", data={"rider_id": b.rider_id, "delivery_id": b.id,
                                    "text": f"Flooded road, order {b.id} 20 min late"},
                   files={"photo": ("copy.jpg", f, "image/jpeg")})
    claim_b = r.json()["claim_id"]
    body = c.get(f"/claims/{claim_b}").json()
    assert body["vision"]["metadata"]["reused"] is True
    assert body["claim"]["status"] == "in_review" and body["awaiting_review"] is True

    queue = c.get("/review/queue").json()
    assert claim_b in [q["claim_id"] for q in queue]

    r = c.post(f"/review/{claim_b}", json={"verdict": "approve", "reviewer": "asha", "notes": "checked"})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert c.get(f"/claims/{claim_b}").json()["review"]["reviewer"] == "asha"
    # a second verdict is refused
    assert c.post(f"/review/{claim_b}", json={"verdict": "reject", "reviewer": "x"}).status_code == 409

    stats = c.get("/stats").json()
    assert stats["decided"] >= 2 and 0 <= stats["automation_rate"] <= 1


def test_validation_errors(client):
    c, _ = client
    assert c.post("/claims", data={"rider_id": 999999, "text": "some delay text"}).status_code == 404
    assert c.post("/claims", data={"rider_id": 1, "text": "x"}).status_code == 422
    r = c.post("/claims", data={"rider_id": 1, "text": "flooded road"},
               files={"photo": ("a.txt", b"not an image", "text/plain")})
    assert r.status_code == 415
    assert c.get("/claims/99999999").status_code == 404
    assert c.post("/review/1", json={"verdict": "maybe", "reviewer": "x"}).status_code == 422


def test_rider_deliveries(client):
    c, _ = client
    rows = c.get("/riders/1/deliveries", params={"limit": 3}).json()
    assert 1 <= len(rows) <= 3 and {"id", "zone", "picked_up_at", "claims"} <= set(rows[0])
