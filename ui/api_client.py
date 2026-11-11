"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI backend."""
from __future__ import annotations

import os

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


class ApiClient:
    def __init__(self, base_url: str = API_URL, timeout: float = 30):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def _json(self, resp: httpx.Response):
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = resp.text
            raise RuntimeError(f"{resp.status_code}: {detail}")
        return resp.json()

    def health(self) -> dict:
        return self._json(self.http.get("/health"))

    def deliveries(self, rider_id: int, limit: int = 15) -> list[dict]:
        return self._json(self.http.get(f"/riders/{rider_id}/deliveries", params={"limit": limit}))

    def submit_claim(self, rider_id: int, text: str, delivery_id: int | None = None,
                     photo: tuple[str, bytes, str] | None = None) -> dict:
        data = {"rider_id": str(rider_id), "text": text}
        if delivery_id is not None:
            data["delivery_id"] = str(delivery_id)
        files = {"photo": photo} if photo else None
        return self._json(self.http.post("/claims", data=data, files=files))

    def claim(self, claim_id: int) -> dict:
        return self._json(self.http.get(f"/claims/{claim_id}"))

    def photo(self, claim_id: int) -> bytes | None:
        resp = self.http.get(f"/claims/{claim_id}/photo")
        return resp.content if resp.status_code == 200 else None

    def review_queue(self) -> list[dict]:
        return self._json(self.http.get("/review/queue"))

    def review(self, claim_id: int, verdict: str, reviewer: str, notes: str | None) -> dict:
        return self._json(self.http.post(f"/review/{claim_id}",
                                         json={"verdict": verdict, "reviewer": reviewer, "notes": notes}))

    def stats(self) -> dict:
        return self._json(self.http.get("/stats"))
