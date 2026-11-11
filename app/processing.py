"""Runs claims through the agent graph in the background, one at a time.

Local LLM inference on a CPU can take minutes, so the API answers right away
(202 Accepted) and the claim is processed afterwards. A lock makes sure only
one claim uses the models at a time, which keeps memory use predictable on a
low-RAM laptop. Job status is kept in memory for the UI to poll.
"""
from __future__ import annotations

import logging
import threading
import traceback

from agents.deps import Deps
from agents.graph import process_claim, resume_review

log = logging.getLogger(__name__)


class ClaimProcessor:
    def __init__(self, deps: Deps, graph):
        self.deps = deps
        self.graph = graph
        self._lock = threading.Lock()
        self.jobs: dict[int, str] = {}      # claim_id -> queued | running | done | error: ...

    def enqueue(self, claim_id: int) -> None:
        self.jobs[claim_id] = "queued"

    def run(self, claim_id: int) -> None:
        with self._lock:
            self.jobs[claim_id] = "running"
            try:
                process_claim(self.graph, self.deps, claim_id)
                self.jobs[claim_id] = "done"
            except Exception as exc:   # keep the server alive; report the error to the UI
                log.error("claim %s failed: %s", claim_id, traceback.format_exc())
                self.jobs[claim_id] = f"error: {exc}"

    def review(self, claim_id: int, verdict: str, reviewer: str, notes: str | None) -> dict:
        with self._lock:
            return resume_review(self.graph, claim_id, verdict, reviewer, notes)
