"""Reviewer CLI: list claims waiting for review, or approve/reject one.

    python scripts/review_claim.py --list
    python scripts/review_claim.py --claim-id 2060 --verdict approve --reviewer asha --notes "photo is old but delay is real"

Resuming works after a restart: the paused graph state is read back from the
SQLite checkpoint file (CHECKPOINT_PATH).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.deps import Deps  # noqa: E402
from agents.graph import build_graph, is_paused_for_review, open_checkpointer, resume_review, thread_config  # noqa: E402
from db.session import get_engine  # noqa: E402
from ml.claim_scoring import Models  # noqa: E402
from agents.llm import OfflineLLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--claim-id", type=int)
    ap.add_argument("--verdict", choices=["approve", "reject"])
    ap.add_argument("--reviewer")
    ap.add_argument("--notes")
    args = ap.parse_args()

    engine = get_engine()
    # Resuming needs no LLM: the review node only records the verdict.
    deps = Deps(engine=engine, llm=OfflineLLM(), models=Models.load())
    graph = build_graph(deps, checkpointer=open_checkpointer())

    if args.list:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT id, rider_id, submitted_at, left(text, 60) FROM claims "
                                     "WHERE status = 'in_review' ORDER BY submitted_at")).all()
        for r in rows:
            paused = is_paused_for_review(graph, r.id)
            snap = graph.get_state(thread_config(r.id)).values if paused else {}
            score = snap.get("risk", {}).get("score")
            print(f"claim {r.id:>5}  rider {r.rider_id:>4}  score {score if score is not None else '-'}  "
                  f"{'paused' if paused else 'no checkpoint'}  {r[3]}")
        print(f"{len(rows)} claim(s) in review")
        return 0

    if not (args.claim_id and args.verdict and args.reviewer):
        ap.error("--claim-id, --verdict and --reviewer are required (or use --list)")
    state = resume_review(graph, args.claim_id, args.verdict, args.reviewer, args.notes)
    print(f"Claim {args.claim_id}: {state['status']} by {state['review']['reviewer']}")
    if state.get("payout"):
        print(f"Payout: Rs {state['payout']['amount_inr']:.0f} ({state['payout']['formula']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
