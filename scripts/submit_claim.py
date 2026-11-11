"""Submit a claim from the command line and print the decision.

Examples:
    # a new claim (text is parsed by the intake agent)
    python scripts/submit_claim.py --rider 42 --delivery-id 45210 \
        --text "Road near Bellandur was flooded, order 45210 was 40 min late"

    # re-run an existing seeded claim through the agents
    python scripts/submit_claim.py --claim-id 1800

    # pick a held-out seeded claim with a given ground-truth label
    python scripts/submit_claim.py --sample genuine_external

    # no Ollama? use the deterministic fallbacks only
    python scripts/submit_claim.py --sample fraudulent --no-llm
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.deps import Deps  # noqa: E402
from agents.graph import build_graph, open_checkpointer, process_claim  # noqa: E402
from agents.llm import LLMError, OllamaClient  # noqa: E402
from agents.service import create_claim  # noqa: E402
from db import queries  # noqa: E402
from db.session import get_engine  # noqa: E402
from ml.claim_scoring import Models  # noqa: E402
from ml.config import TRAIN_CUTOFF  # noqa: E402


class OfflineLLM:
    """Always fails, so every agent uses its deterministic fallback."""

    def chat_json(self, *args, **kwargs):
        raise LLMError("LLM disabled (--no-llm)")

    def unload(self, model):  # noqa: D401
        pass


def pick_sample(engine, label: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text("""
            SELECT c.id FROM claims c JOIN deliveries d ON d.id = c.delivery_id
            WHERE c.true_label = :l AND d.picked_up_at >= :t ORDER BY c.id LIMIT 1
        """), {"l": label, "t": TRAIN_CUTOFF}).scalar_one())


def print_result(state: dict, seconds: float, true_label: str | None) -> None:
    d, risk = state["decision"], state.get("risk", {})
    print("\n" + "=" * 72)
    print(f"Claim {state['claim_id']}  rider {state['rider_id']}  ->  {d['decision'].upper()}  (status: {state['status']})")
    if true_label:
        print(f"Ground-truth label (synthetic): {true_label}")
    print("=" * 72)
    print(f"Text: {state['text']}")
    i = state.get("intake", {})
    print(f"Intake ({i.get('source')}): delivery {i.get('resolved_delivery_id')} [{i.get('resolution')}], "
          f"reason {i.get('reason_category')}, claimed {i.get('claimed_delay_minutes')} min")
    if risk.get("available"):
        rows = [{"signal": k, "value": b["value"], "weight": b["weight"], "contribution": b["contribution"]}
                for k, b in risk["breakdown"].items()]
        print(f"\nRisk score: {risk['score']:.3f}  (approve < {risk['thresholds']['auto_approve_below']}, "
              f"reject >= {risk['thresholds']['auto_reject_at_or_above']})")
        print(pd.DataFrame(rows).to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.3f}"))
    if d["rules"]:
        print("\nRules applied: " + "; ".join(d["rules"]))
    print(f"\nExplanation ({d['explanation_source']}):\n  {d['explanation']}")
    print(f"Facts cited: {d['facts_cited']}")
    if state.get("payout"):
        print(f"\nPayout: Rs {state['payout']['amount_inr']:.0f}  ({state['payout']['formula']})")
    if state.get("errors"):
        print("\nNotes: " + " | ".join(state["errors"]))
    print(f"\nProcessed in {seconds:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="claim text for a NEW claim (needs --rider)")
    src.add_argument("--claim-id", type=int, help="process an existing claim")
    src.add_argument("--sample", choices=["genuine_external", "rider_caused", "fraudulent"])
    ap.add_argument("--rider", type=int)
    ap.add_argument("--delivery-id", type=int)
    ap.add_argument("--submitted-at", help="ISO time; default = 20 min after the delivery, or now")
    ap.add_argument("--no-llm", action="store_true", help="skip Ollama and use deterministic fallbacks")
    args = ap.parse_args()

    engine = get_engine()
    if args.text:
        if args.rider is None:
            ap.error("--text needs --rider")
        if args.submitted_at:
            submitted = datetime.fromisoformat(args.submitted_at)
        elif args.delivery_id:
            d = queries.get_delivery(engine, args.delivery_id).iloc[0]
            submitted = (pd.Timestamp(d["picked_up_at"]) + timedelta(minutes=float(d["actual_minutes"]) + 20)).to_pydatetime()
        else:
            submitted = datetime.now()
        claim_id = create_claim(engine, args.rider, args.text, submitted, args.delivery_id)
        print(f"Created claim {claim_id}")
    else:
        claim_id = args.claim_id or pick_sample(engine, args.sample)

    llm = OfflineLLM() if args.no_llm else OllamaClient()
    deps = Deps(engine=engine, llm=llm, models=Models.load())
    graph = build_graph(deps, checkpointer=open_checkpointer())
    t0 = time.perf_counter()
    state = process_claim(graph, deps, claim_id)
    true_label = (queries.get_claim(engine, claim_id) or {}).get("true_label")
    print_result(state, time.perf_counter() - t0, true_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
