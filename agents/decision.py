"""Decision agent: rules + score thresholds, then an LLM-written explanation.

The decision itself is deterministic:
  1. Hard rules  -> no matching delivery: human_review; duplicate claim: auto_reject
  2. Thresholds  -> score below approve threshold: auto_approve; at/above reject
                    threshold: auto_reject; otherwise human_review
  3. Caution     -> an auto_approve is downgraded to human_review if the photo
                    looks reused or the rider is over the weekly claim limit

The LLM only writes the explanation, from a numbered list of facts, and must
say which facts it used. If it fails or contradicts the decision, a template
explanation is used instead. Payouts come from policy/payout.py.
"""
from __future__ import annotations

from agents.config import TEXT_MODEL
from agents.deps import Deps
from agents.llm import LLMError
from agents.schemas import ClaimState, Explanation
from ml.claim_scoring import decide
from policy.payout import Policy, calculate_payout

SYSTEM = """You explain insurance decisions on delivery-delay claims to riders and reviewers.
Write 2-4 short, plain-English sentences. Use ONLY the numbered facts given; do not invent numbers.
State the decision first, then the main reasons. Be respectful to the rider.
Do not confuse the total delivery time with the delay (how many minutes late it was).
facts_cited must list the numbers of the facts you used."""

LABELS = {"auto_approve": "approved automatically", "auto_reject": "rejected automatically",
          "human_review": "sent to a human reviewer"}

SIGNAL_TEXT = {
    "condition_support": "how much of the delay the conditions explain",
    "claim_inflation": "whether the claimed delay matches the real delay",
    "zone_disruption": "whether other riders in the zone were delayed",
    "rider_anomaly": "how unusual the rider's behaviour is",
    "image_check": "the photo checks",
    "claim_history": "the rider's recent claim history",
}


def apply_rules(state: ClaimState, deps: Deps) -> tuple[str, list[str]]:
    ev, risk = state.get("evidence", {}), state.get("risk", {})
    if not ev.get("available"):
        return "human_review", ["could not match the claim to one of the rider's deliveries"]
    if ev.get("duplicate_claims", 0) > 0:
        return "auto_reject", ["a claim already exists for this delivery"]
    decision = decide(risk["score"], deps.models.cfg)
    rules = []
    if decision == "auto_approve":
        vision = state.get("vision") or {}
        photo = vision.get("metadata") or ev.get("photo") or {}
        if photo.get("reused"):
            rules.append("the photo matches a previously submitted photo")
        if vision.get("content_checked") and vision.get("content_risk", 0) >= 0.6:
            rules.append("the photo does not show the claimed situation")
        cap = int(ev["policy"]["max_claims_per_week"])
        if ev["rider"]["claims_last_7d"] >= cap:
            rules.append(f"rider already filed {ev['rider']['claims_last_7d']} claims this week (limit {cap})")
        if rules:
            decision = "human_review"
    return decision, rules


def build_facts(state: ClaimState, decision: str, rules: list[str], payout: dict | None) -> list[str]:
    ev, risk = state.get("evidence", {}), state.get("risk", {})
    facts = [f"Decision: the claim was {LABELS[decision]}."]
    if not ev.get("available"):
        return facts + [f"Reason: {r}." for r in rules]
    th = risk["thresholds"]
    facts.append(f"Risk score {risk['score']:.2f} on a 0-1 scale (auto-approve below {th['auto_approve_below']}, "
                 f"auto-reject from {th['auto_reject_at_or_above']}).")
    d, c, z, inp = ev["delivery"], ev["conditions"], ev["zone_disruption"], ev["inputs"]
    facts.append(f"Conditions in {ev['zone_name']} at pickup: {c['event_type'].replace('_', ' ')}, "
                 f"rain {c['rain_mm']:.1f} mm/h, traffic level {c['traffic_level']:.2f} (0-1).")
    facts.append(f"The delivery took {d['actual_minutes']:.0f} min; a normal day would take about "
                 f"{d['eta_clear_minutes']:.0f} min, so it was {inp['observed_delay']:.0f} min late. "
                 f"Conditions explain about {inp['condition_delay']:.0f} min.")
    facts.append(f"The rider claimed a {inp['claimed_delay']:.0f} min delay; GPS shows {d['idle_minutes']:.0f} min idle "
                 f"and {d['restaurant_wait_minutes']:.0f} min restaurant wait.")
    if z["n_deliveries"] >= 4:
        facts.append(f"{z['n_delayed']} of {z['n_deliveries']} other deliveries in the zone around that time were "
                     f"delayed (usually {z['baseline_share']:.0%}).")
    else:
        facts.append("There were too few other deliveries in the zone at that time to compare "
                     "(this neither supports nor weakens the claim).")
    r = ev["rider"]
    facts.append(f"Rider behaviour is more unusual than {r['anomaly_percentile']:.0%} of riders; "
                 f"{inp['claims_last_30d']} claims in the previous 30 days."
                 + (f" Notable: {'; '.join(r['notes'])}." if r["notes"] else ""))
    vision = state.get("vision") or {}
    photo = vision.get("metadata") or ev.get("photo")
    if photo:
        facts.append("Photo checks: " + ("; ".join(photo["reasons"]) if photo["reasons"] else "no problems found") + ".")
    if vision.get("content_checked"):
        facts.append(f"Photo content: {vision['description']} "
                     f"({'matches' if vision['matches_claim'] else 'does not match'} the claim).")
    if decision != "auto_approve":
        # Only meaningful when the claim was not approved; small models misread it otherwise.
        top = [SIGNAL_TEXT[s] for s in risk["top_signals"]]
        facts.append("Factors that counted AGAINST the claim the most: " + ", ".join(top) + ".")
    facts += [f"Rule applied: {r}." for r in rules]
    if payout and payout["amount_inr"] > 0:
        facts.append(f"Payout: Rs {payout['amount_inr']:.0f} ({payout['formula']}).")
    return facts


def template_explanation(decision: str, facts: list[str]) -> str:
    """Deterministic fallback: the decision plus the facts that matter for it."""
    def having(*starts):
        return [f for f in facts[1:] if f.startswith(starts)]

    if decision == "auto_approve":
        picked = having("Conditions", "The delivery took") + [f for f in facts if "other deliveries" in f] + having("Payout")
    elif decision == "auto_reject":
        picked = having("Rule", "The delivery took", "The rider claimed", "Factors")
    else:
        picked = having("Reason", "Rule", "Risk score", "Factors")
    return " ".join([facts[0]] + picked)


def _contradicts(text: str, decision: str) -> bool:
    low = text.lower()
    if decision == "auto_approve":
        return "reject" in low or "denied" in low
    if decision == "auto_reject":
        return " approved" in low and "not approved" not in low
    return False


def decision_node(state: ClaimState, deps: Deps) -> dict:
    decision, rules = apply_rules(state, deps)
    ev = state.get("evidence", {})
    payout = None
    if ev.get("available"):
        p = ev["policy"]
        policy = Policy(**{k: p[k] for k in Policy.__dataclass_fields__})
        inp = ev["inputs"]
        payout = calculate_payout(policy, inp["claimed_delay"], inp["observed_delay"],
                                  deps.models.cfg.params["tolerance_minutes"]).to_dict()
    facts = build_facts(state, decision, rules, payout if decision == "auto_approve" else None)

    errors = list(state.get("errors", []))
    numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))
    try:
        if not getattr(deps, "llm_explanations", True):
            raise LLMError("LLM explanations disabled")
        out = deps.llm.chat_json(TEXT_MODEL, SYSTEM, f"Facts:\n{numbered}", Explanation)
        cited = [i for i in out.facts_cited if 1 <= i <= len(facts)]
        if _contradicts(out.explanation, decision) or not cited:
            raise LLMError("explanation contradicted the decision or cited no facts")
        explanation, source = out.explanation.strip(), "llm"
    except LLMError as exc:
        errors.append(f"decision explanation: {exc}; used template")
        explanation, cited, source = template_explanation(decision, facts), list(range(1, len(facts) + 1)), "template"

    return {
        "decision": {"decision": decision, "rules": rules, "facts": facts, "facts_cited": cited,
                     "explanation": explanation, "explanation_source": source},
        "payout": payout if decision == "auto_approve" else None,
        "proposed_payout": payout,
        "errors": errors,
    }
