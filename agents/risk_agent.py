"""Risk agent: turn the evidence into the weighted risk score (no LLM)."""
from __future__ import annotations

from agents.deps import Deps
from agents.schemas import ClaimState
from ml.risk import ClaimInputs, combine, compute_signals


def risk_node(state: ClaimState, deps: Deps) -> dict:
    ev = state.get("evidence", {})
    if not ev.get("available"):
        return {"risk": {"available": False}}
    inputs = ClaimInputs(**ev["inputs"])
    vision = state.get("vision")
    if vision and vision.get("image_risk") is not None:
        # Vision agent combines metadata checks with the photo's content.
        inputs.image_risk = float(vision["image_risk"])
    result = combine(compute_signals(inputs, deps.models.cfg), deps.models.cfg)
    top = sorted(((k, b["contribution"]) for k, b in result.breakdown.items() if b["used"]), key=lambda kv: -kv[1])
    return {"risk": {
        "available": True,
        "score": result.score,
        "signals": result.signals,
        "breakdown": result.breakdown,
        "top_signals": [k for k, _ in top[:3]],
        "thresholds": deps.models.cfg.thresholds,
    }}
