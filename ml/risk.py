"""Weighted risk score for a claim.

    risk = sum(weight_i * signal_i) / sum(weight_i over available signals)

Every signal is in [0, 1]; 0 = consistent with a genuine external delay,
1 = consistent with a rider-caused or fraudulent claim. Weights and
parameters live in ml/risk_config.toml. The breakdown records each signal's
value, its (rescaled) weight and its contribution, so the UI and the
decision agent can show exactly why a claim scored what it did.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SIGNALS = ["condition_support", "claim_inflation", "zone_disruption", "rider_anomaly", "image_check", "claim_history"]
CONFIG_PATH = Path(__file__).with_name("risk_config.toml")


@dataclass
class RiskConfig:
    weights: dict[str, float]
    params: dict[str, float]
    thresholds: dict[str, float]

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> RiskConfig:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        cfg = cls(weights=raw["weights"], params=raw["params"], thresholds=raw["thresholds"])
        cfg.validate()
        return cfg

    def validate(self) -> None:
        missing = set(SIGNALS) - set(self.weights)
        if missing:
            raise ValueError(f"risk config is missing weights for: {sorted(missing)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("risk weights must be non-negative")
        if not 0 <= self.thresholds["auto_approve_below"] <= self.thresholds["auto_reject_at_or_above"] <= 1:
            raise ValueError("thresholds must satisfy 0 <= auto_approve_below <= auto_reject_at_or_above <= 1")


@dataclass
class ClaimInputs:
    """Everything the score needs about one claim (all minutes)."""
    claimed_delay: float          # what the rider says
    observed_delay: float         # actual - ETA on a normal day
    condition_delay: float        # ETA(actual conditions) - ETA(normal day)
    restaurant_wait: float        # recorded wait at the restaurant
    zone_disruption_score: float  # 0-1 from the zone detector
    zone_judgeable: bool          # enough deliveries in the window to judge?
    rider_anomaly: float          # 0-1 percentile
    claims_last_30d: int
    image_risk: float | None = None   # None when no photo was attached


@dataclass
class RiskResult:
    score: float
    signals: dict[str, float | None]
    breakdown: dict[str, dict] = field(default_factory=dict)


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def compute_signals(x: ClaimInputs, cfg: RiskConfig) -> dict[str, float | None]:
    tol = cfg.params["tolerance_minutes"]
    restaurant_excess = max(x.restaurant_wait - cfg.params["normal_restaurant_wait_minutes"], 0.0)
    explained = x.condition_delay + restaurant_excess
    # Fraction of the real delay that outside causes do not explain. A tiny
    # real delay counts as tolerance-sized so the ratio stays meaningful.
    observed = max(x.observed_delay, tol)
    condition_support = _clip01((observed - explained - tol) / observed)
    # Claimed minutes beyond what really happened, relative to the claim.
    claimed = max(x.claimed_delay, 1.0)
    claim_inflation = _clip01((x.claimed_delay - max(x.observed_delay, 0.0) - tol) / claimed)
    zone = 1.0 - x.zone_disruption_score if x.zone_judgeable else cfg.params["zone_neutral_value"]
    history = _clip01(x.claims_last_30d / cfg.params["claim_history_saturation"])
    return {
        "condition_support": round(condition_support, 4),
        "claim_inflation": round(claim_inflation, 4),
        "zone_disruption": round(_clip01(zone), 4),
        "rider_anomaly": round(_clip01(x.rider_anomaly), 4),
        "image_check": None if x.image_risk is None else round(_clip01(x.image_risk), 4),
        "claim_history": round(history, 4),
    }


def combine(signals: dict[str, float | None], cfg: RiskConfig) -> RiskResult:
    """Weighted average over the signals that are available."""
    available = {k: v for k, v in signals.items() if v is not None and cfg.weights.get(k, 0) > 0}
    total_w = sum(cfg.weights[k] for k in available)
    breakdown: dict[str, dict] = {}
    score = 0.0
    for name in SIGNALS:
        value = signals.get(name)
        if name not in available:
            breakdown[name] = {"value": value, "weight": 0.0, "contribution": 0.0, "used": False}
            continue
        w = cfg.weights[name] / total_w
        score += w * value
        breakdown[name] = {"value": value, "weight": round(w, 4), "contribution": round(w * value, 4), "used": True}
    return RiskResult(score=round(score, 4) if total_w else 0.5, signals=signals, breakdown=breakdown)


def score_claim(x: ClaimInputs, cfg: RiskConfig | None = None) -> RiskResult:
    cfg = cfg or RiskConfig.load()
    return combine(compute_signals(x, cfg), cfg)
