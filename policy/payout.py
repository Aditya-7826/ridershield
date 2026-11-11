"""Deterministic payout policy. The LLM never decides amounts.

    payable_minutes = min(claimed delay, observed delay + tolerance)
    if payable_minutes < min_delay_minutes: payout = 0
    else payout = base + per_minute * (payable_minutes - min_delay_minutes), capped at max

The observed delay is an ETA-model estimate (test MAE about 5 minutes), so the
rider gets the benefit of that tolerance; inflated claims beyond it are not paid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Policy:
    id: int
    name: str
    base_payout_inr: float
    per_minute_inr: float
    max_payout_inr: float
    min_delay_minutes: int
    max_claims_per_week: int


@dataclass
class PayoutResult:
    amount_inr: float
    payable_minutes: float
    formula: str

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_payout(policy: Policy, claimed_minutes: float | None, observed_minutes: float,
                     tolerance_minutes: float = 5.0) -> PayoutResult:
    observed = max(float(observed_minutes), 0.0) + tolerance_minutes
    payable = observed if claimed_minutes is None else min(float(claimed_minutes), observed)
    payable = round(payable, 1)
    if payable < policy.min_delay_minutes:
        return PayoutResult(0.0, payable, f"{payable} min is below the {policy.min_delay_minutes} min minimum")
    extra = payable - policy.min_delay_minutes
    raw = policy.base_payout_inr + policy.per_minute_inr * extra
    amount = float(round(min(raw, policy.max_payout_inr)))
    formula = (f"Rs {policy.base_payout_inr:.0f} + Rs {policy.per_minute_inr:.0f} x {extra:.1f} min"
               f" = Rs {raw:.0f}" + (f", capped at Rs {policy.max_payout_inr:.0f}" if raw > policy.max_payout_inr else ""))
    return PayoutResult(amount, payable, formula)
