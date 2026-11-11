"""Score held-out claims and show how well the risk score separates the labels.

Usage:
    python scripts/risk_report.py

Held-out = claims on deliveries after ml.config.TRAIN_CUTOFF. Rider behaviour
features are a snapshot at the cutoff, so nothing from the test period leaks
into them. Writes ml/model_cards/risk_score.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.session import get_engine  # noqa: E402
from ml.claim_scoring import Models, decide, load_tables, score_claims  # noqa: E402
from ml.config import TRAIN_CUTOFF  # noqa: E402
from ml.model_card import md_table, write_model_card  # noqa: E402
from ml.risk import SIGNALS  # noqa: E402

LABEL_ORDER = ["genuine_external", "rider_caused", "fraudulent"]


# Wrongly rejecting a genuine rider costs them income, so rejection gets the
# stricter limit; wrongly approving costs the insurer a small payout.
MAX_APPROVE_ERROR = 0.02  # <= 2% of auto-approved claims may be non-genuine
MAX_REJECT_ERROR = 0.01   # <= 1% of auto-rejected claims may be genuine


def calibrate(models: Models, tables: dict) -> int:
    """Suggest thresholds from TRAINING-period claims only (never the test set).

    auto_approve_below = highest cut where <= 2% of auto-approved claims are not genuine
    auto_reject_at     = lowest cut where <= 1% of auto-rejected claims are genuine
    """
    frame = tables["frame"]
    train_deliveries = set(frame.loc[frame["picked_up_at"] < TRAIN_CUTOFF, "id"])
    train_claims = tables["claims"][tables["claims"]["delivery_id"].isin(train_deliveries)]
    print(f"Calibrating on {len(train_claims)} training-period claims...")
    s = score_claims(train_claims, tables, models, rider_features_as_of=TRAIN_CUTOFF)
    genuine = s["true_label"] == "genuine_external"
    rows = []
    for t in [round(x * 0.01, 2) for x in range(10, 91)]:
        appr, rej = s["risk_score"] < t, s["risk_score"] >= t
        rows.append({
            "t": t,
            "approve_share": appr.mean(), "approve_error": (~genuine[appr]).mean() if appr.any() else 0.0,
            "reject_share": rej.mean(), "reject_error": genuine[rej].mean() if rej.any() else 0.0,
        })
    grid = pd.DataFrame(rows)
    approve_t = grid.loc[grid["approve_error"] <= MAX_APPROVE_ERROR, "t"].max()
    reject_t = grid.loc[(grid["reject_error"] <= MAX_REJECT_ERROR) & (grid["t"] >= approve_t), "t"].min()
    print(grid[grid["t"].isin([approve_t, reject_t])].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSuggested thresholds (training claims; approve error <= {MAX_APPROVE_ERROR:.0%}, "
          f"reject error <= {MAX_REJECT_ERROR:.0%}):")
    print(f"  auto_approve_below = {approve_t}\n  auto_reject_at_or_above = {reject_t}")
    print("Copy them into ml/risk_config.toml if you want to use them.")
    return 0


def main() -> int:
    models = Models.load()
    tables = load_tables(get_engine())
    if "--calibrate" in sys.argv:
        return calibrate(models, tables)
    frame = tables["frame"]
    test_deliveries = set(frame.loc[frame["picked_up_at"] >= TRAIN_CUTOFF, "id"])
    test_claims = tables["claims"][tables["claims"]["delivery_id"].isin(test_deliveries)]
    print(f"Scoring {len(test_claims)} held-out claims (deliveries after {TRAIN_CUTOFF:%Y-%m-%d})...")

    scored = score_claims(test_claims, tables, models, rider_features_as_of=TRAIN_CUTOFF)
    scored["decision"] = scored["risk_score"].apply(lambda s: decide(s, models.cfg))

    g = scored.groupby("true_label")["risk_score"]
    dist = pd.DataFrame({
        "claims": g.size(), "mean": g.mean(), "p10": g.quantile(0.1),
        "median": g.median(), "p90": g.quantile(0.9),
    }).reindex(LABEL_ORDER).reset_index()

    sig_cols = [f"sig_{s}" for s in SIGNALS]
    signals = scored.groupby("true_label")[sig_cols].mean().reindex(LABEL_ORDER)
    signals.columns = SIGNALS
    signals = signals.reset_index()

    decisions = pd.crosstab(scored["true_label"], scored["decision"], normalize="index").reindex(LABEL_ORDER)
    decisions = decisions.reindex(columns=["auto_approve", "human_review", "auto_reject"], fill_value=0).reset_index()

    is_bad = scored["true_label"] != "genuine_external"
    auc_bad = roc_auc_score(is_bad, scored["risk_score"])
    nonfraud = scored["true_label"] != "fraudulent"
    auc_fraud_vs_genuine = roc_auc_score(
        scored.loc[scored["true_label"] != "rider_caused", "true_label"] == "fraudulent",
        scored.loc[scored["true_label"] != "rider_caused", "risk_score"],
    )
    auc_rider_vs_genuine = roc_auc_score(
        scored.loc[nonfraud, "true_label"] == "rider_caused", scored.loc[nonfraud, "risk_score"],
    )
    aucs = pd.DataFrame([
        {"comparison": "genuine vs (rider_caused + fraudulent)", "ROC AUC": auc_bad},
        {"comparison": "genuine vs rider_caused", "ROC AUC": auc_rider_vs_genuine},
        {"comparison": "genuine vs fraudulent", "ROC AUC": auc_fraud_vs_genuine},
    ])

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print("\n== Risk score by true label (held-out) ==")
    print(dist.to_string(index=False, float_format=fmt))
    print("\n== Mean signal value by label (0 = looks genuine, 1 = looks risky) ==")
    print(signals.to_string(index=False, float_format=fmt))
    print(f"\n== Decision share by label (approve < {models.cfg.thresholds['auto_approve_below']}, "
          f"reject >= {models.cfg.thresholds['auto_reject_at_or_above']}) ==")
    print(decisions.to_string(index=False, float_format=fmt))
    print("\n== Separation ==")
    print(aucs.to_string(index=False, float_format=fmt))

    w = models.cfg.weights
    card = write_model_card("risk_score", "Weighted claim risk score", {
        "Purpose": "Combine the evidence about a claim into one 0-1 score: 0 = consistent with a genuine "
                   "external delay, 1 = consistent with a rider-caused or fraudulent claim.",
        "Signals and weights": "Defined in `ml/risk_config.toml`; unavailable signals are skipped and the "
                               "remaining weights rescaled.\n\n"
                               + "\n".join(f"- `{s}`: {w[s]}" for s in SIGNALS),
        "Evaluation data": f"{len(scored)} synthetic claims on deliveries after {TRAIN_CUTOFF:%Y-%m-%d} "
                           "(none of the models saw them).",
        "Risk score by label": md_table(dist),
        "Mean signal by label": md_table(signals),
        "Decision share by label": md_table(decisions),
        "Separation (ROC AUC)": md_table(aucs),
        "Thresholds": "Weights were set by hand. Thresholds were calibrated on training-period claims only "
                      "(`--calibrate`): at most 2% of auto-approvals non-genuine and at most 1% of auto-rejections "
                      "genuine, because wrongly rejecting a rider is the more harmful mistake.",
        "Limitations": "The synthetic generator makes the signals unrealistically clean, so the near-perfect "
                       "separation here shows the pipeline works end to end, not real-world accuracy. Real claims "
                       "would overlap far more and need a wider human-review band. Phase 7 (`evals/`) compares this "
                       "against a rules-only baseline and the full agent pipeline.",
    })
    print(f"\nWrote {Path(card).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
