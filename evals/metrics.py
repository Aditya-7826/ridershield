"""Evaluation metrics for claim decisions.

Each row has a ground-truth label (genuine_external / rider_caused /
fraudulent) and a decision (auto_approve / auto_reject / human_review).

Definitions
-----------
genuine P / R / F1     positive = auto_approve, truth = genuine_external.
                       A genuine claim sent to review counts as not
                       identified automatically (it still gets a human).
fraud recall           share of fraudulent claims NOT auto-approved
                       (auto-rejected or stopped for a human).
automation rate        (auto_approve + auto_reject) / all claims
manual reviews         claims needing a human; reduction is measured
                       against reviewing every claim and against the
                       rules-only baseline
auto-decision error    wrong automatic decisions / automatic decisions,
                       wrong = auto_approve of a non-genuine claim or
                       auto_reject of a genuine claim
"""
from __future__ import annotations

import pandas as pd

LABELS = ["genuine_external", "rider_caused", "fraudulent"]
DECISIONS = ["auto_approve", "human_review", "auto_reject"]


def compute_metrics(df: pd.DataFrame) -> dict:
    y = df["true_label"]
    d = df["decision"]
    genuine = y == "genuine_external"
    approve = d == "auto_approve"
    auto = d != "human_review"

    tp = int((approve & genuine).sum())
    fp = int((approve & ~genuine).sum())
    fn = int((~approve & genuine).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    fraud = y == "fraudulent"
    wrong_auto = (approve & ~genuine) | ((d == "auto_reject") & genuine)
    return {
        "n_claims": int(len(df)),
        "genuine_precision": round(precision, 3),
        "genuine_recall": round(recall, 3),
        "genuine_f1": round(f1, 3),
        "fraud_recall": round(float((~approve[fraud]).mean()), 3) if fraud.any() else None,
        "fraud_auto_rejected": round(float((d[fraud] == "auto_reject").mean()), 3) if fraud.any() else None,
        "automation_rate": round(float(auto.mean()), 3),
        "manual_reviews": int((~auto).sum()),
        "auto_decision_error_rate": round(float(wrong_auto[auto].mean()), 3) if auto.any() else 0.0,
        "wrong_auto_approvals": int((approve & ~genuine).sum()),
        "wrong_auto_rejections": int(((d == "auto_reject") & genuine).sum()),
    }


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = true label, columns = decision (counts)."""
    return (pd.crosstab(df["true_label"], df["decision"])
            .reindex(index=LABELS, columns=DECISIONS, fill_value=0))
