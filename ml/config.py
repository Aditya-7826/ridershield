"""Shared ML settings: paths and the time-based train/test split.

We split by TIME, not randomly: models are trained on the first 70 days and
evaluated on the last 20. That mimics real use (train on the past, score the
future) and keeps the Phase 7 held-out claims unseen by every model.
"""
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"          # trained models (gitignored)
MODEL_CARDS_DIR = ROOT / "ml" / "model_cards"      # human-readable cards (committed)

# Synthetic data covers 2026-06-01 .. 2026-08-29. Everything picked up before
# this moment is training data; everything after is held-out test data.
TRAIN_CUTOFF = datetime(2026, 8, 10)

ZONE_IDS = list(range(1, 13))
VEHICLE_TYPES = ["motorbike", "scooter", "e-bike"]
