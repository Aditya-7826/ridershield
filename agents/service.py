"""Create new claims and attach photos (used by the CLI and the API)."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, text

from agents import config
from ml.photo_io import compute_phash, read_exif


def attach_photo(engine: Engine, claim_id: int, source: Path | str, upload_dir: Path | None = None) -> dict:
    """Copy the photo into the upload folder, fingerprint it and store its metadata.

    The perceptual hash and EXIF are computed once here, so every later check
    (reuse, time, location) reads them from claim_evidence.
    """
    upload_dir = Path(upload_dir or config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    dest = upload_dir / f"claim_{claim_id}{source.suffix.lower() or '.jpg'}"
    shutil.copyfile(source, dest)
    phash = compute_phash(dest)
    captured_at, gps = read_exif(dest)
    record = {"c": claim_id, "f": str(dest), "h": phash, "t": captured_at,
              "la": gps[0] if gps else None, "lo": gps[1] if gps else None}
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO claim_evidence (claim_id, kind, file_path, phash, captured_at, gps_lat, gps_lon)
            VALUES (:c, 'photo', :f, :h, :t, :la, :lo)
        """), record)
    return {"file_path": str(dest), "phash": phash, "captured_at": captured_at, "gps": gps}


def create_claim(engine: Engine, rider_id: int, claim_text: str, submitted_at: datetime,
                 delivery_id: int | None = None) -> int:
    """Insert a claim with placeholder fields; the intake agent fills them in.

    true_label stays NULL: only synthetic claims have ground truth.
    """
    with engine.begin() as conn:
        claim_id = conn.execute(text("""
            INSERT INTO claims (delivery_id, rider_id, submitted_at, text, reason_category,
                                claimed_delay_minutes, status, true_label)
            VALUES (:d, :r, :t, :x, 'other', 0, 'pending', NULL)
            RETURNING id
        """), {"d": delivery_id, "r": rider_id, "t": submitted_at, "x": claim_text}).scalar_one()
    return int(claim_id)
