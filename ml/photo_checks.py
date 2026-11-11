"""Deterministic photo checks: reuse, EXIF time and EXIF location.

These work on photo *metadata* (perceptual hash, EXIF timestamp, EXIF GPS),
so they are cheap and need no model. The vision agent adds a content check
("does the photo actually show a flooded road?") on top.

Perceptual hashes of the same picture stay within a few bits of each other
even after resizing or re-compression, so we compare by Hamming distance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

REUSE_MAX_HAMMING = 6            # bits out of 64
MAX_TIME_GAP = timedelta(hours=3)
MAX_DISTANCE_KM = 5.0

RISK_REUSED = 1.0
RISK_MISMATCH = 0.7
RISK_NO_EXIF = 0.4
RISK_OK = 0.0


def hamming(a_hex: str, b_hex: str) -> int:
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


@dataclass
class PhotoCheck:
    risk: float
    reused: bool = False
    closest_hamming: int | None = None
    exif_status: str = "ok"               # ok | missing | time_mismatch | location_mismatch
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_photo(
    phash: str | None,
    captured_at: datetime | None,
    gps: tuple[float, float] | None,
    delivery_start: datetime,
    delivery_end: datetime,
    zone_center: tuple[float, float],
    previous_hashes: list[str],
) -> PhotoCheck:
    """Check one photo against the delivery and all previously submitted photos."""
    reasons: list[str] = []
    closest = min((hamming(phash, h) for h in previous_hashes), default=None) if phash else None
    reused = closest is not None and closest <= REUSE_MAX_HAMMING
    if reused:
        reasons.append(f"photo matches a previously submitted photo ({closest} of 64 bits differ)")

    if captured_at is None:
        exif = "missing"
        reasons.append("photo has no EXIF timestamp")
    elif not (delivery_start - MAX_TIME_GAP <= captured_at <= delivery_end + MAX_TIME_GAP):
        exif = "time_mismatch"
        reasons.append(f"photo taken at {captured_at:%Y-%m-%d %H:%M}, not during the delivery")
    elif gps is not None and haversine_km(*gps, *zone_center) > MAX_DISTANCE_KM:
        exif = "location_mismatch"
        reasons.append(f"photo GPS is {haversine_km(*gps, *zone_center):.1f} km from the delivery zone")
    else:
        exif = "ok"

    if reused:
        risk = RISK_REUSED
    elif exif in ("time_mismatch", "location_mismatch"):
        risk = RISK_MISMATCH
    elif exif == "missing":
        risk = RISK_NO_EXIF
    else:
        risk = RISK_OK
    return PhotoCheck(risk=risk, reused=reused, closest_hamming=closest, exif_status=exif, reasons=reasons)
