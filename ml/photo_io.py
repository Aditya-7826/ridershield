"""Read photo metadata: perceptual hash, EXIF timestamp and EXIF GPS.

phash (imagehash.phash) is a 64-bit fingerprint of what the picture looks
like. Resizing, re-compressing or small edits change only a few bits, so two
copies of the same photo stay within a small Hamming distance.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import imagehash
from PIL import ExifTags, Image

EXIF_IFD = ExifTags.IFD.Exif
GPS_IFD = ExifTags.IFD.GPSInfo
DATETIME_ORIGINAL = ExifTags.Base.DateTimeOriginal
EXIF_TIME_FORMAT = "%Y:%m:%d %H:%M:%S"


def compute_phash(path: Path | str) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(img.convert("RGB")))   # 16 hex chars


def _to_degrees(dms, ref: str) -> float:
    d, m, s = (float(x) for x in dms)
    value = d + m / 60 + s / 3600
    return -value if ref in ("S", "W") else value


def read_exif(path: Path | str) -> tuple[datetime | None, tuple[float, float] | None]:
    """Return (captured_at, (lat, lon)); either can be None when absent or unreadable."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            captured = None
            raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(ExifTags.Base.DateTime)
            if raw:
                captured = datetime.strptime(str(raw).strip("\x00"), EXIF_TIME_FORMAT)
            gps = None
            g = exif.get_ifd(GPS_IFD)
            if g and 2 in g and 4 in g:
                gps = (_to_degrees(g[2], g.get(1, "N")), _to_degrees(g[4], g.get(3, "E")))
            return captured, gps
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def _dms(value: float) -> tuple[float, float, float]:
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = round((value - d - m / 60) * 3600, 2)
    return (float(d), float(m), s)


def write_exif(img: Image.Image, path: Path | str, captured_at: datetime | None,
               gps: tuple[float, float] | None, quality: int = 85) -> None:
    """Save a JPEG with the given EXIF time/GPS (used to build test fixtures)."""
    exif = Image.Exif()
    if captured_at is not None:
        stamp = captured_at.strftime(EXIF_TIME_FORMAT)
        exif[ExifTags.Base.DateTime] = stamp
        exif.get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] = stamp
    if gps is not None:
        lat, lon = gps
        exif.get_ifd(GPS_IFD).update({1: "N" if lat >= 0 else "S", 2: _dms(lat), 3: "E" if lon >= 0 else "W", 4: _dms(lon)})
    img.convert("RGB").save(path, "JPEG", quality=quality, exif=exif)
