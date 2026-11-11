"""Generate the small test photos in tests/fixtures/ (drawn with Pillow, no copyright).

    python scripts/make_fixtures.py

Creates:
  flooded_road.jpg        road under water, EXIF time/GPS set by the test
  flooded_road_copy.jpg   the same photo resized + re-compressed (reuse case)
  traffic_jam.jpg         rows of vehicles
  clear_road.jpg          an empty dry road
The EXIF here is a fixed placeholder; tests rewrite it to match a delivery.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.photo_io import write_exif  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SIZE = (480, 320)
WHEN = datetime(2026, 8, 20, 19, 30)
WHERE = (12.9166, 77.6101)


def _street(draw: ImageDraw.ImageDraw, sky="#9aa5b1"):
    w, h = SIZE
    draw.rectangle([0, 0, w, h // 2], fill=sky)                      # sky
    for x in range(0, w, 70):                                         # buildings
        draw.rectangle([x + 5, 40 + (x * 7) % 50, x + 60, h // 2], fill="#6b6f76")
    draw.polygon([(0, h), (w, h), (w * 0.62, h // 2), (w * 0.38, h // 2)], fill="#55585c")  # road


def flooded_road() -> Image.Image:
    img = Image.new("RGB", SIZE)
    d = ImageDraw.Draw(img)
    _street(d, sky="#7d8794")
    w, h = SIZE
    d.rectangle([0, int(h * 0.62), w, h], fill="#5b6e6a")                 # muddy water
    for y in range(int(h * 0.64), h, 9):                                  # ripples
        d.line([(0, y), (w, y + 3)], fill="#7f9690", width=2)
    d.rectangle([200, 175, 280, 215], fill="#b23a3a")                     # half-submerged car
    for x in range(0, w, 14):                                             # rain streaks
        d.line([(x, 0), (x - 8, h // 2)], fill="#b8c2cc", width=1)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def traffic_jam() -> Image.Image:
    img = Image.new("RGB", SIZE)
    d = ImageDraw.Draw(img)
    _street(d, sky="#c9d6e3")
    colors = ["#c0392b", "#f1c40f", "#2c3e50", "#ecf0f1", "#27ae60", "#8e44ad"]
    for row, y in enumerate(range(170, 320, 38)):
        for i, x in enumerate(range(-20, 480, 62)):
            d.rectangle([x + row * 9, y, x + row * 9 + 52, y + 26], fill=colors[(i + row) % len(colors)])
            d.ellipse([x + row * 9 + 6, y + 20, x + row * 9 + 16, y + 30], fill="black")
            d.ellipse([x + row * 9 + 36, y + 20, x + row * 9 + 46, y + 30], fill="black")
    return img


def clear_road() -> Image.Image:
    img = Image.new("RGB", SIZE)
    d = ImageDraw.Draw(img)
    _street(d, sky="#8fc3ea")
    w, h = SIZE
    for y in range(h // 2 + 10, h, 30):                                    # lane markings
        d.rectangle([w // 2 - 3, y, w // 2 + 3, y + 14], fill="white")
    d.ellipse([380, 20, 430, 70], fill="#f9d71c")                          # sun
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    flood = flooded_road()
    write_exif(flood, OUT / "flooded_road.jpg", WHEN, WHERE)
    write_exif(flood.resize((360, 240)), OUT / "flooded_road_copy.jpg", None, None, quality=70)
    write_exif(traffic_jam(), OUT / "traffic_jam.jpg", WHEN, WHERE)
    write_exif(clear_road(), OUT / "clear_road.jpg", WHEN, WHERE)
    for p in sorted(OUT.glob("*.jpg")):
        print(f"{p.name:<24} {p.stat().st_size / 1024:5.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
