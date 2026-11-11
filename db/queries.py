"""Targeted SQL queries used by the agents (one claim at a time).

The batch tools load whole tables; the agents only fetch what a single claim
needs so a claim is processed in well under a second of database time.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import Engine, text

DELIVERY_COLS = "d.id, d.rider_id, d.zone_id, d.picked_up_at, d.distance_km, d.restaurant_wait_minutes, " \
                "d.idle_minutes, d.avg_speed_kmh, d.actual_minutes"


def _df(engine: Engine, sql: str, params: dict, dates: list[str] | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params, parse_dates=dates or [])


def get_claim(engine: Engine, claim_id: int) -> dict | None:
    df = _df(engine, "SELECT * FROM claims WHERE id = :id", {"id": claim_id}, ["submitted_at"])
    return None if df.empty else df.iloc[0].to_dict()


def get_delivery(engine: Engine, delivery_id: int) -> pd.DataFrame:
    return _df(engine, f"SELECT {DELIVERY_COLS} FROM deliveries d WHERE d.id = :id", {"id": delivery_id}, ["picked_up_at"])


def latest_delivery_before(engine: Engine, rider_id: int, before: datetime, within_hours: int = 12) -> int | None:
    df = _df(engine, """
        SELECT id FROM deliveries
        WHERE rider_id = :r AND picked_up_at < :t AND picked_up_at >= :t0
        ORDER BY picked_up_at DESC LIMIT 1
    """, {"r": rider_id, "t": before, "t0": before - timedelta(hours=within_hours)})
    return None if df.empty else int(df.iloc[0]["id"])


def zone_window(engine: Engine, zone_id: int, hour_start: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deliveries and hourly conditions for a zone, one hour either side."""
    start, end = hour_start - timedelta(hours=1), hour_start + timedelta(hours=2)
    p = {"z": zone_id, "s": start, "e": end}
    deliveries = _df(engine, f"SELECT {DELIVERY_COLS} FROM deliveries d WHERE d.zone_id = :z "
                             "AND d.picked_up_at >= :s AND d.picked_up_at < :e", p, ["picked_up_at"])
    conditions = _df(engine, "SELECT zone_id, hour_start, rain_mm, traffic_level, event_type, severity FROM conditions "
                             "WHERE zone_id = :z AND hour_start >= :s AND hour_start < :e", p, ["hour_start"])
    return deliveries, conditions


def riders_vehicle(engine: Engine, rider_ids: list[int]) -> pd.DataFrame:
    return _df(engine, "SELECT id, vehicle_type FROM riders WHERE id = ANY(:ids)", {"ids": list(map(int, rider_ids))})


def rider_history(engine: Engine, rider_id: int, before: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rider's deliveries and claims strictly before `before`."""
    p = {"r": rider_id, "t": before}
    deliveries = _df(engine, f"SELECT {DELIVERY_COLS} FROM deliveries d WHERE d.rider_id = :r AND d.picked_up_at < :t",
                     p, ["picked_up_at"])
    claims = _df(engine, "SELECT id, rider_id, delivery_id, submitted_at FROM claims WHERE rider_id = :r AND submitted_at < :t",
                 p, ["submitted_at"])
    return deliveries, claims


def other_claims_for_delivery(engine: Engine, delivery_id: int, claim_id: int) -> int:
    df = _df(engine, "SELECT COUNT(*) AS n FROM claims WHERE delivery_id = :d AND id <> :c",
             {"d": delivery_id, "c": claim_id})
    return int(df.iloc[0]["n"])


def claim_photo(engine: Engine, claim_id: int) -> dict | None:
    df = _df(engine, "SELECT * FROM claim_evidence WHERE claim_id = :c AND kind = 'photo' ORDER BY id LIMIT 1",
             {"c": claim_id}, ["captured_at"])
    return None if df.empty else df.iloc[0].to_dict()


def previous_photo_hashes(engine: Engine, before: datetime, claim_id: int) -> list[str]:
    df = _df(engine, """
        SELECT e.phash FROM claim_evidence e JOIN claims c ON c.id = e.claim_id
        WHERE e.phash IS NOT NULL AND c.submitted_at < :t AND c.id <> :c
    """, {"t": before, "c": claim_id})
    return df["phash"].tolist()


def zone_center(engine: Engine, zone_id: int) -> tuple[str, float, float]:
    df = _df(engine, "SELECT name, lat, lon FROM zones WHERE id = :z", {"z": zone_id})
    r = df.iloc[0]
    return str(r["name"]), float(r["lat"]), float(r["lon"])


def active_policy(engine: Engine) -> dict:
    df = _df(engine, "SELECT * FROM policies WHERE active ORDER BY id LIMIT 1", {})
    return df.iloc[0].to_dict()
