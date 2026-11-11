"""Turn raw tables into the per-delivery frame that every model uses.

`build_delivery_frame` is the single place where deliveries are joined to
rider and hourly zone conditions. It works on plain DataFrames, so the same
code serves the database (`load_delivery_frame`) and the tests (generator
output, no database needed).
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from ml.config import VEHICLE_TYPES, ZONE_IDS

ETA_FEATURES = [
    "distance_km", "zone_id", "hour", "weekday", "vehicle_type",
    "rain_mm", "traffic_level", "is_flood", "is_road_closure",
]


def build_delivery_frame(deliveries: pd.DataFrame, riders: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery with the conditions of its zone in that hour."""
    df = deliveries.merge(riders[["id", "vehicle_type"]].rename(columns={"id": "rider_id"}), on="rider_id", how="left")
    df["hour_start"] = df["picked_up_at"].dt.floor("h")
    cond = conditions[["zone_id", "hour_start", "rain_mm", "traffic_level", "event_type", "severity"]]
    df = df.merge(cond, on=["zone_id", "hour_start"], how="left")
    df[["rain_mm", "traffic_level", "severity"]] = df[["rain_mm", "traffic_level", "severity"]].fillna(0.0)
    df["event_type"] = df["event_type"].fillna("none")
    df["hour"] = df["picked_up_at"].dt.hour
    df["weekday"] = df["picked_up_at"].dt.day_of_week
    df["is_flood"] = (df["event_type"] == "flood").astype(int)
    df["is_road_closure"] = (df["event_type"] == "road_closure").astype(int)
    return df.sort_values("picked_up_at", kind="stable").reset_index(drop=True)


def load_delivery_frame(engine: Engine) -> pd.DataFrame:
    """Read the raw tables from PostgreSQL and build the delivery frame."""
    with engine.connect() as conn:
        deliveries = pd.read_sql("SELECT * FROM deliveries", conn, parse_dates=["picked_up_at"])
        riders = pd.read_sql("SELECT id, vehicle_type FROM riders", conn)
        conditions = pd.read_sql(
            "SELECT zone_id, hour_start, rain_mm, traffic_level, event_type, severity FROM conditions",
            conn, parse_dates=["hour_start"],
        )
    return build_delivery_frame(deliveries, riders, conditions)


def eta_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the ETA model, with fixed categories so a single row predicts fine."""
    X = df[ETA_FEATURES].copy()
    X["zone_id"] = pd.Categorical(X["zone_id"], categories=ZONE_IDS)
    X["vehicle_type"] = pd.Categorical(X["vehicle_type"], categories=VEHICLE_TYPES)
    return X
