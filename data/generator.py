"""Reproducible synthetic data for RiderShield.

The generator simulates a Bengaluru-style city over a monsoon season:

1. Conditions: for every zone and hour, rainfall, traffic level and any
   disruption event (rain, heavy rain, flood, traffic jam, road closure).
   Events are shared by the whole zone, so a genuine disruption delays MANY
   riders in the same place at the same time.
2. Deliveries: each delivery's time is built from travel time under the real
   conditions, restaurant wait, hand-over time and the rider's own idle stops.
   We keep the parts separate, so we know *why* a delivery was late.
3. Claims: riders file delay claims. Each claim gets a ground-truth label:
     genuine_external  the delay mostly came from conditions / the restaurant
     rider_caused      the delay mostly came from the rider's own idle stops
     fraudulent        a fraudster claims a delay on an undisrupted delivery,
                       inflates it, and often reuses an old photo
   Labels are used only for evaluation, never as model features.

Rider behaviour profiles (honest / careless / fraudster) are hidden: they shape
the data but are not stored in the database.

All timestamps are naive local time (IST). The same seed always gives the
same data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# name, lat, lon, flood_prone, base_traffic
ZONES = [
    ("Koramangala", 12.9352, 77.6245, True, 0.15),
    ("Indiranagar", 12.9784, 77.6408, False, 0.15),
    ("HSR Layout", 12.9116, 77.6474, True, 0.10),
    ("Whitefield", 12.9698, 77.7500, False, 0.20),
    ("Jayanagar", 12.9250, 77.5938, False, 0.05),
    ("Marathahalli", 12.9591, 77.6974, True, 0.20),
    ("BTM Layout", 12.9166, 77.6101, True, 0.10),
    ("Electronic City", 12.8399, 77.6770, False, 0.10),
    ("Hebbal", 13.0358, 77.5970, False, 0.15),
    ("Malleshwaram", 13.0031, 77.5643, False, 0.05),
    ("Yelahanka", 13.1007, 77.5963, False, 0.00),
    ("Bellandur", 12.9258, 77.6762, True, 0.20),
]

FIRST_NAMES = [
    "Aarav", "Vihaan", "Arjun", "Rahul", "Suresh", "Manoj", "Kiran", "Ravi", "Imran", "Salman",
    "Ganesh", "Prakash", "Naveen", "Abdul", "Mohan", "Vijay", "Ajay", "Deepak", "Santosh", "Ramesh",
    "Priya", "Lakshmi", "Divya", "Ayesha", "Kavya", "Sneha", "Pooja", "Anjali", "Meena", "Fatima",
]
LAST_NAMES = [
    "Kumar", "Reddy", "Gowda", "Shetty", "Naik", "Khan", "Sharma", "Rao", "Patil", "Singh",
    "Nair", "Iyer", "Hegde", "Pasha", "Yadav", "Das", "Pillai", "Murthy", "Ali", "Joshi",
]

# Typical congestion by hour of day (0-1), before zone offsets.
TRAFFIC_BY_HOUR = np.array([
    0.10, 0.08, 0.05, 0.05, 0.05, 0.10, 0.20, 0.35, 0.55, 0.65, 0.55, 0.45,
    0.45, 0.45, 0.40, 0.40, 0.45, 0.55, 0.65, 0.65, 0.55, 0.40, 0.25, 0.15,
])
# Share of orders by hour of day: lunch and dinner peaks.
ORDERS_BY_HOUR = np.array([
    1, 0.5, 0.3, 0.2, 0.2, 0.3, 1, 2, 3, 3, 3, 5,
    8, 8, 6, 4, 3, 4, 6, 8, 9, 8, 5, 2,
], dtype=float)
ORDERS_BY_HOUR /= ORDERS_BY_HOUR.sum()

VEHICLE_SPEED_KMH = {"motorbike": 26.0, "scooter": 23.0, "e-bike": 18.0}

HEAVY_RAIN_MM = 15.0      # mm in one hour
LIGHT_RAIN_MM = 2.5
JAM_TRAFFIC = 0.85
CLAIM_MIN_DELAY = 12.0    # riders rarely claim for less than this many extra minutes

DEFAULT_POLICY = {
    "id": 1,
    "name": "Monsoon Delay Cover",
    "base_payout_inr": 30.0,
    "per_minute_inr": 2.0,
    "max_payout_inr": 150.0,
    "min_delay_minutes": 10,
    "max_claims_per_week": 5,
    "active": True,
}

CLAIM_TEMPLATES = {
    "heavy_rain": [
        "Heavy rain in {zone}, could barely see the road. Order #{id} was about {m} min late.",
        "It was pouring near {zone} so I had to ride slowly. Delivery {id} delayed by {m} minutes.",
        "Rain was very heavy, had to stop under a flyover for some time. Order {id}, {m} mins late.",
        "Too much rain today in {zone}, roads were slippery. {m} min delay on my delivery.",
    ],
    "flooding": [
        "Road near {zone} was flooded, water up to the knees. Order #{id} delayed {m} min.",
        "Waterlogging in {zone}, had to take a long route around. Delivery {id} late by {m} minutes.",
        "Underpass flooded so I could not pass. {m} minutes delay for order {id}.",
    ],
    "traffic_jam": [
        "Massive traffic jam in {zone}, nothing was moving. Order #{id} got {m} min late.",
        "Stuck in traffic for a long time near {zone}. Delivery {id} delayed by {m} minutes.",
        "Signal not working and full jam. Order {id} was {m} min late.",
        "Heavy traffic on the main road, {m} min delay, order #{id}.",
    ],
    "road_closure": [
        "Road was closed near {zone} for some work, had to go around. Order #{id}, {m} min late.",
        "Police had blocked the road in {zone}. Delivery {id} delayed {m} minutes.",
        "Barricades on the usual route, long detour. {m} minute delay on order {id}.",
    ],
    "restaurant_delay": [
        "Restaurant took very long to give the order. Waited a lot, order #{id} {m} min late.",
        "Food was not ready at pickup in {zone}, I waited. Delivery {id} delayed by {m} minutes.",
        "Restaurant was overloaded, order {id} came out late. {m} min delay.",
    ],
}
# Some riders do not mention the order number; intake has to find it.
CLAIM_TEMPLATES_NO_ID = {
    k: [t.replace("Order #{id}", "My order").replace("order #{id}", "my order")
         .replace("Delivery {id}", "Delivery").replace("delivery {id}", "delivery")
         .replace("order {id}", "my order").replace("Order {id}", "My order")
         .replace("for order {id}", "") for t in v]
    for k, v in CLAIM_TEMPLATES.items()
}


@dataclass
class GeneratorConfig:
    seed: int = 42
    n_riders: int = 500
    n_deliveries: int = 50_000
    n_days: int = 90
    start: datetime = datetime(2026, 6, 1)
    careless_share: float = 0.10
    fraud_share: float = 0.05


@dataclass
class SyntheticData:
    zones: pd.DataFrame
    riders: pd.DataFrame
    policies: pd.DataFrame
    conditions: pd.DataFrame
    deliveries: pd.DataFrame
    claims: pd.DataFrame
    claim_evidence: pd.DataFrame
    # Hidden ground truth, for tests only (not written to the database).
    rider_profiles: pd.Series = field(repr=False, default=None)


# --------------------------------------------------------------------------- conditions


def _make_conditions(rng: np.random.Generator, cfg: GeneratorConfig) -> dict[str, np.ndarray]:
    """Return arrays of shape (zones, hours) describing every zone-hour."""
    n_zones, n_hours = len(ZONES), cfg.n_days * 24
    flood_prone = np.array([z[3] for z in ZONES])
    base_traffic = np.array([z[4] for z in ZONES])
    hour_of_day = np.arange(n_hours) % 24

    # Rain: on about half the days, 1-2 storms move across the city.
    rain = np.zeros((n_zones, n_hours))
    for day in range(cfg.n_days):
        if rng.random() < 0.55:
            for _ in range(rng.integers(1, 3)):
                start = day * 24 + int(rng.integers(0, 24))
                end = min(start + int(rng.integers(1, 6)), n_hours)
                intensity = rng.gamma(2.0, 5.0)                  # city-wide mm/h
                zone_factor = rng.uniform(0.4, 1.6, n_zones)     # storms are uneven
                rain[:, start:end] += intensity * zone_factor[:, None] * rng.uniform(0.6, 1.4, (n_zones, end - start))

    # Flood: lots of rain in the last 3 hours; low-lying zones flood sooner.
    # Water takes ~2 hours to drain after the trigger.
    padded = np.concatenate([np.zeros((n_zones, 2)), rain], axis=1)
    rain_3h = padded[:, 2:] + padded[:, 1:-1] + padded[:, :-2]
    trigger = rain_3h >= np.where(flood_prone, 45.0, 75.0)[:, None]
    flood = trigger.copy()
    for lag in (1, 2):
        flood[:, lag:] |= trigger[:, :-lag]

    # Normal traffic = hour-of-day profile + zone offset (what riders expect).
    normal_traffic = np.clip(TRAFFIC_BY_HOUR[hour_of_day][None, :] + base_traffic[:, None], 0, 1)
    traffic = normal_traffic + rng.normal(0, 0.05, (n_zones, n_hours)) + 0.15 * np.minimum(rain / 20, 1)

    # Random jams and closures (accidents, protests, road works).
    closure = np.zeros((n_zones, n_hours), dtype=bool)
    for z in range(n_zones):
        for day in range(cfg.n_days):
            if rng.random() < 0.08:
                start = day * 24 + int(rng.integers(7, 22))
                end = min(start + int(rng.integers(1, 4)), n_hours)
                traffic[z, start:end] = np.maximum(traffic[z, start:end], rng.uniform(0.85, 1.0))
            if rng.random() < 0.02:
                start = day * 24 + int(rng.integers(6, 22))
                closure[z, start:min(start + int(rng.integers(2, 7)), n_hours)] = True
    traffic = np.clip(traffic, 0, 1)

    return {
        "rain": rain, "traffic": traffic, "normal_traffic": normal_traffic,
        "flood": flood, "closure": closure,
    }


def _label_conditions(c: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pick one event type per zone-hour (most serious wins) and a 0-1 severity."""
    rain, traffic, flood, closure = c["rain"], c["traffic"], c["flood"], c["closure"]
    event = np.full(rain.shape, "none", dtype=object)
    event[rain >= LIGHT_RAIN_MM] = "rain"
    event[traffic >= JAM_TRAFFIC] = "traffic_jam"
    event[rain >= HEAVY_RAIN_MM] = "heavy_rain"
    event[closure] = "road_closure"
    event[flood] = "flood"

    severity = np.maximum(np.minimum(rain / 30, 1), np.clip((traffic - 0.6) / 0.4, 0, 1))
    severity = np.where(closure, np.maximum(severity, 0.7), severity)
    severity = np.where(flood, np.maximum(severity, 0.9), severity)
    return event, np.round(severity, 3)


# --------------------------------------------------------------------------- generate


def generate(cfg: GeneratorConfig | None = None) -> SyntheticData:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n_zones, n_hours = len(ZONES), cfg.n_days * 24
    start = pd.Timestamp(cfg.start)

    # ---- zones
    zones = pd.DataFrame(ZONES, columns=["name", "lat", "lon", "flood_prone", "base_traffic"])
    zones.insert(0, "id", np.arange(1, n_zones + 1))

    # ---- riders (profile is hidden ground truth)
    n_r = cfg.n_riders
    profile = rng.choice(
        ["honest", "careless", "fraudster"], size=n_r,
        p=[1 - cfg.careless_share - cfg.fraud_share, cfg.careless_share, cfg.fraud_share],
    )
    vehicle = rng.choice(list(VEHICLE_SPEED_KMH), size=n_r, p=[0.6, 0.3, 0.1])
    riders = pd.DataFrame({
        "id": np.arange(1, n_r + 1),
        "name": [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n_r)],
        "home_zone_id": rng.integers(1, n_zones + 1, n_r),
        "vehicle_type": vehicle,
        "joined_at": start - pd.to_timedelta(rng.integers(30, 900, n_r), unit="D"),
        "rating": np.round(np.clip(rng.normal(4.6, 0.2, n_r), 3.5, 5.0), 2),
    })

    # ---- conditions
    cond = _make_conditions(rng, cfg)
    event, severity = _label_conditions(cond)
    zone_idx, hour_idx = np.meshgrid(np.arange(n_zones), np.arange(n_hours), indexing="ij")
    conditions = pd.DataFrame({
        "zone_id": zone_idx.ravel() + 1,
        "hour_start": start + pd.to_timedelta(hour_idx.ravel(), unit="h"),
        "rain_mm": np.round(cond["rain"].ravel(), 2),
        "traffic_level": np.round(cond["traffic"].ravel(), 3),
        "event_type": event.ravel(),
        "severity": severity.ravel(),
    })
    conditions.insert(0, "id", np.arange(1, len(conditions) + 1))

    # ---- deliveries
    n = cfg.n_deliveries
    day = rng.integers(0, cfg.n_days, n)
    hod = rng.choice(24, size=n, p=ORDERS_BY_HOUR)
    minute = rng.integers(0, 60, n)
    hour = day * 24 + hod
    activity = rng.gamma(2.0, 1.0, n_r)                      # some riders work more
    r = rng.choice(n_r, size=n, p=activity / activity.sum())  # 0-based rider index
    z = np.where(rng.random(n) < 0.8, riders["home_zone_id"].to_numpy()[r] - 1, rng.integers(0, n_zones, n))

    rain_h, traffic_h = cond["rain"][z, hour], cond["traffic"][z, hour]
    normal_traffic_h = cond["normal_traffic"][z, hour]
    flood_h, closure_h = cond["flood"][z, hour], cond["closure"][z, hour]

    distance = np.round(np.clip(rng.lognormal(np.log(3.0), 0.5, n), 0.5, 12.0), 2)
    free_speed = np.array([VEHICLE_SPEED_KMH[v] for v in vehicle])[r]
    # Congestion slows riders non-linearly: mild traffic costs little, gridlock a lot.
    normal_speed = free_speed / (1 + 3 * normal_traffic_h**2)
    actual_speed = (free_speed / (1 + 3 * traffic_h**2)
                    * (1 - np.minimum(rain_h / 30, 0.5))
                    * np.where(flood_h, 0.45, 1.0))
    detour = np.where(closure_h, rng.uniform(6, 15, n), 0.0)
    travel_normal = distance / normal_speed * 60
    travel_actual = distance / actual_speed * 60 + detour

    restaurant_wait = np.minimum(rng.exponential(4.0, n), 12.0)
    restaurant_extra = np.where(rng.random(n) < 0.025, rng.uniform(15, 35, n), 0.0)
    handover = 3 + rng.exponential(1.5, n)

    # Idle stops (tea breaks, personal errands): rider-caused by definition.
    prof = profile[r]
    long_stop_p = np.select([prof == "careless", prof == "fraudster"], [0.25, 0.08], 0.03)
    long_stop = np.where(
        rng.random(n) < long_stop_p,
        np.where(prof == "careless", rng.uniform(15, 40, n), rng.uniform(10, 25, n)),
        0.0,
    )
    idle = rng.exponential(np.where(prof == "careless", 1.5, 0.8)) + long_stop

    noise = rng.normal(0, 1.5, n)
    actual = np.maximum(travel_actual + restaurant_wait + restaurant_extra + handover + idle + noise, 5.0)
    external = (travel_actual - travel_normal) + restaurant_extra   # minutes caused by outside factors
    delay = external + idle + noise                                 # vs. a normal delivery

    picked_up_at = start + pd.to_timedelta(hour * 60 + minute, unit="m")
    deliveries = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "rider_id": r + 1,
        "zone_id": z + 1,
        "picked_up_at": picked_up_at,
        "distance_km": distance,
        "restaurant_wait_minutes": np.round(restaurant_wait + restaurant_extra, 1),
        "idle_minutes": np.round(idle, 1),
        "avg_speed_kmh": np.round(distance / (travel_actual / 60), 1),
        "actual_minutes": np.round(actual, 1),
    }).sort_values("picked_up_at", kind="stable").reset_index(drop=True)
    # Re-number so delivery ids follow time order (nicer to read), keep arrays aligned.
    order = deliveries["id"].to_numpy() - 1
    deliveries["id"] = np.arange(1, n + 1)
    external, idle, delay, r, z = external[order], idle[order], delay[order], r[order], z[order]
    rain_h, flood_h, closure_h, hour = rain_h[order], flood_h[order], closure_h[order], hour[order]
    restaurant_extra, prof = restaurant_extra[order], prof[order]

    # ---- who claims, and the ground-truth label
    ext_dominant = external >= idle
    late = delay >= CLAIM_MIN_DELAY
    u = rng.random(n)
    genuine = late & ext_dominant & (u < np.select([prof == "careless"], [0.40], 0.35))
    rider_caused = late & ~ext_dominant & (u < np.select([prof == "careless", prof == "fraudster"], [0.45, 0.40], 0.04))
    # Fraudsters also claim on deliveries with little or no outside disruption.
    fraud = (prof == "fraudster") & (external < 5) & ~rider_caused & (rng.random(n) < 0.10)

    label = np.full(n, None, dtype=object)
    label[genuine] = "genuine_external"
    label[rider_caused] = "rider_caused"
    label[fraud] = "fraudulent"
    claim_idx = np.flatnonzero(label != None)  # noqa: E711 (numpy elementwise)

    # Fraudsters keep a few photos to reuse; a stock pool is shared among them.
    stock_pool = [int(x) for x in rng.integers(0, 2**63, 8)]
    personal_pool = {rid: [int(x) for x in rng.integers(0, 2**63, 2)] for rid in np.flatnonzero(profile == "fraudster")}

    claims_rows, evidence_rows = [], []
    for claim_id, i in enumerate(claim_idx, start=1):
        lab = label[i]
        zone_name = ZONES[z[i]][0]
        reason = _claimed_reason(rng, lab, rain_h[i], flood_h[i], closure_h[i], restaurant_extra[i], external[i])
        if lab == "fraudulent":
            minutes = int(rng.integers(20, 46))
        else:
            minutes = int(round(max(delay[i], 10) + rng.normal(0, 3)))
        delivered_at = deliveries.at[i, "picked_up_at"] + pd.Timedelta(minutes=float(deliveries.at[i, "actual_minutes"]))
        submitted_at = delivered_at + pd.Timedelta(minutes=int(rng.integers(5, 121)))
        templates = CLAIM_TEMPLATES if rng.random() < 0.85 else CLAIM_TEMPLATES_NO_ID
        text = rng.choice(templates[reason]).format(zone=zone_name, id=int(deliveries.at[i, "id"]), m=minutes)

        claims_rows.append({
            "id": claim_id,
            "delivery_id": int(deliveries.at[i, "id"]),
            "rider_id": int(r[i] + 1),
            "submitted_at": submitted_at,
            "text": " ".join(text.split()),
            "reason_category": reason,
            "claimed_delay_minutes": float(minutes),
            "status": "pending",
            "true_label": lab,
        })
        photo = _photo_evidence(rng, lab, int(r[i]), deliveries.loc[i], ZONES[z[i]], stock_pool, personal_pool)
        if photo:
            evidence_rows.append({"id": len(evidence_rows) + 1, "claim_id": claim_id, **photo})

    evidence_cols = ["id", "claim_id", "kind", "file_path", "phash", "captured_at", "gps_lat", "gps_lon"]
    return SyntheticData(
        zones=zones,
        riders=riders,
        policies=pd.DataFrame([DEFAULT_POLICY]),
        conditions=conditions,
        deliveries=deliveries,
        claims=pd.DataFrame(claims_rows),
        claim_evidence=pd.DataFrame(evidence_rows, columns=evidence_cols),
        rider_profiles=pd.Series(profile, index=riders["id"], name="profile"),
    )


def _claimed_reason(rng, label, rain, flood, closure, restaurant_extra, external) -> str:
    """The reason the rider writes in the claim."""
    if label == "genuine_external":
        if restaurant_extra > 0 and restaurant_extra >= external - restaurant_extra:
            return "restaurant_delay"
        if flood:
            return "flooding"
        if closure:
            return "road_closure"
        return "heavy_rain" if rain >= 5 else "traffic_jam"
    if label == "rider_caused":
        # They blame something plausible, not the real cause (their idle time).
        if rain >= LIGHT_RAIN_MM:
            return "heavy_rain"
        return str(rng.choice(["traffic_jam", "restaurant_delay", "road_closure"], p=[0.6, 0.3, 0.1]))
    return str(rng.choice(["heavy_rain", "traffic_jam", "flooding", "road_closure"], p=[0.4, 0.35, 0.15, 0.1]))


def _near_duplicate(h: int, rng) -> int:
    """Flip 0-3 bits: a re-saved/re-compressed copy of the same photo."""
    for bit in rng.choice(64, size=int(rng.integers(0, 4)), replace=False):
        h ^= 1 << int(bit)
    return h & (2**64 - 1)


def _photo_evidence(rng, label, rider_idx, delivery, zone, stock_pool, personal_pool) -> dict | None:
    """Simulated photo metadata: perceptual hash, EXIF time and GPS."""
    attach_p = {"genuine_external": 0.40, "rider_caused": 0.20, "fraudulent": 0.55}[label]
    if rng.random() >= attach_p:
        return None
    _, lat, lon, _, _ = zone
    taken = delivery["picked_up_at"] + pd.Timedelta(minutes=float(rng.uniform(0, delivery["actual_minutes"])))
    photo = {
        "kind": "photo", "file_path": None,
        "phash": f"{int(rng.integers(0, 2**63)):016x}",
        "captured_at": taken,
        "gps_lat": round(lat + rng.normal(0, 0.005), 5),
        "gps_lon": round(lon + rng.normal(0, 0.005), 5),
    }
    if label == "fraudulent":
        if rng.random() < 0.70:   # reused photo
            pool = stock_pool if rng.random() < 0.5 else personal_pool[rider_idx]
            photo["phash"] = f"{_near_duplicate(int(rng.choice(pool)), rng):016x}"
        if rng.random() < 0.5:    # EXIF stripped
            photo.update(captured_at=None, gps_lat=None, gps_lon=None)
        else:                     # old photo from elsewhere
            photo["captured_at"] = taken - pd.Timedelta(days=int(rng.integers(3, 200)))
            photo["gps_lat"] = round(lat + rng.normal(0, 0.08), 5)
            photo["gps_lon"] = round(lon + rng.normal(0, 0.08), 5)
    return photo


if __name__ == "__main__":
    data = generate()
    print(f"riders={len(data.riders)} deliveries={len(data.deliveries)} conditions={len(data.conditions)}")
    print(data.claims["true_label"].value_counts().to_string())
