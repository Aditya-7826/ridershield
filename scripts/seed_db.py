"""Fill PostgreSQL with reproducible synthetic data, then print summary stats.

Usage:
    python scripts/seed_db.py                 # default: seed 42, 500 riders, 50k deliveries, 90 days
    python scripts/seed_db.py --seed 7 --deliveries 10000
    python scripts/seed_db.py --stats-only    # just print stats for what is already loaded

Seeding wipes the existing data in every table first. Run `alembic upgrade head` before.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import Connection, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generator import GeneratorConfig, SyntheticData, generate  # noqa: E402
from db.models import Base  # noqa: E402
from db.session import get_engine  # noqa: E402

# Parent tables first so foreign keys are satisfied.
LOAD_ORDER = ["zones", "riders", "policies", "conditions", "deliveries", "claims", "claim_evidence"]


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN/NaT turned into None."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def load(conn: Connection, data: SyntheticData) -> None:
    table_names = ", ".join(t.name for t in Base.metadata.sorted_tables)
    conn.execute(text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE"))
    for name in LOAD_ORDER:
        df: pd.DataFrame = getattr(data, name)
        if df.empty:
            continue
        table = Base.metadata.tables[name]
        rows = _records(df)
        for i in range(0, len(rows), 5000):
            conn.execute(table.insert(), rows[i : i + 5000])
        # We inserted explicit ids, so move each id sequence past the max.
        conn.execute(text(f"SELECT setval(pg_get_serial_sequence('{name}', 'id'), (SELECT MAX(id) FROM {name}))"))
        print(f"  loaded {name:<15} {len(rows):>7,} rows")


STATS_QUERIES = {
    "Row counts": """
        SELECT 'riders' AS table_name, COUNT(*) AS rows FROM riders
        UNION ALL SELECT 'zones', COUNT(*) FROM zones
        UNION ALL SELECT 'deliveries', COUNT(*) FROM deliveries
        UNION ALL SELECT 'conditions (zone-hours)', COUNT(*) FROM conditions
        UNION ALL SELECT 'claims', COUNT(*) FROM claims
        UNION ALL SELECT 'claim_evidence', COUNT(*) FROM claim_evidence
        UNION ALL SELECT 'policies', COUNT(*) FROM policies
    """,
    "Claims per ground-truth label": """
        SELECT c.true_label,
               COUNT(*) AS claims,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
               ROUND(AVG(c.claimed_delay_minutes)::numeric, 1) AS avg_claimed_min,
               COUNT(e.id) AS with_photo
        FROM claims c LEFT JOIN claim_evidence e ON e.claim_id = c.id
        GROUP BY c.true_label ORDER BY claims DESC
    """,
    "Condition events per zone (hours)": """
        SELECT z.name AS zone,
               SUM((c.event_type = 'rain')::int)         AS rain,
               SUM((c.event_type = 'heavy_rain')::int)   AS heavy_rain,
               SUM((c.event_type = 'flood')::int)        AS flood,
               SUM((c.event_type = 'traffic_jam')::int)  AS traffic_jam,
               SUM((c.event_type = 'road_closure')::int) AS road_closure,
               SUM((c.event_type <> 'none')::int)        AS total_event_hours
        FROM conditions c JOIN zones z ON z.id = c.zone_id
        GROUP BY z.name ORDER BY total_event_hours DESC
    """,
    "Average delivery time by condition": """
        SELECT c.event_type, COUNT(*) AS deliveries,
               ROUND(AVG(d.actual_minutes)::numeric, 1) AS avg_minutes
        FROM deliveries d
        JOIN conditions c ON c.zone_id = d.zone_id
                         AND c.hour_start = date_trunc('hour', d.picked_up_at)
        GROUP BY c.event_type ORDER BY avg_minutes DESC
    """,
}


def print_stats(conn: Connection) -> None:
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        for title, sql in STATS_QUERIES.items():
            print(f"\n== {title} ==")
            print(pd.read_sql(text(sql), conn).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--riders", type=int, default=500)
    parser.add_argument("--deliveries", type=int, default=50_000)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    if not args.stats_only:
        t0 = time.perf_counter()
        cfg = GeneratorConfig(seed=args.seed, n_riders=args.riders, n_deliveries=args.deliveries, n_days=args.days)
        print(f"Generating data (seed={cfg.seed}, riders={cfg.n_riders}, deliveries={cfg.n_deliveries}, days={cfg.n_days})...")
        data = generate(cfg)
        print("Loading into PostgreSQL...")
        with engine.begin() as conn:
            load(conn, data)
        print(f"Seeding finished in {time.perf_counter() - t0:.1f}s")

    with engine.connect() as conn:
        print_stats(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
