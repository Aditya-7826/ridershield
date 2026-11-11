"""Database connection helpers.

The URL is built from the POSTGRES_* variables in .env so that the app,
Alembic and the scripts all point at the same database.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


def database_url() -> str:
    """Return a SQLAlchemy URL using the psycopg (v3) driver."""
    if url := os.getenv("DATABASE_URL"):
        return url
    user = os.getenv("POSTGRES_USER", "ridershield")
    password = os.getenv("POSTGRES_PASSWORD", "ridershield")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    name = os.getenv("POSTGRES_DB", "ridershield")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"


_engine: Engine | None = None


def get_engine() -> Engine:
    """Create the engine once and reuse it (small pool for a low-RAM laptop)."""
    global _engine
    if _engine is None:
        _engine = create_engine(database_url(), pool_size=5, max_overflow=5, pool_pre_ping=True)
    return _engine


SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Open a session, commit on success, roll back on error."""
    session = SessionLocal(bind=get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
