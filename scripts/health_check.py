"""Check that the local services RiderShield needs are up.

Checks:
  1. PostgreSQL is reachable with the credentials in .env
  2. Ollama is running
  3. The text and vision models named in .env are installed

Run with: python scripts/health_check.py
Exits with code 0 if everything passes, 1 otherwise.
"""
from __future__ import annotations

import os
import sys

import httpx
import psycopg
from dotenv import load_dotenv

load_dotenv()


def check_postgres() -> tuple[bool, str]:
    """Open a connection and run SELECT version()."""
    try:
        with psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            user=os.getenv("POSTGRES_USER", "ridershield"),
            password=os.getenv("POSTGRES_PASSWORD", "ridershield"),
            dbname=os.getenv("POSTGRES_DB", "ridershield"),
            connect_timeout=5,
        ) as conn:
            version = conn.execute("SELECT version()").fetchone()[0]
        return True, version.split(",")[0]
    except psycopg.Error as exc:
        return False, f"{exc.__class__.__name__}: {exc}".strip() + "  ->  try: docker compose up -d"


def installed_ollama_models(host: str) -> list[str]:
    """Return model names from Ollama's /api/tags endpoint."""
    resp = httpx.get(f"{host}/api/tags", timeout=5)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def model_is_installed(wanted: str, installed: list[str]) -> bool:
    """Match 'name' or 'name:tag'; a bare name means ':latest'."""
    if ":" not in wanted:
        wanted = f"{wanted}:latest"
    return wanted in installed


def main() -> int:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    models = {
        "text": os.getenv("TEXT_MODEL", "qwen2.5:3b"),
        "vision": os.getenv("VISION_MODEL", "qwen2.5vl:3b"),
    }
    results: list[tuple[str, bool, str]] = []

    ok, detail = check_postgres()
    results.append(("PostgreSQL", ok, detail))

    try:
        installed = installed_ollama_models(host)
        results.append(("Ollama", True, f"running at {host}"))
        for role, name in models.items():
            present = model_is_installed(name, installed)
            hint = "installed" if present else f"missing  ->  run: ollama pull {name}"
            results.append((f"{role} model {name}", present, hint))
    except httpx.HTTPError as exc:
        results.append(("Ollama", False, f"not reachable at {host} ({exc.__class__.__name__})  ->  start the Ollama app"))

    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")

    all_ok = all(ok for _, ok, _ in results)
    print("\nHealth check passed." if all_ok else "\nHealth check FAILED.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
