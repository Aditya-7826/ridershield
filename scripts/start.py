"""Start the RiderShield API and UI together with one command.

    python scripts/start.py            # uses the local LLM through Ollama
    python scripts/start.py --no-llm   # deterministic fallbacks, no Ollama needed

Then open http://localhost:8501. Press Ctrl+C once to stop both.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait_for(url: str, seconds: int = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="run without Ollama (deterministic fallbacks)")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    env = {**os.environ, "API_URL": f"http://localhost:{args.api_port}"}
    if args.no_llm:
        env["USE_LLM"] = "0"

    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:create_app", "--factory",
                            "--port", str(args.api_port)], cwd=ROOT, env=env)
    print(f"Starting API on http://localhost:{args.api_port} (loading models)...")
    if not wait_for(f"http://localhost:{args.api_port}/health"):
        print("API did not start. Is Docker/PostgreSQL running? Try: docker compose up -d")
        api.terminate()
        return 1

    # Bind to localhost only: the app has no login, so it should not be reachable from the network.
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", str(args.ui_port),
                           "--server.address", "localhost", "--server.headless", "true",
                           "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env)
    url = f"http://localhost:{args.ui_port}"
    if wait_for(f"{url}/_stcore/health"):
        print(f"\nRiderShield is running:\n  UI       {url}\n  API docs http://localhost:{args.api_port}/docs\n"
              "Press Ctrl+C to stop.\n")
        if not args.no_browser:
            webbrowser.open(url)
    else:
        print("UI did not start; check the messages above.")

    try:
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in (ui, api):
            if p.poll() is None:
                p.terminate()
        for p in (ui, api):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
