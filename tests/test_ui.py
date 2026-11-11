"""Smoke tests for the Streamlit UI (no browser needed)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def test_ui_shows_helpful_error_when_api_is_down(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")   # nothing listens here
    import importlib

    import ui.api_client
    importlib.reload(ui.api_client)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Cannot reach the RiderShield API" in e.value for e in at.error)
    assert at.sidebar.radio(key="view").options == ["Rider", "Reviewer"]
    importlib.reload(ui.api_client)
