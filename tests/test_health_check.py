"""Unit tests for the pure helpers in scripts/health_check.py."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "health_check", Path(__file__).resolve().parents[1] / "scripts" / "health_check.py"
)
health_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(health_check)


def test_exact_tag_matches():
    assert health_check.model_is_installed("qwen2.5:3b", ["qwen2.5:3b", "moondream:latest"])


def test_bare_name_means_latest():
    assert health_check.model_is_installed("moondream", ["moondream:latest"])


def test_missing_model():
    assert not health_check.model_is_installed("qwen2.5vl:3b", ["qwen2.5:3b"])


def test_different_tag_is_not_a_match():
    assert not health_check.model_is_installed("qwen2.5:3b", ["qwen2.5:7b"])
