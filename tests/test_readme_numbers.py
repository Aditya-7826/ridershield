"""The README's results table must match evals/results/metrics.json exactly."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW_NAMES = {
    "rules-only baseline": "rules_only",
    "ML scoring (no agents)": "ml_scoring",
    "full pipeline, fallback intake": "full_fallback",
    "full pipeline, qwen2.5:3b intake": "full_llm",
}
COLUMNS = ["n_claims", "genuine_precision", "genuine_recall", "genuine_f1", "fraud_recall",
           "automation_rate", "manual_reviews", "auto_decision_error_rate"]


def test_readme_results_match_metrics_json():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "evals" / "results" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    assert "RESULTS_PLACEHOLDER" not in readme
    found = 0
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in ROW_NAMES:
            m = metrics[ROW_NAMES[cells[0]]]
            for col, cell in zip(COLUMNS, cells[1:]):
                assert float(cell) == round(float(m[col]), 3), f"{cells[0]} / {col}: README {cell} vs {m[col]}"
            found += 1
    assert found == len(ROW_NAMES)


def test_readme_has_no_ai_tool_mentions():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for word in ("claude", "chatgpt", "copilot", "generated with", "co-authored"):
        assert not re.search(rf"\b{re.escape(word)}\b", text), word
