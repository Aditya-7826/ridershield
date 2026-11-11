"""Compare three claim-handling configurations on held-out labelled claims.

    python -m evals.run                    # rules, ML, full pipeline + LLM sample (if Ollama is up)
    python -m evals.run --no-llm           # skip the slow LLM sample
    python -m evals.run --llm-sample 30    # smaller LLM sample (per label, fraud capped by availability)

Configurations
  1. rules_only   parametric trigger, no ML: approve if the zone-hour had a recorded
                  disruption (heavy rain, flood, traffic jam, road closure) or the
                  restaurant wait was 15+ min; light rain -> human review; else reject.
  2. ml_scoring   weighted risk score + thresholds on the structured claim fields;
                  no agents, no photo checks, no decision rules.
  3. full         the LangGraph pipeline: intake from the claim TEXT, evidence,
                  photo checks, risk, decision rules. Run twice:
                    full_fallback  all held-out claims, deterministic intake/explanations
                    full_llm       up to --llm-sample claims per label, qwen2.5:3b intake (CPU is slow)

Held-out = claims on deliveries after ml.config.TRAIN_CUTOFF. Nothing is
written to the database (dry-run mode). Results go to evals/results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agents.config import TEXT_MODEL  # noqa: E402
from agents.deps import Deps  # noqa: E402
from agents.graph import build_graph, process_claim  # noqa: E402
from agents.llm import OfflineLLM, OllamaClient  # noqa: E402
from db.session import get_engine  # noqa: E402
from evals.metrics import DECISIONS, LABELS, compute_metrics, confusion  # noqa: E402
from ml.claim_scoring import Models, decide, load_tables, score_claims  # noqa: E402
from ml.config import TRAIN_CUTOFF  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LLM_CSV = RESULTS / "full_llm_claims.csv"
MAJOR_EVENTS = {"heavy_rain", "flood", "traffic_jam", "road_closure"}


# --------------------------------------------------------------------------- configurations


def run_rules_only(claims: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    d = claims.merge(frame[["id", "event_type", "restaurant_wait_minutes"]], left_on="delivery_id", right_on="id",
                     suffixes=("", "_d"))

    def rule(r) -> str:
        if r["event_type"] in MAJOR_EVENTS or r["restaurant_wait_minutes"] >= 15:
            return "auto_approve"
        if r["event_type"] == "rain":
            return "human_review"
        return "auto_reject"

    return pd.DataFrame({"claim_id": d["id"], "true_label": d["true_label"], "decision": d.apply(rule, axis=1)})


def run_ml_scoring(claims, tables, models) -> pd.DataFrame:
    # No photo checks in this configuration: drop evidence so image_check is unavailable.
    t = {**tables, "evidence": tables["evidence"].iloc[0:0]}
    s = score_claims(claims, t, models, rider_features_as_of=TRAIN_CUTOFF)
    s["decision"] = s["risk_score"].apply(lambda x: decide(x, models.cfg))
    return s[["claim_id", "true_label", "decision", "risk_score"]]


def run_full(claims, deps: Deps, label: str, csv_path: Path | None = None) -> pd.DataFrame:
    """Run the real graph on each claim (dry run, in-memory checkpoints)."""
    graph = build_graph(deps, checkpointer=InMemorySaver())
    done = pd.read_csv(csv_path) if csv_path and csv_path.exists() else pd.DataFrame()
    done_ids = set(done["claim_id"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    todo = claims[~claims["id"].isin(done_ids)]
    if done_ids:
        print(f"  [{label}] resuming: {len(done_ids)} already done, {len(todo)} to go")
    t0 = time.perf_counter()
    for i, (_, c) in enumerate(todo.iterrows(), start=1):
        st = process_claim(graph, deps, int(c["id"]))
        intake = st.get("intake", {})
        rows.append({
            "claim_id": int(c["id"]), "true_label": c["true_label"], "decision": st["decision"]["decision"],
            "risk_score": st.get("risk", {}).get("score"),
            "rules": "; ".join(st["decision"]["rules"]),
            "intake_source": intake.get("source"),
            "intake_delivery_ok": intake.get("resolved_delivery_id") == int(c["delivery_id"]),
            "intake_reason": intake.get("reason_category"), "true_reason": c["reason_category"],
            "intake_delay": intake.get("claimed_delay_minutes"), "true_delay": c["claimed_delay_minutes"],
        })
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        if i % 10 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            print(f"  [{label}] {i}/{len(todo)} claims, {el / i:.1f}s per claim", flush=True)
    return pd.DataFrame(rows)


def stratified_sample(claims: pd.DataFrame, per_label: int) -> pd.DataFrame:
    parts = [g.sample(min(per_label, len(g)), random_state=42) for _, g in claims.groupby("true_label")]
    return pd.concat(parts).sort_values("submitted_at")


def ollama_up() -> bool:
    import httpx

    from agents.config import OLLAMA_HOST
    try:
        return httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


# --------------------------------------------------------------------------- reporting


def plot_confusions(results: dict[str, pd.DataFrame], path: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.9), constrained_layout=True)
    for ax, name in zip(axes, names):
        cm = confusion(results[name])
        share = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(DECISIONS)):
                v = share.iat[i, j]
                ax.text(j, i, f"{cm.iat[i, j]}\n{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.6 else "black")
        ax.set_xticks(range(3), ["approve", "review", "reject"])
        ax.set_yticks(range(3), ["genuine", "rider-caused", "fraud"])
        ax.set_title(f"{name} (n={int(cm.to_numpy().sum())})", fontsize=10)
        ax.set_xlabel("decision")
    axes[0].set_ylabel("true label")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("-" if pd.isna(r[c]) else
                                     (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    return "\n".join(out)


def write_summary(results: dict[str, pd.DataFrame], meta: dict, llm_intake: dict | None) -> None:
    metrics = {k: compute_metrics(v) for k, v in results.items()}
    base_reviews = metrics["rules_only"]["manual_reviews"]
    rows = []
    for name, m in metrics.items():
        rows.append({
            "configuration": name, "claims": m["n_claims"],
            "genuine precision": m["genuine_precision"], "genuine recall": m["genuine_recall"],
            "genuine F1": m["genuine_f1"], "fraud recall": m["fraud_recall"],
            "automation rate": m["automation_rate"], "manual reviews": m["manual_reviews"],
            "auto-decision error": m["auto_decision_error_rate"],
        })
    table = pd.DataFrame(rows)

    n = meta["n_heldout"]
    lines = [
        "# RiderShield evaluation",
        "",
        f"_Generated by `python -m evals.run` on {meta['generated']}. Do not edit by hand._",
        "",
        "**Data:** synthetic claims from `data/generator.py` (seed 42). Held-out set = "
        f"{n} claims on deliveries after {TRAIN_CUTOFF:%Y-%m-%d} "
        f"({meta['label_counts']}). No model was trained or tuned on them.",
        "",
        "## Results",
        "",
        md_table(table),
        "",
        "- **genuine P/R/F1:** a claim counts as identified when it is auto-approved; genuine claims sent to "
        "review count as misses for recall (they still reach a human).",
        "- **fraud recall:** share of fraudulent claims that were *not* auto-approved.",
        "- **auto-decision error:** wrong automatic decisions / automatic decisions (auto-approving a "
        "non-genuine claim or auto-rejecting a genuine one).",
        "",
        "## Manual review workload",
        "",
    ]
    for name, m in metrics.items():
        share = m["manual_reviews"] / m["n_claims"]
        vs_rules = ""
        if name != "rules_only" and m["n_claims"] == metrics["rules_only"]["n_claims"] and base_reviews:
            change = 1 - m["manual_reviews"] / base_reviews
            vs_rules = f"; {abs(change):.0%} {'fewer' if change >= 0 else 'more'} than the rules-only baseline"
        lines.append(f"- **{name}:** {m['manual_reviews']} of {m['n_claims']} claims need a human "
                     f"({share:.1%}); {1 - share:.1%} fewer reviews than checking every claim manually{vs_rules}.")
    lines += ["", "## Wrong automatic decisions", ""]

    def claims(k: int) -> str:
        return f"{k} claim" if k == 1 else f"{k} claims"

    for name, m in metrics.items():
        lines.append(f"- **{name}:** auto-approved but not genuine: {claims(m['wrong_auto_approvals'])}; "
                     f"auto-rejected but genuine: {claims(m['wrong_auto_rejections'])}.")
    lines += ["", "## Confusion matrices", "", "![Confusion matrices](confusion_matrix.png)", ""]
    for name, df in results.items():
        cm = confusion(df).reset_index().rename(columns={"true_label": f"{name}: true \\ decision"})
        lines += [md_table(cm), ""]

    if llm_intake:
        lines += [
            "## LLM intake (qwen2.5:3b, CPU)",
            "",
            f"The `full_llm` row runs the same pipeline with the local LLM parsing the claim text, on a stratified "
            f"sample of {llm_intake['n']} held-out claims (up to {llm_intake['per_label']} per label, sampled with "
            f"seed 42) because CPU inference is slow on the 8 GB test laptop: about {llm_intake['sec_per_claim']:.0f}s "
            f"per claim on average, with the first claims taking over a minute while the model loaded under memory "
            f"pressure. LLM explanations were switched off for this run; they do not affect the decision.",
            "",
            f"- delivery matched correctly: {llm_intake['delivery_ok']:.1%}",
            f"- reason category matches the claim's category: {llm_intake['reason_ok']:.1%}"
            + (f" (errors: {', '.join(f'{k} ({v})' for k, v in llm_intake['reason_errors'].items())})"
               if llm_intake.get("reason_errors") else ""),
            f"- claimed delay extracted within 1 minute: {llm_intake['delay_ok']:.1%}",
            f"- LLM output missing a number that is written in the text, filled by the regex cross-check: "
            f"{llm_intake['regex_filled']:.1%} of claims (order number missing or not found in the text: "
            f"{llm_intake['regex_filled_delivery']:.1%}; delay: {llm_intake['regex_filled_delay']:.1%})",
            f"- LLM failures (fell back to regex completely): {llm_intake['fallbacks']}",
            "",
            f"So the 3B model mostly gets the reason right ({llm_intake['reason_ok']:.0%}) but is unreliable at "
            "copying numbers into the right JSON field; the deterministic cross-check is what makes the intake "
            "dependable. On these template-based texts the regex alone does better still; the LLM's value is on "
            "free-form text the templates don't cover.",
            "",
            "For comparison, the deterministic fallback intake on the same sample: "
            f"delivery {llm_intake['fb_delivery_ok']:.1%}, reason {llm_intake['fb_reason_ok']:.1%}, "
            f"delay {llm_intake['fb_delay_ok']:.1%}.",
            "",
        ]
    lines += [
        "## Caveats",
        "",
        "- The data is synthetic and the generator makes the signals cleaner than real life, so absolute numbers "
        "are optimistic. The comparison between configurations is the meaningful part.",
        "- Seeded photos exist only as metadata (hash, EXIF), so the vision model's content check is not part of "
        "these numbers; photo reuse and EXIF checks are.",
        "- `ml_scoring` uses a rider-behaviour snapshot at the train cutoff; the agent pipeline computes it per "
        "claim from history up to the claim time. Both use only past data.",
        "- Risk weights were set by hand; thresholds were calibrated on training-period claims only.",
        "- The synthetic claim texts come from a small set of templates, so the regex fallback intake parses them "
        "almost perfectly. Real riders write far messier text; that is the job of the LLM intake, and the "
        "`full_llm` row shows how the local model does on the same claims.",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics, "llm_intake": llm_intake},
                                                     indent=2, default=str), encoding="utf-8")


def intake_accuracy(df: pd.DataFrame, fb: pd.DataFrame, seconds: float) -> dict:
    def acc(x: pd.DataFrame) -> tuple[float, float, float]:
        delay_ok = (x["intake_delay"] - x["true_delay"]).abs() <= 1
        return (float(x["intake_delivery_ok"].mean()), float((x["intake_reason"] == x["true_reason"]).mean()),
                float(delay_ok.mean()))
    d, r, dl = acc(df)
    fb = fb[fb["claim_id"].isin(df["claim_id"])]
    fd, fr, fdl = acc(fb)
    src = df["intake_source"].astype(str)
    wrong = df[df["intake_reason"] != df["true_reason"]]
    reason_errors = {f"{t} -> {p}": int(n) for (t, p), n in
                     wrong.groupby(["true_reason", "intake_reason"]).size().items()}
    return {"n": int(len(df)), "sec_per_claim": seconds, "delivery_ok": d, "reason_ok": r, "delay_ok": dl,
            "regex_filled": float(src.str.startswith("llm+regex").mean()),
            "regex_filled_delivery": float(src.str.contains("delivery_id").mean()),
            "regex_filled_delay": float(src.str.contains("claimed_delay_minutes").mean()),
            "reason_errors": reason_errors,
            "fallbacks": int((df["intake_source"] == "fallback").sum()),
            "fb_delivery_ok": fd, "fb_reason_ok": fr, "fb_delay_ok": fdl}


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM sample")
    ap.add_argument("--llm-sample", type=int, default=30, help="claims per label for the LLM run (default 30)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved LLM results and start over")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    models = Models.load()
    tables = load_tables(engine)
    frame = tables["frame"]
    test_ids = set(frame.loc[frame["picked_up_at"] >= TRAIN_CUTOFF, "id"])
    claims = tables["claims"]
    heldout = claims[claims["delivery_id"].isin(test_ids) & claims["true_label"].notna()].copy()
    print(f"Held-out claims: {len(heldout)}  {heldout['true_label'].value_counts().to_dict()}")

    results: dict[str, pd.DataFrame] = {}
    t = time.perf_counter()
    results["rules_only"] = run_rules_only(heldout, frame)
    print(f"rules_only done ({time.perf_counter() - t:.1f}s)")
    t = time.perf_counter()
    results["ml_scoring"] = run_ml_scoring(heldout, tables, models)
    print(f"ml_scoring done ({time.perf_counter() - t:.1f}s)")

    base = Deps(engine=engine, llm=OfflineLLM(), models=models, dry_run=True, llm_explanations=False)
    t = time.perf_counter()
    results["full_fallback"] = run_full(heldout, base, "full_fallback")
    print(f"full_fallback done ({time.perf_counter() - t:.1f}s)")
    results["full_fallback"].to_csv(RESULTS / "full_fallback_claims.csv", index=False)

    llm_intake = None
    if args.no_llm:
        print("Skipping LLM sample (--no-llm)")
    elif not ollama_up():
        print("Ollama is not reachable: skipping LLM sample")
    else:
        if args.fresh and LLM_CSV.exists():
            LLM_CSV.unlink()
        sample = stratified_sample(heldout, args.llm_sample)
        print(f"full_llm: {len(sample)} claims with {TEXT_MODEL} (results saved after every claim)")
        llm = OllamaClient()
        t = time.perf_counter()
        results["full_llm"] = run_full(sample, replace(base, llm=llm), "full_llm", LLM_CSV)
        per_claim = (time.perf_counter() - t) / max(len(sample), 1)
        saved = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8")) if (RESULTS / "metrics.json").exists() else {}
        if llm.calls == 0 and (saved.get("llm_intake") or {}).get("sec_per_claim"):
            per_claim = saved["llm_intake"]["sec_per_claim"]   # everything was resumed from the CSV
        llm_intake = intake_accuracy(results["full_llm"], results["full_fallback"], per_claim)
        llm_intake["per_label"] = args.llm_sample

    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_heldout": int(len(heldout)),
        "label_counts": ", ".join(f"{k}: {v}" for k, v in heldout["true_label"].value_counts().items()),
        "thresholds": models.cfg.thresholds,
        "weights": models.cfg.weights,
        "text_model": TEXT_MODEL,
    }
    plot_confusions(results, RESULTS / "confusion_matrix.png")
    write_summary(results, meta, llm_intake)
    for name, df in results.items():
        m = compute_metrics(df)
        print(f"{name:<14} n={m['n_claims']:<4} F1={m['genuine_f1']:.3f} fraud_recall={m['fraud_recall']} "
              f"automation={m['automation_rate']:.3f} auto_error={m['auto_decision_error_rate']:.3f}")
    print(f"Wrote {RESULTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
