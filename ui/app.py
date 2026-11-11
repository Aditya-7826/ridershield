"""RiderShield Streamlit UI: Rider view and Reviewer dashboard.

Run (with the API already running on :8000):
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.api_client import API_URL, ApiClient  # noqa: E402

st.set_page_config(page_title="RiderShield", page_icon="🛵", layout="wide")
api = ApiClient()

STATUS_TEXT = {
    "pending": ("⏳", "Being processed"),
    "auto_approved": ("✅", "Approved automatically"),
    "auto_rejected": ("❌", "Rejected automatically"),
    "in_review": ("🧑‍⚖️", "Waiting for a human reviewer"),
    "approved": ("✅", "Approved by a reviewer"),
    "rejected": ("❌", "Rejected by a reviewer"),
}
SIGNAL_LABELS = {
    "condition_support": "Delay not explained by conditions",
    "claim_inflation": "Claimed delay above real delay",
    "zone_disruption": "Zone not disrupted",
    "rider_anomaly": "Unusual rider behaviour",
    "image_check": "Photo problems",
    "claim_history": "Recent claim history",
}


def api_ok() -> bool:
    try:
        api.health()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach the RiderShield API at {API_URL}: {exc}\n\n"
                 "Start it with `uvicorn app.main:create_app --factory --port 8000`.")
        return False


def risk_chart(risk: dict) -> None:
    rows = [{"signal": SIGNAL_LABELS.get(k, k), "contribution": b["contribution"], "value": b["value"],
             "weight": b["weight"]} for k, b in risk["breakdown"].items() if b["used"]]
    df = pd.DataFrame(rows).set_index("signal")
    st.markdown(f"**Risk score {risk['score']:.2f}**  ·  auto-approve below {risk['thresholds']['auto_approve_below']}, "
                f"auto-reject from {risk['thresholds']['auto_reject_at_or_above']}")
    st.bar_chart(df[["contribution"]], horizontal=True, height=240)
    st.dataframe(df.round(3), width="stretch")


# --------------------------------------------------------------------------- rider view


def rider_view() -> None:
    st.header("🛵 File a delay claim")
    rider_id = st.number_input("Rider ID", min_value=1, value=int(st.session_state.get("rider_id", 93)), step=1)
    st.session_state["rider_id"] = rider_id
    try:
        deliveries = api.deliveries(int(rider_id))
    except RuntimeError as exc:
        st.warning(str(exc))
        deliveries = []

    options = {"I don't know / most recent": None}
    for d in deliveries:
        label = (f"#{d['id']} · {pd.Timestamp(d['picked_up_at']):%d %b %H:%M} · {d['zone']} · "
                 f"{d['actual_minutes']:.0f} min" + (" · already claimed" if d["claims"] else ""))
        options[label] = d["id"]

    with st.form("claim"):
        choice = st.selectbox("Which delivery was delayed?", list(options))
        text = st.text_area("What happened?", placeholder="e.g. The underpass near BTM was flooded, "
                                                         "my order was about 25 minutes late.")
        photo = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("Submit claim", type="primary")
    if submitted:
        if len(text.strip()) < 5:
            st.error("Please describe what happened.")
        else:
            file = (photo.name, photo.getvalue(), photo.type) if photo else None
            try:
                res = api.submit_claim(int(rider_id), text.strip(), options[choice], file)
                st.session_state["my_claim"] = res["claim_id"]
                st.success(f"Claim {res['claim_id']} submitted. Local AI models are checking it; "
                           "this can take a few minutes on a laptop.")
            except RuntimeError as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Claim status")
    claim_id = st.number_input("Claim ID", min_value=1, value=int(st.session_state.get("my_claim", 1)), step=1)
    st.button("Refresh")
    try:
        c = api.claim(int(claim_id))
    except RuntimeError as exc:
        st.info(str(exc))
        return
    icon, label = STATUS_TEXT.get(c["claim"]["status"], ("•", c["claim"]["status"]))
    st.markdown(f"### {icon} {label}")
    if c["job"] and c["job"].startswith("error"):
        st.error(f"Processing failed: {c['job']}")
    elif c["claim"]["status"] == "pending":
        st.info(f"Processing status: {c['job'] or 'waiting'} — press Refresh in a minute.")
    if c["decision"]:
        st.write(c["decision"]["explanation"])
    if c["payout"]:
        st.success(f"Payout: Rs {c['payout']['amount_inr']:.0f}")


# --------------------------------------------------------------------------- reviewer view


def stats_row() -> None:
    s = api.stats()
    cols = st.columns(5)
    cols[0].metric("Claims decided", s["decided"])
    cols[1].metric("Automation rate", "-" if s["automation_rate"] is None else f"{s['automation_rate']:.0%}")
    cols[2].metric("Approval rate", "-" if s["approval_rate"] is None else f"{s['approval_rate']:.0%}")
    cols[3].metric("Payouts", f"Rs {s['payouts']['total_inr']:,.0f}", f"{s['payouts']['count']} paid")
    cols[4].metric("Waiting for review", s["waiting_for_review"])


def evidence_panel(c: dict) -> None:
    ev = c.get("evidence") or {}
    if not ev.get("available"):
        st.warning(ev.get("reason", "No evidence available"))
        return
    d, cond, z, r = ev["delivery"], ev["conditions"], ev["zone_disruption"], ev["rider"]
    a, b = st.columns(2)
    with a:
        st.markdown("**Delivery**")
        st.write(f"#{d['id']} in {ev['zone_name']} · {d['distance_km']} km · took {d['actual_minutes']:.0f} min "
                 f"(normal day ≈ {d['eta_clear_minutes']:.0f} min)")
        st.write(f"Idle {d['idle_minutes']:.0f} min · restaurant wait {d['restaurant_wait_minutes']:.0f} min")
        st.markdown("**Conditions at pickup**")
        st.write(f"{cond['event_type'].replace('_', ' ')} · rain {cond['rain_mm']:.1f} mm/h · "
                 f"traffic {cond['traffic_level']:.2f}")
    with b:
        st.markdown("**Other deliveries in the zone**")
        if z["n_deliveries"] >= 4:
            st.write(f"{z['n_delayed']} of {z['n_deliveries']} delayed ({z['delayed_share']:.0%}, usually "
                     f"{z['baseline_share']:.0%}) · disrupted: {'yes' if z['disrupted'] else 'no'}")
        else:
            st.write(f"Only {z['n_deliveries']} other deliveries — too few to compare")
        st.markdown("**Rider**")
        st.write(f"More unusual than {r['anomaly_percentile']:.0%} of riders · {r['claims_last_7d']} claims in 7 days")
        for note in r["notes"]:
            st.caption(note)


def reviewer_view() -> None:
    st.header("🧑‍⚖️ Reviewer dashboard")
    stats_row()
    st.divider()
    queue = api.review_queue()
    st.subheader(f"Review queue ({len(queue)})")
    if not queue:
        st.info("No claims are waiting for review.")
        claim_id = st.number_input("Look up any claim", min_value=1, value=1, step=1)
    else:
        df = pd.DataFrame(queue)[["claim_id", "rider_id", "submitted_at", "risk_score", "text"]]
        st.dataframe(df, width="stretch", hide_index=True)
        claim_id = st.selectbox("Open claim", [q["claim_id"] for q in queue])

    try:
        c = api.claim(int(claim_id))
    except RuntimeError as exc:
        st.info(str(exc))
        return
    icon, label = STATUS_TEXT.get(c["claim"]["status"], ("•", c["claim"]["status"]))
    st.markdown(f"### Claim {claim_id} · {icon} {label}")
    st.write(f"> {c['claim']['text']}")

    left, right = st.columns([3, 2])
    with left:
        if c["decision"]:
            st.markdown("**Explanation**")
            st.write(c["decision"]["explanation"])
            rules = (c["decision"]["breakdown"] or {}).get("rules") or []
            for rule in rules:
                st.warning(f"Rule: {rule}")
        if c.get("risk") and c["risk"].get("available"):
            risk_chart(c["risk"])
    with right:
        if c["has_photo"]:
            img = api.photo(int(claim_id))
            if img:
                st.image(img, caption="Rider's photo", width="stretch")
        v = c.get("vision")
        if v:
            m = v["metadata"]
            st.write(f"Reused: **{'yes' if m['reused'] else 'no'}** · EXIF: {m['exif_status']}")
            for reason in m["reasons"]:
                st.caption(reason)
            if v.get("content_checked"):
                st.write(f"Vision model: {v['description']} "
                         f"({'matches' if v['matches_claim'] else 'does not match'}, "
                         f"confidence {v['confidence']:.2f})")

    st.markdown("**Evidence**")
    evidence_panel(c)

    if c["awaiting_review"]:
        st.divider()
        proposed = c.get("proposed_payout") or {}
        st.write(f"Payout if approved: Rs {proposed.get('amount_inr', 0):.0f}")
        with st.form("verdict"):
            reviewer = st.text_input("Your name", value=st.session_state.get("reviewer", ""))
            notes = st.text_area("Notes (optional)")
            col1, col2 = st.columns(2)
            approve = col1.form_submit_button("Approve", type="primary")
            reject = col2.form_submit_button("Reject")
        if approve or reject:
            if not reviewer.strip():
                st.error("Please enter your name.")
            else:
                st.session_state["reviewer"] = reviewer.strip()
                res = api.review(int(claim_id), "approve" if approve else "reject", reviewer.strip(), notes or None)
                st.success(f"Claim {claim_id} {res['status']}"
                           + (f" · payout Rs {res['payout']['amount_inr']:.0f}" if res.get("payout") else ""))
    elif c.get("review"):
        rv = c["review"]
        st.info(f"Reviewed by {rv['reviewer']}: {rv['verdict']}" + (f" — {rv['notes']}" if rv["notes"] else ""))


# --------------------------------------------------------------------------- main

st.sidebar.title("RiderShield")
st.sidebar.caption("AI micro-insurance for delivery riders · all models run locally")
view = st.sidebar.radio("View", ["Rider", "Reviewer"], key="view")
if api_ok():
    rider_view() if view == "Rider" else reviewer_view()
