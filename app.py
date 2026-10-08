"""Streamlit demo: fraud analyst review queue.
Run: streamlit run app.py   (after python -m src.train)

Replays one day of the held-out TEST period as a fraud team would see it: transactions ranked by risk,
split into tiers, each alert with its top reasons. The model never blocks anything by itself."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Fraud Transaction Risk: Analyst Queue", layout="wide")
st.title("Fraud / Transaction Risk Detection")
st.caption("UneeQ Interns · Machine Learning Track · October 2026 · decision support for human reviewers")

TIER_LABEL = {"hold_verify": "HOLD + verify customer", "review": "Analyst review", "approve": "Approve"}


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "fraud_model.joblib")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    scored = pd.read_parquet(ROOT / "data" / "processed" / "test_scored.parquet")
    scored["day"] = scored["ts"].dt.date
    return model, meta, scored


try:
    model, meta, scored = load_artifacts()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

thr = meta["thresholds"]
with st.sidebar:
    st.header("Policy")
    review_thr = st.slider("Review threshold", 0.001, 0.99, float(thr["review"]), 0.001, format="%.3f")
    hold_thr = st.slider("Hold threshold", review_thr, 0.999, max(float(thr["hold"]), review_thr), 0.001,
                         format="%.3f")
    st.caption(f"Defaults were chosen on validation for a capacity of "
               f"{meta['alerts_per_day_capacity']} alerts/day (see the risk threshold memo).")
    show_truth = st.toggle("Show ground truth (hindsight, for the demo only)", value=False)
    days = sorted(scored["day"].unique())
    day = st.selectbox("Day (test period)", days, index=len(days) // 2)

d = scored[scored["day"] == day].copy()
d["tier"] = pd.cut(d["score"], [-1, review_thr, hold_thr, 2], right=False,
                   labels=["approve", "review", "hold_verify"]).astype(str)
alerts = d[d["tier"] != "approve"].sort_values("score", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions today", f"{len(d):,}")
c2.metric("Alerts for humans", f"{len(alerts):,}", f"capacity {meta['alerts_per_day_capacity']}",
          delta_color="off")
c3.metric("Hold + verify", f"{(d['tier'] == 'hold_verify').sum():,}")
c4.metric("Analyst review", f"{(d['tier'] == 'review').sum():,}")

if show_truth:
    frauds = d[d["is_fraud"] == 1]
    caught = frauds[frauds["tier"] != "approve"]
    t1, t2, t3 = st.columns(3)
    t1.metric("Frauds that day", f"{len(frauds)}")
    t2.metric("Reached a human", f"{len(caught)}")
    t3.metric("Fraud $ reached a human", f"${caught['amt'].sum():,.0f} of ${frauds['amt'].sum():,.0f}")

st.subheader(f"Alert queue: {day}")
if alerts.empty:
    st.info("No alerts at these thresholds for this day.")
    st.stop()

cols = ["ts", "cc_num", "category", "amt", "score", "tier"] + (["is_fraud"] if show_truth else [])
view = alerts[cols].assign(tier=alerts["tier"].map(TIER_LABEL)).rename(
    columns={"ts": "time", "cc_num": "card", "amt": "amount ($)", "score": "risk score"})
event = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun",
                     selection_mode="single-row",
                     column_config={"risk score": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                   format="%.3f")})

rows = event.selection.rows if event and event.selection else []
pick = alerts.iloc[rows[0]] if rows else alerts.iloc[0]
st.subheader("Why was this flagged?")
st.caption("Select a row above. Top drivers pushing this transaction's risk up (SHAP). "
           "The analyst decides; the model only prioritises.")
from src.explain import reason_codes, top_drivers   # lazy import: keeps first page load fast

x = pick.to_frame().T[meta["feature_columns"]].astype({c: float for c in meta["feature_columns"]
                                                        if c != "category"})
left, right = st.columns([1, 1])
with left:
    st.markdown(f"**{pick['cc_num']}** · {pick['ts']:%H:%M} · {pick['category']} · **${pick['amt']:,.2f}** · "
                f"risk {pick['score']:.3f} → **{TIER_LABEL[pick['tier']]}**")
    for r in reason_codes(model, x)[0]:
        st.markdown(f"- {r}")
with right:
    st.bar_chart(top_drivers(model, x, k=6).set_index("feature")["shap"], horizontal=True)
