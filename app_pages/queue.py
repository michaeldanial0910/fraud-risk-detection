import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (FRAUD, LEGIT, TIER_SHORT, alert_card, figures, load_meta, load_model, load_scored,
                        money, need_model, use_altair_theme)
from src.evaluate import assign_tier
from src.explain import reason_codes, top_drivers

use_altair_theme()


@st.cache_data(show_spinner=False)
def explain(_x: pd.DataFrame, key: str):
    """Reason codes + SHAP drivers for one alert, cached per transaction (key) so reruns are instant."""
    model = load_model()
    return reason_codes(model, _x)[0], top_drivers(model, _x, k=6)


st.title("Today's fraud review queue")
st.html('<p class="lede">One day of held-out card transactions, replayed as the fraud team would see it. The model '
        'never blocks a payment: it decides which transactions a person looks at, and in what order.</p>')

scored, meta, model = load_scored(), load_meta(), load_model()
if scored is None or model is None:
    need_model()
    st.stop()

thr = meta["thresholds"]
days = sorted(scored["day"].unique())
if "queue_day" not in st.session_state:
    st.session_state.queue_day = days[len(days) // 2]

c1, c2 = st.columns([1, 1.6], vertical_alignment="bottom")
day = c1.selectbox("Day (test period, never seen in training)", days, key="queue_day",
                   format_func=lambda d: f"{d:%a %d %b %Y}")
show_truth = c2.toggle("Show what really happened (hindsight)", key="show_truth")

d = scored[scored["day"] == day].copy()
d["tier"] = assign_tier(d["score"], thr["review"], thr["hold"])
alerts = d[d["tier"] != "approve"].sort_values("score", ascending=False).reset_index(drop=True)

items = [(f"{len(d):,}", "transactions today"),
         (f"{len(alerts)}", f"sent to a person (team capacity {meta['alerts_per_day_capacity']}/day)"),
         (money(alerts["amt"].sum()), "value of the flagged transactions")]
if show_truth:
    frauds = d[d["is_fraud"] == 1]
    caught = frauds[frauds["tier"] != "approve"]
    items += [(f"{len(caught)} of {len(frauds)}", "frauds that reached a person"),
              (f"{(alerts['is_fraud'] == 1).mean():.0%}" if len(alerts) else "n/a", "of alerts were real fraud")]
st.html(figures(items))

if alerts.empty:
    st.info("No alerts on this day at the current thresholds.")
    st.stop()

left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("Alerts, highest risk first")
    view = pd.DataFrame({
        "time": alerts["ts"].dt.strftime("%H:%M"), "category": alerts["category"],
        "amount": alerts["amt"], "risk": alerts["score"], "action": alerts["tier"].map(TIER_SHORT)})
    if show_truth:
        view["really"] = alerts["is_fraud"].map({1: "fraud", 0: "genuine"})
    event = st.dataframe(
        view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 430), key="queue_table",
        on_select="rerun", selection_mode="single-row",
        column_config={"time": st.column_config.TextColumn(width=60),
                       "category": st.column_config.TextColumn(width=120),
                       "amount": st.column_config.NumberColumn(format="$%.2f", width=90),
                       "risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", width=110),
                       "action": st.column_config.TextColumn(width=110)})
    st.html('<p class="muted">Click a row to open the alert.</p>')

rows = event.selection.rows if event and event.selection else []
pick = alerts.iloc[rows[0] if rows else 0]
x = pick.to_frame().T[meta["feature_columns"]]
x = x.astype({c: float for c in meta["feature_columns"] if c != "category"})
with right:
    st.subheader("Why this one?")
    reasons, drv = explain(x, f"{pick['ts']}|{pick['cc_num']}|{pick['amt']}")
    st.html(alert_card(pick, reasons, thr["review"], thr["hold"], show_truth))
    drv = drv.copy()
    drv["direction"] = drv["shap"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    bars = alt.Chart(drv).mark_bar(cornerRadius=3, height=16).encode(
        x=alt.X("shap:Q", title="contribution to the risk score (SHAP, log-odds)"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[FRAUD, LEGIT]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", alt.Tooltip("shap:Q", format="+.2f")])
    st.altair_chart(bars.properties(height=210), theme=None, width="stretch")
