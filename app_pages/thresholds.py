import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (INK, INK_SOFT, MODEL_COLOR, TIER_COLOR, TIER_NAME, dot, figures, load_csv, load_meta, money,
                        pct, table, use_altair_theme)

use_altair_theme()
st.title("Where to draw the line")
st.html('<p class="lede">Every alert costs analyst time, a false alarm also annoys a genuine customer, and a missed '
        'fraud costs its full amount. Move the threshold and watch the trade-off. The threshold was chosen on '
        'validation (Jan–Mar 2020) and only then checked on the test months.</p>')

curve, meta = load_csv("cost_curve_val.csv"), load_meta()
if curve is None or not meta:
    st.warning("reports/cost_curve_val.csv is missing. Run `python -m src.train`.")
    st.stop()

thr, costs, cap = meta["thresholds"], meta["costs"], meta["alerts_per_day_capacity"]
grid = [round(t, 3) for t in curve["threshold"]]
t = st.select_slider("Review threshold: risk scores at or above this go to a person", options=grid,
                     value=min(grid, key=lambda g: abs(g - thr["review"])), key="thr_pick",
                     format_func=lambda v: f"{v:.3f}")
r = curve.loc[(curve["threshold"] - t).abs().idxmin()]
best = curve.loc[(curve["threshold"] - thr["review"]).abs().idxmin()]
st.html(figures([
    (f"{r.alerts_per_day:.0f}", f"alerts per day (capacity {cap})"),
    (pct(r.recall), "of frauds caught"),
    (pct(r.dollar_recall), "of fraud dollars caught"),
    (pct(r.precision, 0), "of alerts are real fraud"),
    (money(r.cost), f"total cost over the quarter (chosen threshold: {money(best.cost)})"),
]))

c1, c2 = st.columns(2, gap="large")
x = alt.X("threshold:Q", scale=alt.Scale(type="log"), title="threshold (log scale)",
          axis=alt.Axis(values=[0.002, 0.01, 0.03, 0.1, 0.3, 1], format="~g"))
marks = pd.DataFrame({"threshold": [thr["review"], t], "what": ["chosen", "you"]})
rules = alt.Chart(marks).mark_rule(strokeDash=[3, 3]).encode(
    x="threshold:Q", color=alt.Color("what:N", scale=alt.Scale(domain=["chosen", "you"], range=[INK_SOFT, INK]),
                                     legend=alt.Legend(orient="bottom", title=None)))
cost_line = alt.Chart(curve).mark_line(strokeWidth=2.5, color=MODEL_COLOR["improved"]).encode(
    x=x, y=alt.Y("cost:Q", title="total cost ($)", axis=alt.Axis(format="$~s")),
    tooltip=[alt.Tooltip("threshold:Q", format=".3f"), alt.Tooltip("cost:Q", format="$,.0f")])
c1.markdown("**Cost of the policy**")
c1.altair_chart((cost_line + rules).properties(height=250), theme=None, width="stretch")

long = curve.melt("threshold", ["recall", "dollar_recall", "precision"], var_name="measure")
long["measure"] = long["measure"].map({"recall": "frauds caught", "dollar_recall": "fraud $ caught",
                                       "precision": "alerts that are fraud"})
lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
    x=x, y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("measure:N", scale=alt.Scale(domain=["frauds caught", "fraud $ caught", "alerts that are fraud"],
                                                range=[MODEL_COLOR["improved"], "#B07D00", "#6250D6"]),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=[alt.Tooltip("threshold:Q", format=".3f"), "measure:N", alt.Tooltip("value:Q", format=".1%")])
now = alt.Chart(pd.DataFrame({"threshold": [t]})).mark_rule(color=INK, strokeDash=[3, 3]).encode(x="threshold:Q")
c2.markdown("**What it catches**")
c2.altair_chart((lines + now).properties(height=250), theme=None, width="stretch")
st.html(f'<p class="muted">Assumed costs: a missed fraud = its amount + ${costs["chargeback"]:.0f}; every alert = '
        f'${costs["review"]:.0f} of analyst time; a false alarm adds ${costs["friction"]:.0f} of customer friction. '
        'The chosen threshold stays the same across every cost and capacity scenario tested.</p>')

st.subheader("Three tiers, never an automatic decline")
tiers = load_csv("tiers_test.csv")
if tiers is not None:
    action = {"approve": "goes through", "review": "goes through; an analyst checks it",
              "hold_verify": "held until the customer confirms (OTP / call)"}
    v = pd.DataFrame({
        "Tier": [f'{dot(TIER_COLOR[k])} {TIER_NAME[k]}' for k in tiers["tier"]],
        "What happens": tiers["tier"].map(action),
        "Transactions": [f"{n:,} ({s:.2f}%)" for n, s in zip(tiers["transactions"], tiers["share_pct"])],
        "Frauds": tiers["frauds"], "Fraud rate": tiers["fraud_rate"].map(lambda x: pct(x)),
        "Share of fraud $": tiers["share_of_fraud_amt"].map(lambda x: pct(x))})
    st.html(table(v))
    st.html(f'<p class="muted">Test period, Apr–Jun 2020. Review from {thr["review"]:.2f}, hold from '
            f'{thr["hold"]:.2f}: the hold tier needed at least 90% precision on validation.</p>')

st.subheader("Threshold comparison on the test months")
pol = load_csv("threshold_comparison_test.csv")
if pol is not None:
    v = pd.DataFrame({
        "Policy": pol["model"].str.replace("Improved", "LightGBM").str.replace("Baseline", "LogReg baseline"),
        "Threshold": pol["threshold"].map("{:.2f}".format),
        "Alerts/day": pol["alerts_per_day"].map("{:.1f}".format),
        "Precision": pol["precision"].map(pct), "Recall": pol["recall"].map(pct),
        "$-recall": pol["dollar_recall"].map(pct), "Fraud $ missed": pol["fraud_amt_missed"].map(money),
        "Total cost": pol["total_cost"].map(money)})
    st.html(table(v, highlight=int(pol.index[pol["model"] == "Improved @review (capacity)"][0])))
    st.html('<p class="muted">The default 0.5 threshold would cost about twice as much as the chosen one.</p>')
