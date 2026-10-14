import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (FRAUD, INK, INK_SOFT, LEGIT, dot, figures, load_csv, load_json, money, table,
                        use_altair_theme)

use_altair_theme()
s = load_json("eda_summary.json")
st.title("The data: 1.3 million card transactions")
st.html('<p class="lede">Simulated card transactions (Sparkov, Kaggle), Jan 2019 to Jun 2020: 983 cardholders, '
        'one row per payment. It is synthetic, so fraud here is easier to separate than in real life; the method '
        'is the point, not the exact numbers.</p>')
if not s:
    st.warning("EDA tables are missing. Run `python -m src.eda`.")
    st.stop()

st.html(figures([
    (f"{s['transactions'] / 1e6:.2f}M", "transactions"),
    (f"{s['fraud_rate']:.2%}", f"are fraud (1 in {1 / s['fraud_rate']:.0f})"),
    (f"{s['fraud_share_of_dollars']:.1%}", "of all dollars: frauds are big"),
    (f"{s['cards_defrauded'] / s['cards']:.0%}", "of cards hit at least once"),
    (f"{s['night_share_of_frauds']:.0%}", f"of frauds at night (legit: {s['night_share_of_legit']:.0%})"),
]))

st.subheader("Fraud comes in bursts, and the rate drifts")
m = load_csv("eda_monthly.csv")
m["month"] = pd.to_datetime(m["month"])
line = alt.Chart(m).mark_line(point=alt.OverlayMarkDef(color=FRAUD, size=40), strokeWidth=2.5, color=FRAUD).encode(
    x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y")),
    y=alt.Y("fraud_rate:Q", axis=alt.Axis(format=".1%"), title="fraud rate"),
    tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("fraud_rate:Q", format=".2%"),
             alt.Tooltip("frauds:Q", format=","), alt.Tooltip("transactions:Q", format=",")])
cuts = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-04-01"])})
rules = alt.Chart(cuts).mark_rule(color=INK_SOFT, strokeDash=[4, 4]).encode(x="d:T")
lab = pd.DataFrame({"d": pd.to_datetime(["2019-06-15", "2020-02-10", "2020-05-08"]),
                    "t": ["train (2019)", "validation", "test"]})
text = alt.Chart(lab).mark_text(y=8, fontSize=12, color=INK).encode(x="d:T", text="t:N")
st.altair_chart((line + rules + text).properties(height=240), theme=None, width="stretch")
st.html(f'<p class="muted">The monthly rate swings between {m.fraud_rate.min():.2%} and {m.fraud_rate.max():.2%}. '
        f'A typical defrauded card is hit about {s["median_frauds_per_card"]:.0f} times within '
        f'{s["median_burst_days"]:.1f} days, so a random split would put the same burst in train and test. '
        'The split is by time.</p>')

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Fraud rate by hour")
    h = load_csv("eda_hourly.csv")
    h["night"] = (h["hour"] >= 22) | (h["hour"] < 4)
    hb = alt.Chart(h).mark_bar(cornerRadius=2).encode(
        x=alt.X("hour:O", title="hour of day", axis=alt.Axis(labelAngle=0, values=list(range(0, 24, 3)))),
        y=alt.Y("fraud_rate:Q", axis=alt.Axis(format=".1%", tickCount=4), title=None),
        color=alt.condition("datum.night", alt.value(FRAUD), alt.value("#C9CED8")),
        tooltip=["hour", alt.Tooltip("fraud_rate:Q", format=".2%"), alt.Tooltip("frauds:Q", format=",")])
    st.altair_chart(hb.properties(height=230), theme=None, width="stretch")
with c2:
    st.subheader("Fraud rate by merchant category")
    c = load_csv("eda_category.csv")
    cb = alt.Chart(c).mark_bar(cornerRadius=2, height=11).encode(
        x=alt.X("fraud_rate:Q", axis=alt.Axis(format=".1%", tickCount=4), title=None),
        y=alt.Y("category:N", sort="-x", title=None),
        color=alt.condition("datum.fraud_rate > 0.01", alt.value(FRAUD), alt.value("#C9CED8")),
        tooltip=["category", alt.Tooltip("fraud_rate:Q", format=".2%"),
                 alt.Tooltip("fraud_amt:Q", format="$,.0f", title="fraud $")])
    st.altair_chart(cb.properties(height=230), theme=None, width="stretch")

st.subheader("Fraud changes how a card behaves")
b = load_csv("eda_card_behaviour.csv").set_index("feature")
names = {"amt": ("Amount", money), "amt_vs_card_avg": ("Times the card's usual amount", "{:.2f}x".format),
         "amt_sum_24h": ("Card's spend in the previous 24h", money),
         "n_txn_24h": ("Card's transactions in the previous 24h", "{:.0f}".format),
         "hours_since_prev": ("Hours since the card's previous transaction", "{:.1f} h".format)}
v = pd.DataFrame({"Median": [names[k][0] for k in names],
                  f"{dot(LEGIT)} genuine": [names[k][1](b.loc[k, "legit_median"]) for k in names],
                  f"{dot(FRAUD)} fraud": [names[k][1](b.loc[k, "fraud_median"]) for k in names]})
st.html(table(v))
st.html('<p class="muted">These card-history features use only each card\'s earlier transactions, exactly what a '
        'payment system knows at the moment of the payment. A unit test checks that a later transaction never '
        'changes an earlier one\'s features.</p>')
