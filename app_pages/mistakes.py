import altair as alt
import pandas as pd
import streamlit as st

from app_lib.ui import (FRAUD, INK, LEGIT, dot, esc, figures, load_csv, load_meta, money, pct, table,
                        use_altair_theme)

use_altair_theme()
st.title("What it misses, and who it bothers")
st.html('<p class="lede">Errors at the chosen review threshold on the validation quarter (Jan–Mar 2020), so the '
        'test months stay untouched. A missed fraud costs money; a false alarm costs a genuine customer\'s time.</p>')

seg, prof, fair = load_csv("error_analysis_segments.csv"), load_csv("fn_vs_tp_profile.csv"), load_csv("fairness_audit.csv")
if seg is None or prof is None:
    st.warning("Error-analysis tables are missing. Run `python -m src.explain`.")
    st.stop()

tod = seg[seg["segment"].str.startswith("time_of_day=")]
missed, frauds, fa = int(tod["missed_frauds"].sum()), int(tod["frauds"].sum()), int(tod["false_alarms"].sum())
st.html(figures([
    (f"{missed} of {frauds:,}", "frauds missed"),
    (money(tod["missed_fraud_amt"].sum()), "fraud dollars missed in the quarter"),
    (f"{fa}", "false alarms over 91 days"),
    (f"{tod.loc[tod.segment.str.contains('day'), 'recall'].iloc[0]:.0%}",
     f"recall on daytime fraud ({tod.loc[tod.segment.str.contains('night'), 'recall'].iloc[0]:.1%} at night)"),
]))

c1, c2 = st.columns([1.2, 1], gap="large")
with c1:
    st.subheader("Recall by segment")
    s = seg[(seg["frauds"] >= 20) & (seg["n"] < 0.9 * seg["n"].max())].copy()
    s["segment"] = s["segment"].str.replace("category=", "").str.replace("amount_band=", "amount ") \
                               .str.replace("time_of_day=day", "daytime").str.replace("time_of_day=", "").str.replace("card_history=<=10 txns", "new card (≤10 txns)")
    s["weak"] = s["recall"] < 0.95
    bars = alt.Chart(s).mark_bar(cornerRadius=3, height=13, clip=True).encode(
        x=alt.X("recall:Q", scale=alt.Scale(domain=[0.8, 1]), axis=alt.Axis(format="%", tickCount=5),
                title="frauds caught"), x2=alt.datum(0.8),
        y=alt.Y("segment:N", sort="x", title=None),
        color=alt.condition("datum.weak", alt.value(FRAUD), alt.value("#C9CED8")),
        tooltip=["segment", alt.Tooltip("recall:Q", format=".1%"), "frauds", "missed_frauds"])
    st.altair_chart(bars.properties(height=330), theme=None, width="stretch")
with c2:
    st.subheader("Missed frauds look ordinary")
    p = prof.set_index(prof.columns[0])
    rows = [("Average amount", "amt", money), ("Times the card's usual spend", "amt_vs_card_avg", "{:.1f}x".format),
            ("Card's spend in the previous 24h", "amt_sum_24h", money),
            ("Share at night (22:00–04:00)", "is_night", lambda v: f"{v:.0%}")]
    v = pd.DataFrame({"": [r[0] for r in rows],
                      f"{dot(FRAUD)} missed": [r[2](p.loc[r[1], "missed (FN)"]) for r in rows],
                      f"{dot(LEGIT)} caught": [r[2](p.loc[r[1], "caught (TP)"]) for r in rows]})
    st.html(table(v))
    st.html('<div class="callout">The model catches fraud that <b>changes how a card behaves</b>: big, sudden, at '
            'night. It misses small daytime fraud that looks like the cardholder\'s normal spending.</div>')

st.subheader("Two alerts, read the way an analyst would")
ex = load_csv("reason_code_examples.csv")
if ex is not None:
    kind = st.segmented_control("Example", ["Caught fraud", "False alarm"], default="Caught fraud", key="ex_kind")
    sub = ex[ex["outcome"] == ("TP" if kind == "Caught fraud" else "FP")].head(2)
    for r in sub.itertuples():
        reasons = "".join(f'<li>{esc(x.strip())}</li>' for x in r.reasons.split("|"))
        st.html(f'<div class="slip"><b>${r.amt:,.2f}</b> · {esc(r.category)} · {pd.Timestamp(r.ts):%b %d, %H:%M}'
                f'<ul class="reasons" style="margin-top:8px">{reasons}</ul>'
                f'<div class="meta">risk score {r.prob:.3f} · in hindsight '
                f'<b>{"fraud" if r.outcome == "TP" else "genuine"}</b></div></div>')
    st.html('<p class="muted">' + ("Caught frauds: big, sudden spending on a card, usually at night."
            if kind == "Caught fraud" else "False alarms are mostly genuine big purchases. That is why they go to "
            "a person (or a quick customer check) instead of being declined.") + '</p>')

if fair is not None:
    st.subheader("Who gets the false alarms")
    f = fair.copy()
    v = pd.DataFrame({"Group": f["group"].str.replace("gender=", "gender ").str.replace("age_band=", "age "),
                      "Transactions": f["n"].map("{:,}".format),
                      "False-alarm rate": f["false_alarm_rate_pct"].map(lambda x: f"{x:.2f}%"),
                      "Frauds caught": f["recall"].map(pct)})
    st.html(table(v))
    st.html('<p class="muted">Gender is not a model input, yet women see more false alarms (0.22% vs 0.13% of '
            'genuine transactions), through correlated spending patterns. It is flagged for monitoring.</p>')
