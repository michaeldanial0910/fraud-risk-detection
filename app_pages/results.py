import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib.ui import (FRAUD, INK, INK_SOFT, MODEL_COLOR, figures, load_csv, load_meta, load_scored, money,
                        use_altair_theme)

use_altair_theme()
st.title("Gradient boosting vs a linear baseline")
st.html('<p class="lede">Models were compared on the validation quarter, after training on 2019. The winner was '
        'scored once on the test months (Apr–Jun 2020). At 0.6% fraud, accuracy and ROC-AUC flatter every model, '
        'so the comparison uses precision-recall and what each model catches within the review budget.</p>')

cmp, pol, meta = load_csv("model_comparison_val.csv"), load_csv("threshold_comparison_test.csv"), load_meta()
if cmp is None or pol is None:
    st.warning("Result tables are missing. Run `python -m src.train`.")
    st.stop()

p = pol.set_index("model")
imp, base = p.loc["Improved @review (capacity)"], p.loc["Baseline @review (capacity)"]
st.html(figures([
    (f"{imp.pr_auc:.3f}", "test PR-AUC, LightGBM"),
    (f"{base.pr_auc:.3f}", "test PR-AUC, logistic-regression baseline"),
    (f"{imp.recall:.1%}", f"of frauds caught (baseline {base.recall:.0%})"),
    (f"{money(imp.total_cost)}", f"total cost on test (baseline {money(base.total_cost)})"),
]))

st.subheader("Model and imbalance variants, validation PR-AUC")
d = cmp.copy()
d.loc[len(d)] = {"model": "Tuned LightGBM (final)", "val_pr_auc": meta.get("val_best_pr_auc", np.nan),
                 "val_roc_auc": np.nan}
d["family"] = np.where(d["model"].str.contains("LightGBM"), "LightGBM", "Logistic regression")
d["final"] = d["model"].str.contains("final")
order = d.sort_values("val_pr_auc", ascending=False)["model"].tolist()
bars = alt.Chart(d).mark_bar(cornerRadius=3, height=18).encode(
    x=alt.X("val_pr_auc:Q", title="PR-AUC on validation", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5)),
    y=alt.Y("model:N", sort=order, title=None),
    color=alt.Color("family:N", scale=alt.Scale(domain=["LightGBM", "Logistic regression"],
                                                range=[MODEL_COLOR["improved"], MODEL_COLOR["baseline"]]),
                    legend=alt.Legend(orient="bottom", title=None)),
    opacity=alt.condition("datum.final", alt.value(1), alt.value(0.6)),
    tooltip=["model", alt.Tooltip("val_pr_auc:Q", format=".4f", title="PR-AUC"),
             alt.Tooltip("val_roc_auc:Q", format=".4f", title="ROC-AUC")])
lbl = alt.Chart(d).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
    x="val_pr_auc:Q", y=alt.Y("model:N", sort=order), text=alt.Text("val_pr_auc:Q", format=".3f"))
st.altair_chart((bars + lbl).properties(height=260), theme=None, width="stretch")
st.html('<p class="muted">Every LightGBM variant lands within 0.01 of the others, so how imbalance is handled '
        'barely matters here; tuning added nothing beyond noise. The baseline is linear and cannot combine '
        'signals such as night + online shopping + an amount far above the card\'s usual.</p>')

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("What a review budget catches")
    scored = load_scored()
    if scored is not None:
        days = (scored["ts"].max() - scored["ts"].min()).total_seconds() / 86400
        s = scored.sort_values("score", ascending=False)
        y, a = s["is_fraud"].to_numpy(), s["amt"].to_numpy()
        k = np.unique(np.round(np.geomspace(1, len(s), 120)).astype(int))
        cur = pd.DataFrame({"alerts per day": k / days, "frauds caught": np.cumsum(y)[k - 1] / y.sum(),
                            "fraud $ caught": np.cumsum(a * y)[k - 1] / (a * y).sum()})
        cur = cur[(cur["alerts per day"] >= 1) & (cur["alerts per day"] <= 300)].melt("alerts per day", var_name="measure")
        line = alt.Chart(cur).mark_line(strokeWidth=2.5).encode(
            x=alt.X("alerts per day:Q", scale=alt.Scale(type="log"), title="alerts per day (log)"),
            y=alt.Y("value:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])),
            strokeDash=alt.StrokeDash("measure:N", legend=alt.Legend(orient="bottom", title=None)),
            color=alt.value(MODEL_COLOR["improved"]),
            tooltip=[alt.Tooltip("alerts per day:Q", format=".0f"), "measure:N", alt.Tooltip("value:Q", format=".1%")])
        cap = alt.Chart(pd.DataFrame({"x": [meta["alerts_per_day_capacity"]]})).mark_rule(
            color=INK_SOFT, strokeDash=[3, 3]).encode(x="x:Q")
        st.altair_chart((line + cap).properties(height=260), theme=None, width="stretch")
        st.html('<p class="muted">Test months, highest risk first. 20 alerts a day already reach 98% of frauds; '
                'the dashed line is the team\'s capacity.</p>')
with c2:
    st.subheader("What the model looks at")
    imp_t = load_csv("feature_importance.csv")
    if imp_t is not None:
        imp_t = imp_t.head(10)
        imp_t["kind"] = imp_t["type"].map({"engineered": "card history / time (engineered)", "original": "raw column"})
        fi = alt.Chart(imp_t).mark_bar(cornerRadius=3, height=16).encode(
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds)"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["card history / time (engineered)", "raw column"],
                                                      range=[FRAUD, "#8A94A6"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".2f")])
        st.altair_chart(fi.properties(height=260), theme=None, width="stretch")
        st.html('<p class="muted">7 of the top 9 signals are built from each card\'s own past transactions.</p>')
