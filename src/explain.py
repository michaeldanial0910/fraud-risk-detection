"""Explainability, error analysis and fairness audit.  Run: python -m src.explain  (after src.train)
- Global : SHAP beeswarm / bar on a validation sample (all frauds + random legit rows)
- Local  : reason_codes() -> plain-English "why was this flagged" for the analyst queue
- Errors : which kinds of fraud do we miss at the review threshold? (validation, test stays untouched)
- Audit  : alert / false-alarm rates by gender and age band (gender is NOT a model input)"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import evaluate as ev
from src.config import SEED, SHAP_SAMPLE, TARGET
from src.data import load_dataset, split
from src.features import ENGINEERED
from src.train import MODELS, REPORTS, load

warnings.filterwarnings("ignore")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def shap_values(model, X: pd.DataFrame):
    """Run X through the fitted preprocessing, then explain the final tree model (log-odds)."""
    Z = model[:-1].transform(X)
    return Z, shap.TreeExplainer(model.named_steps["clf"])(Z)


def top_drivers(model, X_row: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Signed log-odds contribution of the k most influential features for one row."""
    Z, sv = shap_values(model, X_row)
    s = pd.Series(sv.values[0], index=Z.columns)
    top = s.reindex(s.abs().sort_values(ascending=False).index)[:k]
    return pd.DataFrame({"feature": top.index, "shap": top.values})


def _phrase(feature: str, row: pd.Series) -> str:
    """Turn a model feature + the transaction's raw values into an analyst-readable sentence."""
    if feature.startswith("category_"):
        return f"Merchant category: {feature.removeprefix('category_')}"
    v = row.get(feature)
    return {
        "amt": lambda: f"Amount ${v:,.2f}",
        "amt_vs_card_avg": lambda: f"Amount is {v:.1f}x this card's usual spend",
        "n_txn_24h": lambda: f"{v:.0f} other transactions on this card in the previous 24h",
        "amt_sum_24h": lambda: f"${v:,.0f} spent on this card in the previous 24h",
        "hours_since_prev": lambda: f"{v:.1f}h since this card's previous transaction",
        "hour": lambda: f"Made at {int(v):02d}:00",
        "is_night": lambda: f"Night-time transaction ({int(row['hour']):02d}:00)",
        "day_of_week": lambda: f"Day: {DAYS[int(v)]}",
        "new_category_for_card": lambda: "First time this card is used in this category",
        "card_prior_txns": lambda: f"Card has {v:.0f} earlier transactions on record",
        "age": lambda: f"Cardholder age {v:.0f}",
        "city_pop": lambda: f"Cardholder city population {v:,.0f}",
    }.get(feature, lambda: f"{feature} = {v}")()


def reason_codes(model, X_rows: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """Top-k features pushing EACH row's risk UP, as sentences. Only positive SHAP contributions:
    the analyst needs to know what looks suspicious, not what looks normal."""
    Z, sv = shap_values(model, X_rows)
    out = []
    for i in range(len(X_rows)):
        s = pd.Series(sv.values[i], index=Z.columns)
        s = s[s > 0].sort_values(ascending=False)[:k]
        # is_night and hour describe the same fact -- keep only the stronger one
        if {"is_night", "hour"} <= set(s.index):
            s = s.drop("hour" if s["is_night"] >= s["hour"] else "is_night")
        out.append([_phrase(f, X_rows.iloc[i]) for f in s.index])
    return out


def shap_sample(X: pd.DataFrame, y: pd.Series, n=SHAP_SAMPLE, seed=SEED) -> pd.DataFrame:
    """All frauds + random legit rows (a uniform sample would contain ~25 frauds)."""
    fraud = X[y == 1]
    legit = X[y == 0].sample(max(n - len(fraud), 0), random_state=seed)
    return pd.concat([fraud, legit])


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    Z, sv = shap_values(model, X)
    for kind, fn in [("summary", shap.plots.beeswarm), ("bar", shap.plots.bar)]:
        plt.figure(); fn(sv, max_display=15, show=False)
        plt.title(f"SHAP {kind} (validation sample: all frauds + random legit)"); plt.tight_layout()
        plt.savefig(ev.FIG_DIR / f"shap_{kind}.png", dpi=150, bbox_inches="tight"); plt.close()
    vals = pd.DataFrame(np.abs(sv.values), columns=Z.columns)
    # merchant-category one-hots summed back into one "category" feature (SHAP is additive)
    signed = pd.DataFrame(sv.values, columns=Z.columns)
    cat_cols = [c for c in Z.columns if c.startswith("category_")]
    grouped = vals.drop(columns=cat_cols).mean()
    grouped["category"] = signed[cat_cols].sum(axis=1).abs().mean()
    imp = (grouped.sort_values(ascending=False).rename("mean_abs_shap").rename_axis("feature")
           .reset_index())
    imp["type"] = np.where(imp["feature"].isin(ENGINEERED), "engineered", "original")
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    return imp


def _bands(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["amount_band"] = pd.cut(d["amt"], [0, 20, 100, 300, 1000, np.inf],
                              labels=["<$20", "$20-100", "$100-300", "$300-1k", ">$1k"])
    d["age_band"] = pd.cut(d["age"], [0, 30, 45, 60, 75, 200], labels=["<30", "30-44", "45-59", "60-74", "75+"])
    d["time_of_day"] = np.where(d["is_night"] == 1, "night (22-04)", "day")
    d["card_history"] = pd.cut(d["card_prior_txns"], [-1, 10, 100, np.inf], labels=["<=10 txns", "11-100", ">100"])
    return d


def error_analysis(df: pd.DataFrame, prob: np.ndarray, thr: float):
    """Outcome per row at the review threshold, summarised by segment, sorted by missed fraud $."""
    d = _bands(df)
    d["prob"], d["pred"] = prob, (prob >= thr).astype(int)
    d["outcome"] = np.select([(d.pred == 1) & (d[TARGET] == 1), (d.pred == 0) & (d[TARGET] == 1),
                              (d.pred == 1) & (d[TARGET] == 0)], ["TP", "FN", "FP"], "TN")
    rows = []
    for col in ["category", "amount_band", "time_of_day", "card_history"]:
        for val, g in d.groupby(col, observed=True):
            frauds = (g[TARGET] == 1).sum()
            rows.append({"segment": f"{col}={val}", "n": len(g), "frauds": int(frauds),
                         "recall": (g.outcome == "TP").sum() / frauds if frauds else np.nan,
                         "missed_frauds": int((g.outcome == "FN").sum()),
                         "missed_fraud_amt": float(g.loc[g.outcome == "FN", "amt"].sum()),
                         "false_alarms": int((g.outcome == "FP").sum()),
                         "alert_rate_pct": g.pred.mean() * 100})
    seg = pd.DataFrame(rows).sort_values("missed_fraud_amt", ascending=False).reset_index(drop=True)
    return seg, d


def fn_vs_tp_profile(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["amt", "amt_vs_card_avg", "n_txn_24h", "amt_sum_24h", "hours_since_prev", "is_night", "prob"]
    return pd.DataFrame({"missed (FN)": d.loc[d.outcome == "FN", cols].mean(),
                         "caught (TP)": d.loc[d.outcome == "TP", cols].mean()})


def fairness_audit(d: pd.DataFrame) -> pd.DataFrame:
    """Do some customer groups get more false alarms (= more friction) than others?"""
    rows = []
    for col in ["gender", "age_band"]:
        for val, g in d.groupby(col, observed=True):
            legit, frauds = (g[TARGET] == 0).sum(), (g[TARGET] == 1).sum()
            rows.append({"group": f"{col}={val}", "n": len(g), "fraud_rate": g[TARGET].mean(),
                         "alert_rate_pct": g.pred.mean() * 100,
                         "false_alarm_rate_pct": (g.outcome == "FP").sum() / max(legit, 1) * 100,
                         "recall": (g.outcome == "TP").sum() / frauds if frauds else np.nan})
    return pd.DataFrame(rows)


def main():
    ev.FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = load("fraud_model")
    thr = json.loads((MODELS / "meta.json").read_text())["thresholds"]
    _, X_val, _, _, y_val, _ = split(load_dataset())
    val = X_val.assign(**{TARGET: y_val.to_numpy()})

    imp = global_importance(model, shap_sample(X_val, y_val))
    print("Feature importance (mean |SHAP|, category one-hots grouped):\n",
          imp.head(12).round(3).to_string(index=False))

    seg, d = error_analysis(val, model.predict_proba(X_val)[:, 1], thr["review"])
    seg.to_csv(REPORTS / "error_analysis_segments.csv", index=False)
    print(f"\nSegments with most missed fraud $ (validation, review thr={thr['review']:.3f}):")
    print(seg.head(10).round(3).to_string(index=False))

    prof = fn_vs_tp_profile(d)
    prof.to_csv(REPORTS / "fn_vs_tp_profile.csv")
    print("\nMissed vs caught frauds (means):\n", prof.round(2).to_string())

    fair = fairness_audit(d)
    fair.to_csv(REPORTS / "fairness_audit.csv", index=False)
    print("\nFairness audit (validation):\n", fair.round(3).to_string(index=False))

    # example reason codes for the 5 highest-scored frauds and the 5 highest-scored false alarms
    ex = pd.concat([d[d.outcome == "TP"].nlargest(5, "prob"), d[d.outcome == "FP"].nlargest(5, "prob")])
    if len(ex):
        ex = ex.assign(reasons=[" | ".join(r) for r in reason_codes(model, ex)])
        ex[["ts", "category", "amt", "prob", "outcome", "reasons"]].to_csv(
            REPORTS / "reason_code_examples.csv", index=False)
        print("\nReason-code examples:\n", ex[["outcome", "amt", "prob", "reasons"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = seg[seg.missed_fraud_amt > 0].head(8).iloc[::-1]
    ax.barh(top["segment"], top["missed_fraud_amt"])
    ax.set_xlabel("Missed fraud amount ($)"); ax.set_title("Where the model misses fraud (validation)")
    fig.tight_layout(); fig.savefig(ev.FIG_DIR / "error_segments.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
