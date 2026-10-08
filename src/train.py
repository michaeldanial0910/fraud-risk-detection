"""Training entry point.  Run: python -m src.train
Flow: load (cached features) -> chronological split
      -> [1] compare models x imbalance strategies (fit on train, score on val)
      -> [2] tune LightGBM (random search, val PR-AUC; imbalance weight is a hyperparameter)
      -> [3] decision policy on val: per-transaction cost curve + analyst capacity -> review/hold tiers
      -> [4] evaluate on TEST exactly once -> save model, metadata, tables, figures, scored test set."""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from src import evaluate as ev
from src.config import (ALERTS_PER_DAY, CARD_COL, COST_CHARGEBACK, COST_FRICTION, COST_REVIEW,
                        HOLD_MIN_PRECISION, N_TUNE_ITER, SEED, TIME_COL)
from src.data import PROCESSED, load_dataset, split
from src.features import FEATURES, preprocessing_steps

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
BASELINE = "Logistic Regression (baseline)"
LGB_BASE = dict(random_state=SEED, verbose=-1, n_jobs=-1, subsample_freq=1)


def save(model, name: str):
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{name}.joblib")


def load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")


def n_days(X: pd.DataFrame) -> float:
    t = X[TIME_COL]
    return max((t.max() - t.min()).total_seconds() / 86400, 1.0)


def make_model(clf, scale=False, smote=None) -> ImbPipeline:
    """impute/scale + one-hot -> (SMOTE on train only, inside fit) -> classifier."""
    steps = preprocessing_steps(scale)
    if smote:
        steps.append(("smote", SMOTE(sampling_strategy=smote, random_state=SEED)))
    return ImbPipeline(steps + [("clf", clf)])


def candidates(spw: float) -> dict:
    """Models x imbalance strategies. spw = neg/pos ratio (~170 here) for class weighting."""
    lgb = dict(**LGB_BASE, n_estimators=400, learning_rate=0.05, num_leaves=31,
               min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1)
    return {
        BASELINE: make_model(LogisticRegression(max_iter=2000), scale=True),
        "Logistic Regression + class_weight": make_model(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
        "LightGBM": make_model(LGBMClassifier(**lgb)),
        "LightGBM + sqrt class weight": make_model(LGBMClassifier(**lgb, scale_pos_weight=np.sqrt(spw))),
        "LightGBM + full class weight": make_model(LGBMClassifier(**lgb, scale_pos_weight=spw)),
        "LightGBM + SMOTE (10%)": make_model(LGBMClassifier(**lgb), smote=0.1),
    }


def val_scores(model, X_val, y_val, amt_val, days) -> dict:
    p = model.predict_proba(X_val)[:, 1]
    curve = ev.cost_curve(y_val, p, amt_val, days)
    at_cap = curve[curve.alerts_per_day <= ALERTS_PER_DAY]
    best = at_cap.loc[at_cap.cost.idxmin()] if len(at_cap) else curve.iloc[-1]
    return {"val_pr_auc": average_precision_score(y_val, p), "val_roc_auc": roc_auc_score(y_val, p),
            "val_recall_at_capacity": float(best.recall), "val_dollar_recall_at_capacity": float(best.dollar_recall),
            "val_cost_at_capacity": float(best.cost)}


def compare(models: dict, X_tr, y_tr, X_val, y_val, days) -> tuple[pd.DataFrame, dict]:
    rows, fitted = [], {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_tr, y_tr)
        fitted[name] = m
        rows.append({"model": name, **val_scores(m, X_val, y_val, X_val["amt"], days),
                     "fit_seconds": round(time.time() - t0, 1)})
        r = rows[-1]
        print(f"  {name:38s} PR-AUC {r['val_pr_auc']:.4f} | recall@cap {r['val_recall_at_capacity']:.3f} "
              f"| $recall@cap {r['val_dollar_recall_at_capacity']:.3f} | {r['fit_seconds']}s")
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True), fitted


def tune_lgbm(X_tr, y_tr, X_val, y_val, spw: float, n_iter=N_TUNE_ITER):
    """Random search scored on the (later) validation period -- no shuffled CV across time.
    Class weighting is searched as a hyperparameter: none / sqrt / full."""
    space = {
        "n_estimators": randint(200, 800),
        "learning_rate": loguniform(0.02, 0.15),
        "num_leaves": randint(15, 128),
        "min_child_samples": randint(20, 300),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(0.1, 30),
        "scale_pos_weight": [1.0, float(np.sqrt(spw)), float(spw)],
    }
    best, best_ap, log = None, -1, []
    for i, params in enumerate(ParameterSampler(space, n_iter=n_iter, random_state=SEED)):
        t0 = time.time()
        m = make_model(LGBMClassifier(**LGB_BASE, **params)).fit(X_tr, y_tr)
        ap = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        log.append({**clean_params(params), "val_pr_auc": ap, "seconds": round(time.time() - t0, 1)})
        print(f"  [{i + 1:2d}/{n_iter}] PR-AUC {ap:.4f}  ({log[-1]['seconds']}s)")
        if ap > best_ap:
            best, best_ap, best_params = m, ap, clean_params(params)
    return best, best_params, best_ap, pd.DataFrame(log).sort_values("val_pr_auc", ascending=False)


def clean_params(params: dict) -> dict:
    return {k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 4)) for k, v in params.items()}


def cost_sensitivity(y, p, amt, days) -> pd.DataFrame:
    """How does the chosen review threshold move if our cost / capacity assumptions are wrong? (val)"""
    rows = []
    for cb, fr, cap in [(20, 10, 60), (0, 10, 60), (50, 10, 60), (20, 0, 60), (20, 30, 60),
                        (20, 10, 30), (20, 10, 120), (20, 10, 10_000)]:
        c = ev.cost_curve(y, p, amt, days, chargeback=cb, friction=fr)
        t = ev.choose_thresholds(c, cap, HOLD_MIN_PRECISION)
        r = c.loc[(c.threshold - t["review"]).abs().idxmin()]
        rows.append({"chargeback": cb, "friction": fr, "capacity_per_day": cap,
                     "review_threshold": t["review"], "alerts_per_day": r.alerts_per_day,
                     "recall": r.recall, "dollar_recall": r.dollar_recall, "precision": r.precision})
    return pd.DataFrame(rows)


def policy_row(name, y, p, amt, thr, days) -> dict:
    pred = (p >= thr).astype(int)
    c = ev.txn_costs(y, pred, amt)
    total_fraud_amt = float(np.asarray(amt)[np.asarray(y) == 1].sum())
    return {**ev.summary(y, p, thr, name), "alerts_per_day": pred.sum() / days,
            "dollar_recall": 1 - c["fraud_amt_missed"] / max(total_fraud_amt, 1e-9),
            "fraud_amt_missed": c["fraud_amt_missed"], "total_cost": c["cost"]}


def main():
    for d in (MODELS, REPORTS, ev.FIG_DIR, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_dataset()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(df)
    d_val, d_te = n_days(X_val), n_days(X_te)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s")
    for name, X, y in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]:
        print(f"  {name:5s} {X[TIME_COL].min():%Y-%m-%d} .. {X[TIME_COL].max():%Y-%m-%d} | "
              f"{len(X):>9,} rows | {int(y.sum()):>5,} frauds ({y.mean():.3%}) | {n_days(X):.0f} days")

    print("\n[1/4] model x imbalance comparison (fit on train, scored on val)")
    cmp_table, fitted = compare(candidates(spw), X_tr, y_tr, X_val, y_val, d_val)
    baseline = fitted[BASELINE]

    print(f"\n[2/4] tuning LightGBM ({N_TUNE_ITER} random configs, val PR-AUC)")
    improved, best_params, best_ap, tune_log = tune_lgbm(X_tr, y_tr, X_val, y_val, spw)
    print(f"  best val PR-AUC {best_ap:.4f}  params {best_params}")

    print("\n[3/4] decision policy on validation")
    p_val = {"Baseline": baseline.predict_proba(X_val)[:, 1], "Improved": improved.predict_proba(X_val)[:, 1]}
    curve = ev.cost_curve(y_val, p_val["Improved"], X_val["amt"], d_val)
    curve_b = ev.cost_curve(y_val, p_val["Baseline"], X_val["amt"], d_val)
    thr = ev.choose_thresholds(curve, ALERTS_PER_DAY, HOLD_MIN_PRECISION)
    thr_b = ev.choose_thresholds(curve_b, ALERTS_PER_DAY, HOLD_MIN_PRECISION)
    sens = cost_sensitivity(y_val, p_val["Improved"], X_val["amt"], d_val)
    print(f"  thresholds: {thr}")
    print(sens.round(3).to_string(index=False))

    print("\n[4/4] test evaluation (all choices frozen; evaluated once per run)")
    p_te = {"Baseline": baseline.predict_proba(X_te)[:, 1], "Improved": improved.predict_proba(X_te)[:, 1]}
    amt_te = X_te["amt"].to_numpy()
    policies = pd.DataFrame([
        policy_row("Baseline @0.5", y_te, p_te["Baseline"], amt_te, 0.5, d_te),
        policy_row("Baseline @review (capacity)", y_te, p_te["Baseline"], amt_te, thr_b["review"], d_te),
        policy_row("Improved @0.5", y_te, p_te["Improved"], amt_te, 0.5, d_te),
        policy_row("Improved @cost-opt (no capacity limit)", y_te, p_te["Improved"], amt_te, thr["cost_opt"], d_te),
        policy_row("Improved @review (capacity)", y_te, p_te["Improved"], amt_te, thr["review"], d_te),
        policy_row("Improved @hold (high precision)", y_te, p_te["Improved"], amt_te, thr["hold"], d_te),
    ])
    no_model = ev.txn_costs(y_te, np.zeros(len(y_te)), amt_te)
    tiers = ev.tier_table(y_te, p_te["Improved"], amt_te, thr["review"], thr["hold"])
    budget = ev.budget_table(y_te, p_te["Improved"], amt_te, d_te)
    pred_rev = (p_te["Improved"] >= thr["review"]).astype(int)
    uncertainty = {"improved_at_review": ev.bootstrap_metrics(y_te, p_te["Improved"], thr["review"]),
                   "no_model_cost": no_model["cost"], "no_model_fraud_amt": no_model["fraud_amt_missed"]}
    print(policies.round(3).to_string(index=False))
    print("\n  tiers (test):\n", tiers.round(3).to_string(index=False))
    print(f"  cost with no model (approve all): ${no_model['cost']:,.0f}")

    # figures
    ev.plot_confusion(y_te, pred_rev, labels=["Legit", "Fraud"],
                      title=f"Test @ review threshold {thr['review']:.3f}")
    ev.plot_pr_curves(y_te, p_te)
    ev.plot_cost_curve(curve, thr, ALERTS_PER_DAY)
    ev.plot_calibration(y_te, p_te["Improved"])
    ev.plot_budget({k: ev.cost_curve(y_te, v, amt_te, d_te) for k, v in p_te.items()}, ALERTS_PER_DAY)

    # persist
    save(improved, "fraud_model")
    save(baseline, "baseline_model")
    meta = {
        "thresholds": thr, "baseline_thresholds": thr_b,
        "costs": {"chargeback": COST_CHARGEBACK, "review": COST_REVIEW, "friction": COST_FRICTION},
        "alerts_per_day_capacity": ALERTS_PER_DAY, "hold_min_precision": HOLD_MIN_PRECISION,
        "best_params": best_params, "val_best_pr_auc": float(best_ap), "seed": SEED,
        "feature_columns": FEATURES, "scale_pos_weight_full": spw,
        "periods": {n: {"start": str(X[TIME_COL].min()), "end": str(X[TIME_COL].max()), "rows": len(X),
                        "frauds": int(y.sum()), "days": round(n_days(X), 1)}
                    for n, X, y in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]},
        "test_report": ev.report(y_te, pred_rev, labels=["Legit", "Fraud"]),
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    cmp_table.to_csv(REPORTS / "model_comparison_val.csv", index=False)
    tune_log.to_csv(REPORTS / "tuning_log.csv", index=False)
    curve.to_csv(REPORTS / "cost_curve_val.csv", index=False)
    sens.to_csv(REPORTS / "cost_sensitivity_val.csv", index=False)
    policies.to_csv(REPORTS / "threshold_comparison_test.csv", index=False)
    tiers.to_csv(REPORTS / "tiers_test.csv", index=False)
    budget.to_csv(REPORTS / "alert_budget_test.csv", index=False)
    (REPORTS / "test_uncertainty.json").write_text(json.dumps(uncertainty, indent=2))

    # scored test set for the Streamlit analyst queue (card number masked)
    scored = X_te[[TIME_COL, CARD_COL, "gender"] + FEATURES].copy()
    scored[CARD_COL] = "card ..." + scored[CARD_COL].astype(str).str[-4:]
    scored["score"], scored["is_fraud"] = p_te["Improved"], y_te.to_numpy()
    scored["tier"] = ev.assign_tier(scored["score"], thr["review"], thr["hold"])
    scored.to_parquet(PROCESSED / "test_scored.parquet", index=False)
    print(f"\nsaved model, metadata, tables and figures. total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
