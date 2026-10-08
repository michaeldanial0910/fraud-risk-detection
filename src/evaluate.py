"""Evaluation helpers.
Part 1 is reused from the churn project (generic binary-classifier metrics + plots).
Part 2 is fraud-specific: per-transaction (amount-dependent) cost, alert capacity and decision tiers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, classification_report,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from src.config import COST_CHARGEBACK, COST_FRICTION, COST_REVIEW

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Fine grid at the low end: with ~0.5% fraud, useful thresholds are often well below 0.1.
THRESHOLDS = np.unique(np.round(np.concatenate([np.linspace(0.002, 0.01, 5),
                                                np.linspace(0.01, 0.99, 99)]), 4))
TIERS = ["approve", "review", "hold_verify"]


# ============================================================ part 1: generic (from churn project)
def summary(y_true, y_prob, threshold=0.5, name="model") -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name, "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def report(y_true, y_pred, labels=None) -> str:
    return classification_report(y_true, y_pred, target_names=labels, digits=3)


def bootstrap_metrics(y_true, y_prob, threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    """95% bootstrap intervals for precision and recall at a fixed threshold (descriptive uncertainty)."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y_true), np.asarray(y_prob) >= threshold
    rec, prec = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        tp = (pred[i] & (y[i] == 1)).sum()
        rec.append(tp / max((y[i] == 1).sum(), 1)); prec.append(tp / max(pred[i].sum(), 1))
    return {"recall_ci95": [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))],
            "precision_ci95": [float(np.percentile(prec, 2.5)), float(np.percentile(prec, 97.5))]}


def _save(fig, fname):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels=None, fname="confusion_matrix.png", title=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax,
                                            colorbar=False, values_format=",d")
    if title:
        ax.set_title(title)
    _save(fig, fname)


def plot_pr_curves(y_true, probs: dict, fname="pr_curve.png"):
    """Overlay PR curves for several models (probs = {name: y_prob})."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), ls="--", c="grey", label=f"No skill ({np.mean(y_true):.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall (test)")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, fname)


def plot_calibration(y_true, y_prob, fname="calibration.png", n_bins=10):
    frac, mean = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", c="grey"); ax.plot(mean, frac, "o-")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Predicted score"); ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration (test, quantile bins)")
    _save(fig, fname)


# ============================================================ part 2: fraud-specific
def txn_costs(y, pred, amt, chargeback=COST_CHARGEBACK, review=COST_REVIEW, friction=COST_FRICTION) -> dict:
    """Cost of a set of hard decisions where a missed fraud costs ITS OWN amount (+ chargeback fee).
    pred = 1 means "sent to a human" (review or hold), never an automatic decline."""
    y, pred, amt = np.asarray(y).astype(bool), np.asarray(pred).astype(bool), np.asarray(amt, float)
    fn, fp = y & ~pred, ~y & pred
    missed_amt = float(amt[fn].sum())
    return {"cost": missed_amt + fn.sum() * chargeback + pred.sum() * review + fp.sum() * friction,
            "fraud_amt_missed": missed_amt, "n_alerts": int(pred.sum()),
            "tp": int((y & pred).sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}


def cost_curve(y_true, y_prob, amt, n_days: float, thresholds=THRESHOLDS,
               chargeback=COST_CHARGEBACK, review=COST_REVIEW, friction=COST_FRICTION) -> pd.DataFrame:
    """For every threshold -> alerts/day, precision, recall, $-recall and total cost.
    Sort once by score, then cumulative sums: O(n log n), no n x thresholds matrix."""
    y, p, a = np.asarray(y_true).astype(int), np.asarray(y_prob, float), np.asarray(amt, float)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    cum_tp = np.concatenate([[0], np.cumsum(y[order])])
    cum_caught_amt = np.concatenate([[0], np.cumsum((a * y)[order])])
    thresholds = np.asarray(thresholds, float)
    alerts = np.searchsorted(-ps, -thresholds, side="right")       # how many rows have p >= t
    tp = cum_tp[alerts]
    fp, fn = alerts - tp, y.sum() - tp
    total_fraud_amt = (a * y).sum()
    missed = total_fraud_amt - cum_caught_amt[alerts]
    return pd.DataFrame({
        "threshold": thresholds,
        "alerts_per_day": alerts / n_days,
        "alert_rate_pct": alerts / len(y) * 100,
        "precision": np.divide(tp, alerts, out=np.zeros(len(alerts)), where=alerts > 0),
        "recall": tp / max(y.sum(), 1),
        "dollar_recall": 1 - missed / max(total_fraud_amt, 1e-9),
        "cost": missed + fn * chargeback + alerts * review + fp * friction,
    })


def choose_thresholds(curve: pd.DataFrame, max_alerts_per_day: float, hold_min_precision: float) -> dict:
    """Decision policy, picked on VALIDATION:
    - cost_opt : unconstrained cheapest threshold (what we'd do with unlimited analysts)
    - review   : cheapest threshold whose alert volume fits the team's daily capacity
    - hold     : lowest threshold (>= review) with precision >= hold_min_precision; these alerts are
                 held pending customer verification (OTP / call) -- still a human-in-the-loop step"""
    cost_opt = curve.loc[curve["cost"].idxmin()]
    feas = curve[curve["alerts_per_day"] <= max_alerts_per_day]
    review = feas.loc[feas["cost"].idxmin()] if len(feas) else curve.iloc[-1]
    hold_c = curve[(curve["precision"] >= hold_min_precision) & (curve["threshold"] >= review["threshold"])]
    hold_thr = float(hold_c["threshold"].min()) if len(hold_c) else float(curve["threshold"].max())
    return {"cost_opt": float(cost_opt["threshold"]), "review": float(review["threshold"]),
            "hold": max(hold_thr, float(review["threshold"])),
            "capacity_binding": bool(cost_opt["alerts_per_day"] > max_alerts_per_day)}


def assign_tier(y_prob, review_thr: float, hold_thr: float) -> np.ndarray:
    p = np.asarray(y_prob)
    return np.where(p >= hold_thr, "hold_verify", np.where(p >= review_thr, "review", "approve"))


def tier_table(y, y_prob, amt, review_thr, hold_thr) -> pd.DataFrame:
    d = pd.DataFrame({"tier": assign_tier(y_prob, review_thr, hold_thr), "y": np.asarray(y),
                      "amt": np.asarray(amt, float)})
    rows = []
    for t in TIERS:
        g = d[d.tier == t]
        rows.append({"tier": t, "transactions": len(g), "share_pct": len(g) / len(d) * 100,
                     "frauds": int(g.y.sum()), "fraud_rate": g.y.mean() if len(g) else np.nan,
                     "share_of_all_frauds": g.y.sum() / max(d.y.sum(), 1),
                     "fraud_amt": float(g.amt[g.y == 1].sum()),
                     "share_of_fraud_amt": g.amt[g.y == 1].sum() / max(d.amt[d.y == 1].sum(), 1e-9)})
    return pd.DataFrame(rows)


def budget_table(y_true, y_prob, amt, n_days: float, alerts_per_day=(10, 20, 40, 60, 100, 200)) -> pd.DataFrame:
    """'If the team reviews the top-K highest-risk transactions per day, what do we catch?'
    (ranked over the whole period: K/day x days alerts in total)."""
    y, a = np.asarray(y_true), np.asarray(amt, float)
    order = np.argsort(-np.asarray(y_prob), kind="stable")
    ys, as_ = y[order], a[order]
    rows = []
    for k in alerts_per_day:
        n = min(int(round(k * n_days)), len(y))
        rows.append({"alerts_per_day": k, "alerts_total": n, "alert_rate_pct": n / len(y) * 100,
                     "frauds_caught": int(ys[:n].sum()), "recall": ys[:n].sum() / max(y.sum(), 1),
                     "dollar_recall": (as_[:n] * ys[:n]).sum() / max((a * y).sum(), 1e-9),
                     "precision": ys[:n].mean() if n else np.nan})
    return pd.DataFrame(rows)


def plot_cost_curve(curve: pd.DataFrame, thr: dict, capacity: float, fname="cost_curve.png",
                    title="Cost and alert volume vs threshold (validation)"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve["threshold"], curve["cost"] / 1e3, c="C0", label="total cost ($k)")
    ax.set_xscale("log"); ax.set_xlabel("Decision threshold (log scale)"); ax.set_ylabel("Total cost ($k)")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], curve["alerts_per_day"], c="C1", alpha=.7, label="alerts / day")
    ax2.axhline(capacity, c="C1", ls=":", label=f"capacity = {capacity}/day")
    ax2.set_ylabel("Alerts per day"); ax2.set_yscale("log")
    for k, ls in [("cost_opt", "--"), ("review", "-"), ("hold", "-.")]:
        ax.axvline(thr[k], c="k", ls=ls, lw=1, label=f"{k} = {thr[k]:.3f}")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left"); ax.set_title(title)
    _save(fig, fname)


def plot_budget(curves: dict, capacity: float, fname="alert_budget.png"):
    """Recall and $-recall vs alerts/day for several models (curves = {name: cost_curve df})."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["alerts_per_day"], c["recall"], c=f"C{i}", label=f"{name}: recall")
        ax.plot(c["alerts_per_day"], c["dollar_recall"], c=f"C{i}", ls="--", label=f"{name}: $ recall")
    ax.axvline(capacity, c="grey", ls=":", label=f"capacity {capacity}/day")
    ax.set_xscale("log"); ax.set_xlabel("Alerts per day (log)"); ax.set_ylabel("Share of fraud caught")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=7); ax.set_title("What a given review budget catches (test)")
    _save(fig, fname)
