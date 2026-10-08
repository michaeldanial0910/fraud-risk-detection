"""Builds reports/risk_threshold_memo.md from the saved results, so every number in the memo
matches the latest run.  Run: python -m src.memo  (after src.train and src.explain)"""
import json

import pandas as pd

from src.train import MODELS, REPORTS


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    """Minimal markdown table (avoids an extra dependency). fmt = {column: format string}."""
    fmt = fmt or {}
    cells = [[(fmt[c].format(v) if c in fmt and pd.notna(v) else str(v)) for c, v in row.items()]
             for _, row in df.iterrows()]
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in cells)


def main():
    meta = json.loads((MODELS / "meta.json").read_text())
    thr, costs, cap = meta["thresholds"], meta["costs"], meta["alerts_per_day_capacity"]
    pol = pd.read_csv(REPORTS / "threshold_comparison_test.csv")
    tiers = pd.read_csv(REPORTS / "tiers_test.csv")
    budget = pd.read_csv(REPORTS / "alert_budget_test.csv")
    sens = pd.read_csv(REPORTS / "cost_sensitivity_val.csv")
    unc = json.loads((REPORTS / "test_uncertainty.json").read_text())
    fair = pd.read_csv(REPORTS / "fairness_audit.csv") if (REPORTS / "fairness_audit.csv").exists() else None
    seg = (pd.read_csv(REPORTS / "error_analysis_segments.csv")
           if (REPORTS / "error_analysis_segments.csv").exists() else None)

    rev = pol[pol.model == "Improved @review (capacity)"].iloc[0]
    hold = pol[pol.model == "Improved @hold (high precision)"].iloc[0]
    base = pol[pol.model == "Baseline @review (capacity)"].iloc[0]
    flagged = tiers[tiers.tier != "approve"]
    ci = unc["improved_at_review"]
    no_model = unc["no_model_cost"]
    saving = no_model - rev.total_cost
    test = meta["periods"]["test"]

    pol_view = pol[["model", "threshold", "alerts_per_day", "precision", "recall", "dollar_recall",
                    "fraud_amt_missed", "total_cost"]]
    tier_view = tiers.assign(action=tiers.tier.map({
        "approve": "Approve automatically (no human action)",
        "review": "Analyst review queue -- transaction goes through, analyst checks within SLA",
        "hold_verify": "Hold pending customer verification (OTP / call); analyst confirms outcome"}))
    tier_view = tier_view[["tier", "action", "transactions", "share_pct", "frauds", "fraud_rate",
                           "share_of_all_frauds", "share_of_fraud_amt"]]

    miss_txt = ""
    if seg is not None and (seg.missed_fraud_amt > 0).any():
        # lowest-recall segments with enough frauds to mean something; skip segments covering ~all rows
        top = (seg[(seg.missed_frauds > 0) & (seg.frauds >= 20) & (seg.n < 0.9 * seg.n.max())]
               .sort_values("recall").head(3))
        miss_txt = ("Lowest recall on validation: "
                    + "; ".join(f"`{r.segment}` ({r.recall:.1%}, {r.missed_frauds} of {r.frauds} frauds missed)"
                                for r in top.itertuples())
                    + ". Missed frauds look more like ordinary spending: smaller, mostly daytime, less unusual for the card "
                    "(see `reports/fn_vs_tp_profile.csv`).")
    fair_txt = ""
    if fair is not None:
        f = fair.dropna(subset=["false_alarm_rate_pct"])
        g = f[f.group.str.startswith("gender=")].set_index("group").false_alarm_rate_pct
        a = f[f.group.str.startswith("age_band=")]
        hi, lo = a.loc[a.false_alarm_rate_pct.idxmax()], a.loc[a.false_alarm_rate_pct.idxmin()]
        fair_txt = (f"False-alarm rate (share of genuine transactions flagged): "
                    + ", ".join(f"`{k}` {v:.2f}%" for k, v in g.items())
                    + f"; by age from {lo.false_alarm_rate_pct:.2f}% (`{lo.group}`) to "
                    f"{hi.false_alarm_rate_pct:.2f}% (`{hi.group}`). Gender is not a model input, so a gap there "
                    f"comes through correlated behaviour (spending patterns) and should be monitored. Age is a "
                    f"model input. See `reports/fairness_audit.csv`.")

    memo = f"""# Risk Threshold Memo: Card Transaction Fraud Scoring

**Decision requested:** approve the alert thresholds below for a human-review fraud workflow.
**Scope:** scores card transactions for fraud risk. The model **never declines a transaction on its own**;
it decides which transactions a person looks at, and in what order.
**Evidence:** chosen on validation ({meta['periods']['val']['start'][:10]} to {meta['periods']['val']['end'][:10]}),
reported once on an untouched later test period ({test['start'][:10]} to {test['end'][:10]},
{test['rows']:,} transactions, {test['frauds']:,} frauds).

## 1. Recommendation

| Setting | Threshold | Meaning |
|---|---|---|
| Review threshold | **{thr['review']:.3f}** | score at or above -> analyst queue |
| Hold threshold | **{thr['hold']:.3f}** | score at or above -> hold + verify with the customer |

On the test period this means:
- **{rev.alerts_per_day:.0f} alerts per day** against a team capacity of {cap}/day
  ({flagged.share_pct.sum():.2f}% of all transactions reach a human).
- **{rev.recall:.1%} of frauds** (95% CI {ci['recall_ci95'][0]:.1%} to {ci['recall_ci95'][1]:.1%}) and
  **{rev.dollar_recall:.1%} of fraud dollars** are caught.
- **{rev.precision:.1%} of alerts are real fraud** (95% CI {ci['precision_ci95'][0]:.1%} to {ci['precision_ci95'][1]:.1%}).
- Total cost **${rev.total_cost:,.0f}** vs **${no_model:,.0f}** with no screening:
  **${saving:,.0f} saved** over {test['days']:.0f} days (~${saving / test['days'] * 30:,.0f} per 30 days).
- The simple baseline (logistic regression) at the same capacity costs **${base.total_cost:,.0f}**
  and catches {base.dollar_recall:.1%} of fraud dollars.
{'- **Capacity is the binding constraint**: with unlimited analysts the cheapest threshold would be ' + f"{thr['cost_opt']:.3f}." if thr['capacity_binding'] else '- Capacity is **not** binding: the cheapest threshold already fits within the team capacity.'}

## 2. Decision tiers (test period)

{md_table(tier_view, {'share_pct': '{:.2f}%', 'fraud_rate': '{:.1%}', 'share_of_all_frauds': '{:.1%}', 'share_of_fraud_amt': '{:.1%}', 'transactions': '{:,}'})}

The hold tier requires >= {meta['hold_min_precision']:.0%} precision on validation; on test it reached
{hold.precision:.1%}. Holding a genuine customer's payment is the costliest kind of false alarm, which is why
this tier is reserved for near-certain cases and always ends with a person (the customer or an analyst).

## 3. Threshold comparison (test)

{md_table(pol_view, {'threshold': '{:.3f}', 'alerts_per_day': '{:.1f}', 'precision': '{:.3f}', 'recall': '{:.3f}', 'dollar_recall': '{:.3f}', 'fraud_amt_missed': '${:,.0f}', 'total_cost': '${:,.0f}'})}

The default 0.5 threshold is not a business decision; it is shown only for reference.

## 4. Cost assumptions

| Item | Assumed cost | Applies to |
|---|---|---|
| Missed fraud | transaction amount + ${costs['chargeback']:.0f} chargeback handling | every false negative |
| Analyst review | ${costs['review']:.0f} | every alert |
| Customer friction | ${costs['friction']:.0f} | every false alarm (genuine customer held / called) |

These are **assumptions**, not measured values. Because a missed fraud costs its own amount, the model is rewarded
for catching large frauds first, which plain recall cannot see.

## 5. How sensitive is the choice? (validation)

{md_table(sens, {'review_threshold': '{:.3f}', 'alerts_per_day': '{:.1f}', 'recall': '{:.3f}', 'dollar_recall': '{:.3f}', 'precision': '{:.3f}'})}

## 6. What a different review budget buys (test)

{md_table(budget, {'alert_rate_pct': '{:.2f}%', 'recall': '{:.1%}', 'dollar_recall': '{:.1%}', 'precision': '{:.1%}', 'alerts_total': '{:,}'})}

## 7. Risks and guardrails

- **No automatic fraud decisions.** Every flagged transaction ends with a person: an analyst or the verified customer.
- **Analysts see the reasons.** Each alert shows its top three risk drivers (e.g. "amount is 9x this card's
  usual spend"), so reviewers can challenge the score instead of rubber-stamping it.
- **Known blind spots.** {miss_txt or 'See reports/error_analysis_segments.csv.'}
- **Fairness.** {fair_txt or 'See reports/fairness_audit.csv.'}
- **Drift.** Fraud tactics change. Monitor weekly: alert volume, precision of reviewed alerts, and score
  distribution. Re-pick thresholds monthly on the latest labelled month; retrain quarterly.
- **Label delay.** Fraud labels arrive weeks later (chargebacks), so recent performance is always partly unknown.
- **Data.** Results come from simulated data (Sparkov). The *process* transfers to real data; the *numbers* must be
  re-established on real transactions before any production use.
"""
    (REPORTS / "risk_threshold_memo.md").write_text(memo)
    print(f"wrote {REPORTS / 'risk_threshold_memo.md'}")


if __name__ == "__main__":
    main()
