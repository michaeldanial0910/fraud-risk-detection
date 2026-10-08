# Risk Threshold Memo: Card Transaction Fraud Scoring

**Decision requested:** approve the alert thresholds below for a human-review fraud workflow.
**Scope:** scores card transactions for fraud risk. The model **never declines a transaction on its own**;
it decides which transactions a person looks at, and in what order.
**Evidence:** chosen on validation (2020-01-01 to 2020-03-31),
reported once on an untouched later test period (2020-04-01 to 2020-06-21,
198,982 transactions, 1,163 frauds).

## 1. Recommendation

| Setting | Threshold | Meaning |
|---|---|---|
| Review threshold | **0.020** | score at or above -> analyst queue |
| Hold threshold | **0.110** | score at or above -> hold + verify with the customer |

On the test period this means:
- **18 alerts per day** against a team capacity of 60/day
  (0.75% of all transactions reach a human).
- **97.8% of frauds** (95% CI 96.8% to 98.6%) and
  **99.6% of fraud dollars** are caught.
- **76.7% of alerts are real fraud** (95% CI 74.7% to 78.8%).
- Total cost **$14,065** vs **$642,890** with no screening:
  **$628,825 saved** over 82 days (~$231,469 per 30 days).
- The simple baseline (logistic regression) at the same capacity costs **$152,594**
  and catches 79.8% of fraud dollars.
- Capacity is **not** binding: the cheapest threshold already fits within the team capacity.

## 2. Decision tiers (test period)

| tier | action | transactions | share_pct | frauds | fraud_rate | share_of_all_frauds | share_of_fraud_amt |
|---|---|---|---|---|---|---|---|
| approve | Approve automatically (no human action) | 197,499 | 99.25% | 26 | 0.0% | 2.2% | 0.4% |
| review | Analyst review queue -- transaction goes through, analyst checks within SLA | 263 | 0.13% | 18 | 6.8% | 1.5% | 0.3% |
| hold_verify | Hold pending customer verification (OTP / call); analyst confirms outcome | 1,220 | 0.61% | 1119 | 91.7% | 96.2% | 99.3% |

The hold tier requires >= 90% precision on validation; on test it reached
91.7%. Holding a genuine customer's payment is the costliest kind of false alarm, which is why
this tier is reserved for near-certain cases and always ends with a person (the customer or an analyst).

## 3. Threshold comparison (test)

| model | threshold | alerts_per_day | precision | recall | dollar_recall | fraud_amt_missed | total_cost |
|---|---|---|---|---|---|---|---|
| Baseline @0.5 | 0.500 | 4.5 | 0.802 | 0.254 | 0.384 | $381,425 | $401,355 |
| Baseline @review (capacity) | 0.020 | 23.5 | 0.423 | 0.696 | 0.798 | $124,894 | $152,594 |
| Improved @0.5 | 0.500 | 13.7 | 0.968 | 0.930 | 0.967 | $20,631 | $28,201 |
| Improved @cost-opt (no capacity limit) | 0.020 | 18.2 | 0.767 | 0.978 | 0.996 | $2,670 | $14,065 |
| Improved @review (capacity) | 0.020 | 18.2 | 0.767 | 0.978 | 0.996 | $2,670 | $14,065 |
| Improved @hold (high precision) | 0.110 | 15.0 | 0.917 | 0.962 | 0.993 | $4,329 | $12,319 |

The default 0.5 threshold is not a business decision; it is shown only for reference.

## 4. Cost assumptions

| Item | Assumed cost | Applies to |
|---|---|---|
| Missed fraud | transaction amount + $20 chargeback handling | every false negative |
| Analyst review | $5 | every alert |
| Customer friction | $10 | every false alarm (genuine customer held / called) |

These are **assumptions**, not measured values. Because a missed fraud costs its own amount, the model is rewarded
for catching large frauds first, which plain recall cannot see.

## 5. How sensitive is the choice? (validation)

| chargeback | friction | capacity_per_day | review_threshold | alerts_per_day | recall | dollar_recall | precision |
|---|---|---|---|---|---|---|---|
| 20.0 | 10.0 | 60.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 0.0 | 10.0 | 60.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 50.0 | 10.0 | 60.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 20.0 | 0.0 | 60.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 20.0 | 30.0 | 60.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 20.0 | 10.0 | 30.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 20.0 | 10.0 | 120.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |
| 20.0 | 10.0 | 10000.0 | 0.020 | 15.6 | 0.988 | 0.996 | 0.782 |

## 6. What a different review budget buys (test)

| alerts_per_day | alerts_total | alert_rate_pct | frauds_caught | recall | dollar_recall | precision |
|---|---|---|---|---|---|---|
| 10.0 | 815.0 | 0.41% | 814.0 | 70.0% | 77.4% | 99.9% |
| 20.0 | 1,630.0 | 0.82% | 1141.0 | 98.1% | 99.6% | 70.0% |
| 40.0 | 3,260.0 | 1.64% | 1155.0 | 99.3% | 99.7% | 35.4% |
| 60.0 | 4,890.0 | 2.46% | 1158.0 | 99.6% | 99.9% | 23.7% |
| 100.0 | 8,151.0 | 4.10% | 1161.0 | 99.8% | 99.9% | 14.2% |
| 200.0 | 16,301.0 | 8.19% | 1162.0 | 99.9% | 99.9% | 7.1% |

## 7. Risks and guardrails

- **No automatic fraud decisions.** Every flagged transaction ends with a person: an analyst or the verified customer.
- **Analysts see the reasons.** Each alert shows its top three risk drivers (e.g. "amount is 9x this card's
  usual spend"), so reviewers can challenge the score instead of rubber-stamping it.
- **Known blind spots.** Lowest recall on validation: `category=food_dining` (86.4%, 3 of 22 frauds missed); `category=misc_pos` (92.3%, 3 of 39 frauds missed); `amount_band=$20-100` (92.9%, 4 of 56 frauds missed). Missed frauds look more like ordinary spending: smaller, mostly daytime, less unusual for the card (see `reports/fn_vs_tp_profile.csv`).
- **Fairness.** False-alarm rate (share of genuine transactions flagged): `gender=F` 0.22%, `gender=M` 0.13%; by age from 0.07% (`age_band=60-74`) to 0.24% (`age_band=30-44`). Gender is not a model input, so a gap there comes through correlated behaviour (spending patterns) and should be monitored. Age is a model input. See `reports/fairness_audit.csv`.
- **Drift.** Fraud tactics change. Monitor weekly: alert volume, precision of reviewed alerts, and score
  distribution. Re-pick thresholds monthly on the latest labelled month; retrain quarterly.
- **Label delay.** Fraud labels arrive weeks later (chargebacks), so recent performance is always partly unknown.
- **Data.** Results come from simulated data (Sparkov). The *process* transfers to real data; the *numbers* must be
  re-established on real transactions before any production use.
