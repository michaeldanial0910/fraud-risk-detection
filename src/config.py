"""Single source of truth for project constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = "fraudTrain.csv"          # Sparkov simulated card transactions (Kaggle) -> data/raw/ (see README)
TARGET = "is_fraud"
TIME_COL = "ts"                      # parsed from trans_date_trans_time in data.clean()
CARD_COL = "cc_num"                  # used ONLY as a grouping key for card history, never as a feature
SEED = 42

# Time-based split (fraudTrain.csv covers 2019-01-01 .. 2020-06-21).
# Train on the past, choose model + thresholds on the next quarter, report on the latest months.
TRAIN_END = "2020-01-01"             # train: 2019-01-01 .. 2019-12-31  (12 months)
VAL_END = "2020-04-01"               # val  : 2020-01-01 .. 2020-03-31  (3 months)
                                     # test : 2020-04-01 .. end of file (~2.7 months), touched once

# ---------------------------------------------------------------------------------------------
# Business-cost ASSUMPTIONS (USD, per transaction). Not facts from the data -- change and re-run.
#   Missed fraud (FN)  : the issuer reimburses the cardholder -> full amount lost + chargeback handling.
#   Every alert        : an analyst reviews it (~3-4 min of a ~$80/h fully loaded analyst).
#   False alarm (FP)   : additionally, a genuine customer is held / called -> friction, some churn risk.
#   Caught fraud (TP)  : loss prevented; only the review cost is paid.
COST_CHARGEBACK = 20.0
COST_REVIEW = 5.0
COST_FRICTION = 10.0

# Operational constraint: how many alerts the fraud team can review per day.
# Validation/test volume is roughly 2,400 transactions/day, so 60/day ~ 2.5% of traffic.
ALERTS_PER_DAY = 60

# Tier 2 ("hold + verify with customer") is used only where the model is very sure:
HOLD_MIN_PRECISION = 0.90            # pick the lowest threshold whose validation precision >= this

# Tuning budget (each LightGBM fit on ~870k rows takes ~20-60 s on a laptop).
N_TUNE_ITER = 15
SHAP_SAMPLE = 5000                   # rows used for global SHAP plots (all val frauds + random legit)
