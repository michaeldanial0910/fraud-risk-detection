"""Feature engineering + preprocessing.

Two kinds of features:
1. Card-history ("behavioural") features -- built ONCE on the full chronologically sorted table,
   each from the same card's EARLIER transactions only. They cannot live inside the sklearn
   pipeline because they need the card's history, not just the row. They use no labels, so they
   are exactly what a real authorisation system would know at the moment of the transaction.
2. The sklearn preprocessor (impute / scale / one-hot) -- inside the model pipeline, fitted on
   train only, same pattern as the churn project."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CARD_COL, TIME_COL

CATEGORICAL = ["category"]
NUMERIC = [
    "amt", "hour", "is_night", "day_of_week", "age", "city_pop",
    # card history (past only)
    "card_prior_txns", "hours_since_prev", "n_txn_24h", "amt_sum_24h",
    "amt_vs_card_avg", "new_category_for_card",
]
FEATURES = NUMERIC + CATEGORICAL
ENGINEERED = ["hour", "is_night", "day_of_week", "card_prior_txns", "hours_since_prev",
              "n_txn_24h", "amt_sum_24h", "amt_vs_card_avg", "new_category_for_card"]

GAP_CAP_H = 24 * 30   # first transaction of a card / long gaps are capped at 30 days


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects clean() output (sorted by time). Returns the same rows in the same order + features."""
    df = df.copy()
    t = df[TIME_COL]
    df["hour"] = t.dt.hour
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 4)).astype(int)
    df["day_of_week"] = t.dt.dayofweek

    # --- card history: sort by card then time, compute on PREVIOUS rows only, restore order ---
    order = df.sort_values([CARD_COL, TIME_COL], kind="stable").index
    d = df.loc[order, [CARD_COL, TIME_COL, "amt", "category"]]
    g = d.groupby(CARD_COL, sort=False)

    prior = g.cumcount()                                         # txns this card made before
    gap_h = g[TIME_COL].diff().dt.total_seconds() / 3600
    prev_mean = g["amt"].transform(lambda s: s.shift(1).expanding().mean())
    seen_cat = d.groupby([CARD_COL, "category"], sort=False).cumcount()

    # 24h rolling window that EXCLUDES the current transaction (closed="left")
    roll = (d.set_index(TIME_COL).groupby(CARD_COL, sort=False)["amt"]
              .rolling("24h", closed="left"))
    n24 = roll.count().to_numpy()                                # same card-then-time order as d
    s24 = roll.sum().to_numpy()

    hist = pd.DataFrame({
        "card_prior_txns": prior.to_numpy(),
        "hours_since_prev": np.minimum(gap_h.fillna(GAP_CAP_H).to_numpy(), GAP_CAP_H),
        "n_txn_24h": np.nan_to_num(n24, nan=0.0),
        "amt_sum_24h": np.nan_to_num(s24, nan=0.0),
        "amt_vs_card_avg": (d["amt"] / prev_mean).fillna(1.0).to_numpy(),
        "new_category_for_card": (seen_cat == 0).astype(int).to_numpy(),
    }, index=order)
    return df.join(hist)


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    num_steps = [SimpleImputer(strategy="median")]
    if scale:
        num_steps.append(StandardScaler())
    return ColumnTransformer(
        [("num", make_pipeline(*num_steps), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def preprocessing_steps(scale: bool = False) -> list:
    """Flat list of steps so imblearn pipelines (SMOTE) can reuse them."""
    return [("prep", build_preprocessor(scale))]


def build_pipeline(clf, scale: bool = False) -> Pipeline:
    return Pipeline(preprocessing_steps(scale) + [("clf", clf)])
