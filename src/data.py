"""Data loading, cleaning and the TIME-BASED split.

Why time-based (and not the stratified random split used for churn): fraud patterns drift and
fraudsters hit the same card in bursts. A random split puts transactions from the same burst in
train and test, so the model is graded on the "future" it already saw. Here the model is always
evaluated on transactions that happen strictly after everything it was trained on.

Everything that LEARNS from data (imputer medians, scaler, one-hot vocabulary) stays inside the
sklearn pipeline, fitted on train only. Card-history features are built in features.py from each
card's PAST transactions only, so they are known at authorisation time and leak nothing."""
from pathlib import Path

import pandas as pd

from src.config import CARD_COL, RAW_FILE, TARGET, TIME_COL, TRAIN_END, VAL_END

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED / "features.parquet"

# Columns we read from the raw file. Names, street, job, merchant name, trans_num are dropped:
# PII or identifiers with no legitimate signal. gender is kept ONLY for the fairness audit.
USECOLS = ["trans_date_trans_time", "cc_num", "category", "amt", "gender",
           "city_pop", "dob", TARGET]


def load_raw(filename: str = RAW_FILE) -> pd.DataFrame:
    return pd.read_csv(RAW / filename, usecols=USECOLS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, deterministic cleaning only (nothing is learned from data -> no leakage).
    - parse the timestamp, derive the cardholder's age AT the transaction (dob itself is dropped)
    - sort chronologically (required by the time split and the card-history features)"""
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df.pop("trans_date_trans_time"))
    dob = pd.to_datetime(df.pop("dob"))
    df["age"] = ((df[TIME_COL] - dob).dt.days / 365.25).round(1)
    df[TARGET] = df[TARGET].astype(int)
    return df.sort_values([TIME_COL, CARD_COL], kind="stable").reset_index(drop=True)


def load_dataset(rebuild: bool = False) -> pd.DataFrame:
    """Clean + feature table, cached as parquet (building card history on 1.3M rows takes ~1 min)."""
    from src.features import add_features
    raw_path = RAW / RAW_FILE
    if (not rebuild and FEATURES_FILE.exists()
            and FEATURES_FILE.stat().st_mtime > raw_path.stat().st_mtime):
        return pd.read_parquet(FEATURES_FILE)
    df = add_features(clean(load_raw()))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_FILE, index=False)
    return df


def split(df: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END):
    """Chronological train / val / test. Val picks the model and the thresholds; test is used once.
    X keeps the non-feature columns (ts, cc_num, gender) for analysis -- the model pipeline
    selects its own feature columns and ignores the rest."""
    t = df[TIME_COL]
    parts = [df[t < train_end], df[(t >= train_end) & (t < val_end)], df[t >= val_end]]
    X_tr, X_val, X_te = (p.drop(columns=[TARGET]) for p in parts)
    y_tr, y_val, y_te = (p[TARGET] for p in parts)
    return X_tr, X_val, X_te, y_tr, y_val, y_te
