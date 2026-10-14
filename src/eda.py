"""Small EDA summary tables for the Streamlit app's "The data" page (so the app never loads 1.3M rows).
Run: python -m src.eda   (after the feature cache exists, i.e. after src.train or the EDA notebook)"""
import json

import pandas as pd

from src.config import CARD_COL, TARGET, TIME_COL
from src.data import load_dataset
from src.train import REPORTS


def main():
    df = load_dataset()
    fraud = df[df[TARGET] == 1]

    monthly = df.set_index(TIME_COL).resample("MS")[TARGET].agg(["size", "sum"]).reset_index()
    monthly.columns = ["month", "transactions", "frauds"]
    monthly["fraud_rate"] = monthly["frauds"] / monthly["transactions"]
    monthly.to_csv(REPORTS / "eda_monthly.csv", index=False)

    hourly = df.groupby("hour")[TARGET].agg(["size", "sum", "mean"]).reset_index()
    hourly.columns = ["hour", "transactions", "frauds", "fraud_rate"]
    hourly.to_csv(REPORTS / "eda_hourly.csv", index=False)

    cat = df.groupby("category").agg(transactions=(TARGET, "size"), frauds=(TARGET, "sum"),
                                     fraud_rate=(TARGET, "mean")).reset_index()
    cat["fraud_amt"] = cat["category"].map(fraud.groupby("category")["amt"].sum()).fillna(0)
    cat.sort_values("fraud_rate", ascending=False).to_csv(REPORTS / "eda_category.csv", index=False)

    feats = ["amt", "amt_vs_card_avg", "n_txn_24h", "amt_sum_24h", "hours_since_prev"]
    beh = df.groupby(TARGET)[feats].median().T.rename(columns={0: "legit_median", 1: "fraud_median"})
    beh.rename_axis("feature").reset_index().to_csv(REPORTS / "eda_card_behaviour.csv", index=False)

    bursts = fraud.groupby(CARD_COL)[TIME_COL].agg(["size", "min", "max"])
    span_days = (bursts["max"] - bursts["min"]).dt.total_seconds() / 86400
    summary = {
        "transactions": int(len(df)), "frauds": int(len(fraud)), "fraud_rate": float(df[TARGET].mean()),
        "fraud_share_of_dollars": float(fraud["amt"].sum() / df["amt"].sum()),
        "cards": int(df[CARD_COL].nunique()), "cards_defrauded": int(len(bursts)),
        "median_frauds_per_card": float(bursts["size"].median()), "median_burst_days": float(span_days.median()),
        "night_share_of_frauds": float(fraud["is_night"].mean()),
        "night_share_of_legit": float(df.loc[df[TARGET] == 0, "is_night"].mean()),
        "start": str(df[TIME_COL].min().date()), "end": str(df[TIME_COL].max().date()),
    }
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
