"""Guards for the properties that matter: chronological split, card-history features that only look
at the past, and correct decision/cost logic.
Run: pytest -q   (data tests need data/raw/fraudTrain.csv; the model test also needs `python -m src.train`)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RAW_FILE, TARGET, TIME_COL  # noqa: E402
from src.data import RAW, split  # noqa: E402
from src.features import FEATURES, add_features  # noqa: E402

needs_data = pytest.mark.skipif(not (RAW / RAW_FILE).exists(), reason="raw data not downloaded")
MODEL = ROOT / "models" / "fraud_model.joblib"


def _toy(times, amts, cats=None, card=1):
    """Minimal clean()-style frame for one card."""
    n = len(times)
    return pd.DataFrame({TIME_COL: pd.to_datetime(times), "cc_num": card, "amt": amts,
                         "category": cats or ["grocery_pos"] * n, "gender": "F",
                         "city_pop": 1000, "age": 40.0, TARGET: 0})


# ---------------------------------------------------------------- feature logic (synthetic, no data needed)
def test_card_history_uses_only_past_transactions():
    times = ["2020-01-01 10:00", "2020-01-01 12:00", "2020-01-01 20:00", "2020-01-03 09:00"]
    f = add_features(_toy(times, [10.0, 20.0, 30.0, 40.0]))
    assert f["card_prior_txns"].tolist() == [0, 1, 2, 3]
    assert f["n_txn_24h"].tolist() == [0, 1, 2, 0]          # current row excluded; 3rd day outside window
    assert f["amt_sum_24h"].tolist() == [0, 10, 30, 0]
    assert f.loc[0, "amt_vs_card_avg"] == 1.0                # no history -> neutral
    assert f.loc[2, "amt_vs_card_avg"] == pytest.approx(30 / 15)
    assert f["hours_since_prev"].tolist()[1:] == [2.0, 8.0, 37.0]


def test_future_rows_do_not_change_past_features():
    """Leakage guard: changing a LATER transaction must not change any earlier row's features."""
    times = ["2020-01-01 10:00", "2020-01-01 12:00", "2020-01-01 20:00", "2020-01-02 09:00"]
    a = add_features(_toy(times, [10.0, 20.0, 30.0, 40.0]))
    b = add_features(_toy(times, [10.0, 20.0, 30.0, 9999.0]))
    cols = [c for c in FEATURES if c != "category"]
    pd.testing.assert_frame_equal(a.loc[:2, cols], b.loc[:2, cols])


def test_new_category_flag_and_row_order_preserved():
    times = ["2020-01-01 10:00", "2020-01-02 10:00", "2020-01-03 10:00"]
    df = pd.concat([_toy(times, [1.0, 2.0, 3.0], ["a", "b", "a"], card=1),
                    _toy(times, [5.0, 6.0, 7.0], card=2)]).sort_values(TIME_COL, kind="stable")
    f = add_features(df.reset_index(drop=True))
    assert (f[TIME_COL].diff().dropna() >= pd.Timedelta(0)).all()     # still chronological
    assert f[f.cc_num == 1]["new_category_for_card"].tolist() == [1, 1, 0]


# ---------------------------------------------------------------- decision logic (synthetic)
def test_cost_curve_matches_direct_cost_computation():
    from src.evaluate import cost_curve, txn_costs
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000); y = (rng.uniform(0, 1, 5000) < p * 0.05).astype(int)
    amt = rng.lognormal(4, 1, 5000)
    c = cost_curve(y, p, amt, n_days=10, thresholds=[0.3, 0.7])
    for t, row in zip([0.3, 0.7], c.itertuples()):
        direct = txn_costs(y, (p >= t).astype(int), amt)
        assert row.cost == pytest.approx(direct["cost"])
        assert row.alerts_per_day == pytest.approx(direct["n_alerts"] / 10)


def test_thresholds_respect_capacity_and_tiers_are_ordered():
    from src.evaluate import assign_tier, choose_thresholds, cost_curve
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000); y = (rng.uniform(0, 1, 20000) < p ** 4).astype(int)
    c = cost_curve(y, p, np.full(20000, 500.0), n_days=100)
    t = choose_thresholds(c, max_alerts_per_day=20, hold_min_precision=0.9)
    assert c.loc[(c.threshold - t["review"]).abs().idxmin(), "alerts_per_day"] <= 20
    assert t["review"] <= t["hold"]
    tiers = assign_tier(np.array([0.0, t["review"], t["hold"]]), t["review"], t["hold"])
    assert tiers.tolist() == ["approve", "review", "hold_verify"]


# ---------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def df():
    from src.data import load_dataset
    return load_dataset()


@needs_data
def test_split_is_chronological_disjoint_and_complete(df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(df)
    assert X_tr[TIME_COL].max() < X_va[TIME_COL].min()
    assert X_va[TIME_COL].max() < X_te[TIME_COL].min()
    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
    assert all(y.sum() > 0 for y in (y_tr, y_va, y_te))


@needs_data
def test_clean_drops_pii_and_has_binary_target(df):
    assert set(df[TARGET].unique()) == {0, 1}
    assert not {"first", "last", "street", "trans_num", "merchant", "dob"} & set(df.columns)
    assert df[FEATURES].isna().sum().sum() == 0


@needs_data
@pytest.mark.skipif(not MODEL.exists(), reason="model not trained")
def test_saved_model_scores_rows_and_handles_unseen_category(df):
    import joblib
    model = joblib.load(MODEL)
    X = df.drop(columns=[TARGET]).head(50).copy()
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()
    X.loc[X.index[0], "category"] = "crypto_exchange"      # never seen in training
    assert np.isfinite(model.predict_proba(X)[0, 1])
