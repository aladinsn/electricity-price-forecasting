import numpy as np
import pandas as pd
import pytest

from src.baselines import HoltWinters, NaiveD1, NaiveD7, NaiveReference, same_hours_days_before
from src.config import FIT_WINDOW_HOURS, HORIZON, HW_SEASON, TARGET
from src.data import split_for_day
from src.evaluation import RESULT_COLUMNS, QUANTILE_COLUMNS, run_rolling
from tests.conftest import make_toy

# make_toy: price of day d at hour h is 100 * d + h, so D-1 is off by 100 and D-7 by 700 at every hour.
WEEK = pd.date_range("2026-04-06", "2026-04-12", freq="D")  # Monday to Sunday


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return make_toy(start="2026-03-30", n_days=14)  # starts on a Monday, 7 days of history before WEEK


@pytest.mark.parametrize("model, lag_days", [(NaiveD1(), 1), (NaiveD7(), 7)])
def test_naive_copies_the_lagged_day(toy_df, model, lag_days):
    day = WEEK[2]
    history, future_cov = split_for_day(toy_df, day)
    y_pred, quantiles = model.predict(history, future_cov)

    assert quantiles is None
    assert isinstance(y_pred, np.ndarray) and y_pred.shape == (HORIZON,)
    expected = toy_df.loc[day - pd.Timedelta(days=lag_days) :, TARGET].iloc[:HORIZON].to_numpy()
    np.testing.assert_array_equal(y_pred, expected)


def test_reference_naive_lag_by_day_of_week(toy_df):
    # Monday, Saturday, Sunday -> D-7 (off by 700); Tuesday to Friday -> D-1 (off by 100)
    expected_error = {0: 700, 1: 100, 2: 100, 3: 100, 4: 100, 5: 700, 6: 700}
    for day in WEEK:
        history, future_cov = split_for_day(toy_df, day)
        y_pred, _ = NaiveReference().predict(history, future_cov)
        actual = toy_df.loc[future_cov.index, TARGET].to_numpy()
        np.testing.assert_array_equal(actual - y_pred, expected_error[day.day_of_week])


def test_reference_naive_ignores_public_holidays(toy_df):
    # A public holiday on a Tuesday is a non-working day, but the reference naive still uses D-1
    df = toy_df.copy()
    day = WEEK[1]
    df.loc[day : day + pd.Timedelta(hours=23), "non_working_day"] = 1
    history, future_cov = split_for_day(df, day)
    y_pred, _ = NaiveReference().predict(history, future_cov)
    np.testing.assert_array_equal(y_pred, history[TARGET].iloc[-HORIZON:].to_numpy())


def test_naive_d7_rejects_short_history(toy_df):
    history, future_cov = split_for_day(toy_df, pd.Timestamp("2026-04-05"))  # 6 days of history
    with pytest.raises(ValueError):
        NaiveD7().predict(history, future_cov)
    with pytest.raises(ValueError):
        same_hours_days_before(history, future_cov, 0)


def test_naives_in_rolling_loop(toy_df, tmp_path):
    for name, model, error in [("naive_d1", NaiveD1(), 100), ("naive_d7", NaiveD7(), 700)]:
        results = run_rolling(model, name, toy_df, WEEK, results_dir=tmp_path)
        assert list(results.columns) == RESULT_COLUMNS and len(results) == HORIZON * len(WEEK)
        assert results[QUANTILE_COLUMNS].isna().all().all()  # no quantiles
        assert ((results["y_true"] - results["y_pred"]) == error).all()

    ref = run_rolling(NaiveReference(), "naive_ref", toy_df, WEEK, results_dir=tmp_path)
    errors = (ref["y_true"] - ref["y_pred"]).groupby(ref["target_date"].dt.day_of_week).unique()
    assert {dow: list(e) for dow, e in errors.items()} == {0: [700], 1: [100], 2: [100], 3: [100], 4: [100], 5: [700], 6: [700]}


def make_weekly(n_weeks: int, seed: int = 0) -> pd.DataFrame:
    """Dataset whose price is a fixed weekly profile (with negative hours) plus small noise."""
    df = make_toy(start="2026-01-05", n_days=7 * n_weeks)  # starts on a Monday
    rng = np.random.default_rng(seed)
    position = (df.index.day_of_week * 24 + df.index.hour).to_numpy()
    profile = 60 + 40 * np.sin(2 * np.pi * np.arange(HW_SEASON) / 24) - 30 * (np.arange(HW_SEASON) >= 120)
    df[TARGET] = profile[position] + rng.normal(0, 1, len(df))
    assert (df[TARGET] < 0).any()  # additive seasonality must handle negative prices
    return df


def test_holt_winters_recovers_weekly_profile():
    df = make_weekly(n_weeks=10)
    day = df.index[-1].normalize() - pd.Timedelta(days=1)  # a Saturday, low-price regime of the profile
    history, future_cov = split_for_day(df, day)
    y_pred, quantiles = HoltWinters().predict(history, future_cov)

    assert quantiles is None and y_pred.shape == (HORIZON,)
    actual = df.loc[future_cov.index, TARGET].to_numpy()
    assert np.abs(y_pred - actual).mean() < 3  # noise sd is 1 EUR/MWh


def test_holt_winters_only_uses_the_fit_window():
    df = make_weekly(n_weeks=10)
    day = df.index[-1].normalize()
    history, future_cov = split_for_day(df, day)
    altered = history.copy()
    altered.iloc[: len(history) - FIT_WINDOW_HOURS, altered.columns.get_loc(TARGET)] = 1e4  # before the window
    y_pred, _ = HoltWinters().predict(history, future_cov)
    y_altered, _ = HoltWinters().predict(altered, future_cov)
    np.testing.assert_array_equal(y_pred, y_altered)


def test_holt_winters_rejects_short_history():
    df = make_weekly(n_weeks=8)  # 1,344 h in total, so history (up to D-1) is shorter than the window
    history, future_cov = split_for_day(df, df.index[-1].normalize())
    with pytest.raises(ValueError):
        HoltWinters().predict(history, future_cov)
    with pytest.raises(ValueError):
        HoltWinters(window=HW_SEASON)
