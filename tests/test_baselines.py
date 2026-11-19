import numpy as np
import pandas as pd
import pytest

from src.baselines import NaiveD1, NaiveD7, NaiveReference, same_hours_days_before
from src.config import HORIZON, TARGET
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
