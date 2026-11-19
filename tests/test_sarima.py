import numpy as np
import pandas as pd
import pytest

from src.config import EXOG, HORIZON, TARGET
from src.data import split_for_day
from src.evaluation import QUANTILE_COLUMNS, RESULT_COLUMNS, run_rolling
from src.sarima import Sarima
from tests.conftest import make_toy

WINDOW = 14 * 24  # short fit window to keep the tests fast


def make_seasonal(n_days: int = 35, seed: int = 0) -> pd.DataFrame:
    """Dataset starting on a Monday: price = daily profile + 3 * wind_fc + AR(1) noise."""
    df = make_toy(start="2026-03-02", n_days=n_days)
    rng = np.random.default_rng(seed)
    n = len(df)
    df["wind_fc"] = 20 + 10 * np.sin(2 * np.pi * np.arange(n) / (24 * 5.3)) + rng.normal(0, 3, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 2)
    df[TARGET] = 50 + 30 * np.sin(2 * np.pi * df.index.hour / 24) + 3 * df["wind_fc"] + noise
    return df


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_seasonal()


def test_weekly_refit_and_daily_update(df):
    model = Sarima(window=WINDOW)
    days = pd.date_range("2026-03-18", "2026-03-31", freq="D")  # Wednesday to Tuesday, crosses 2 Mondays
    forecasts = {}
    for day in days:
        y_pred, quantiles = model.predict(*split_for_day(df, day))
        assert quantiles is None and y_pred.shape == (HORIZON,) and np.isfinite(y_pred).all()
        forecasts[day] = y_pred

    # One fit per week, dated on the Monday of the week of D, on the window ending Sunday 23:00
    fits = model.fit_table()
    assert list(fits["fit_day"]) == list(pd.to_datetime(["2026-03-16", "2026-03-23", "2026-03-30"]))

    # The forecast of a day does not depend on the first day of the run
    day = pd.Timestamp("2026-03-26")
    y_alone, _ = Sarima(window=WINDOW).predict(*split_for_day(df, day))
    np.testing.assert_allclose(y_alone, forecasts[day])


def test_daily_update_uses_the_new_prices_without_refitting(df):
    model = Sarima(window=WINDOW)
    tuesday, wednesday = pd.Timestamp("2026-03-24"), pd.Timestamp("2026-03-25")
    y_tue, _ = model.predict(*split_for_day(df, tuesday))
    y_wed, _ = model.predict(*split_for_day(df, wednesday))
    assert len(model.fit_log) == 1  # same weekly parameters
    assert not np.allclose(y_tue, y_wed)

    # Changing a price of the current week (after the fit window) changes the forecast, not the fit
    history, future_cov = split_for_day(df, wednesday)
    altered = history.copy()
    altered.iloc[-1, altered.columns.get_loc(TARGET)] += 50
    y_altered, _ = model.predict(altered, future_cov)
    assert len(model.fit_log) == 1 and not np.allclose(y_wed, y_altered)


def test_only_uses_the_fit_window_and_refits_on_a_different_history(df):
    day = pd.Timestamp("2026-04-01")
    history, future_cov = split_for_day(df, day)
    model = Sarima(window=WINDOW)
    y_pred, _ = model.predict(history, future_cov)

    # Data before the 8-week (here 2-week) window before Monday is ignored
    monday = day - pd.Timedelta(days=day.day_of_week)
    altered = history.copy()
    altered.loc[: monday - pd.Timedelta(hours=WINDOW + 1), TARGET] = 1e4
    np.testing.assert_array_equal(Sarima(window=WINDOW).predict(altered, future_cov)[0], y_pred)

    # A different fit window on the same Monday triggers a new fit instead of reusing the cached one
    altered = history.copy()
    altered.loc[monday - pd.Timedelta(hours=1), TARGET] += 50
    model.predict(altered, future_cov)
    assert len(model.fit_log) == 2


def test_sarimax_uses_the_future_covariates(df):
    day = pd.Timestamp("2026-03-31")
    history, future_cov = split_for_day(df, day)
    actual = df.loc[future_cov.index, TARGET].to_numpy()
    sarima, sarimax = Sarima(window=WINDOW), Sarima(exog=EXOG, window=WINDOW)

    y_sarima, _ = sarima.predict(history, future_cov)
    y_sarimax, _ = sarimax.predict(history, future_cov)
    assert np.abs(y_sarimax - actual).mean() < np.abs(y_sarima - actual).mean()
    assert sarimax.fit_log[0]["wind_fc"] == pytest.approx(3, abs=0.5)  # true coefficient

    shifted = future_cov.assign(wind_fc=future_cov["wind_fc"] + 10)
    y_shifted, _ = sarimax.predict(history, shifted)
    np.testing.assert_allclose(y_shifted - y_sarimax, 10 * sarimax.fit_log[0]["wind_fc"])


def test_rejects_short_history(df):
    model = Sarima(window=WINDOW)
    with pytest.raises(ValueError):
        model.predict(*split_for_day(df, pd.Timestamp("2026-03-12")))  # 7 days before its Monday


def test_sarima_in_rolling_loop(df, tmp_path):
    days = pd.date_range("2026-03-30", "2026-04-02", freq="D")
    results = run_rolling(Sarima(exog=EXOG, window=WINDOW), "sarimax", df, days, results_dir=tmp_path)
    assert list(results.columns) == RESULT_COLUMNS and len(results) == HORIZON * len(days)
    assert results[QUANTILE_COLUMNS].isna().all().all()
