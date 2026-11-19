"""Baselines for the common rolling loop (CLAUDE.md section 6): naive forecasts and additive Holt-Winters.

Each forecaster only reads the past prices in `history` (up to D-1 23:00) and, for the reference naive,
the day of week of D in `future_cov`. They return no quantiles.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import FIT_WINDOW_HOURS, HORIZON, HW_SEASON, TARGET

# Days of week (pandas convention, Monday = 0) on which the reference naive uses D-7 instead of D-1
REFERENCE_D7_DAYS = (0, 5, 6)  # Monday, Saturday, Sunday


def same_hours_days_before(history: pd.DataFrame, future_cov: pd.DataFrame, lag_days: int) -> np.ndarray:
    """Prices of the 24 hours of day D - `lag_days` (1 to 7), taken from `history`.

    Checks that the selected rows are exactly the hours of D shifted by `lag_days` days (24 rows per day).
    """
    if not 1 <= lag_days <= 7:
        raise ValueError(f"lag_days must be between 1 and 7, got {lag_days}")
    start = len(history) - lag_days * HORIZON
    if start < 0:
        raise ValueError(f"history has {len(history)} rows, D-{lag_days} needs {lag_days * HORIZON}")
    past = history[TARGET].iloc[start : start + HORIZON]
    expected = future_cov.index - pd.Timedelta(days=lag_days)
    assert past.index.equals(expected), f"D-{lag_days} hours do not match: {past.index[0]} … {past.index[-1]}"
    return past.to_numpy(dtype=float)


class NaiveD1:
    """Price at the same hour the previous day (D-1)."""

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, None]:
        return same_hours_days_before(history, future_cov, 1), None


class NaiveD7:
    """Price at the same hour one week earlier (D-7)."""

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, None]:
        return same_hours_days_before(history, future_cov, 7), None


class NaiveReference:
    """Reference naive (epftoolbox convention, rMAE denominator): D-7 if D is a Monday, Saturday or Sunday,
    D-1 otherwise.

    The choice only depends on the day of week of D, not on `non_working_day`: a public holiday on a weekday
    uses D-1, as in `epftoolbox.evaluation.naive_forecast` with `m=None` (the "standard naive forecast in
    electricity price forecasting" behind epftoolbox's rMAE; Lago et al., 2021).
    """

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, None]:
        dow = int(future_cov["dow"].iloc[0])
        assert (future_cov["dow"] == dow).all(), "future_cov must cover a single day"
        lag_days = 7 if dow in REFERENCE_D7_DAYS else 1
        return same_hours_days_before(history, future_cov, lag_days), None


class HoltWinters:
    """Additive Holt-Winters (lecture 5): level and weekly seasonality (period 168 h), no trend.

    Refitted every day D on the last `FIT_WINDOW_HOURS` (8 weeks = 1,344 h) of prices of `history`, then
    forecasts the 24 hours of D. Recursions (additive, no trend; p = 168):
        L_n = alpha * (y_n - S_{n-p}) + (1 - alpha) * L_{n-1}
        S_n = gamma * (y_n - L_n) + (1 - gamma) * S_{n-p}
        forecast y_{n+i} = L_n + S_{n+i-p},  i = 1 … 24.
    Seasonality is additive because the multiplicative form divides by the seasonal factor and requires
    strictly positive data, which negative prices rule out.

    `statsmodels.tsa.holtwinters.ExponentialSmoothing` with its default settings: alpha, gamma and the
    initial level and 168 seasonal values are estimated jointly by least squares on the one-step-ahead
    errors (`initialization_method="estimated"`). The only change is the L-BFGS-B evaluation budget
    (`maxfun`, default 15,000): with 170 parameters and numerical gradients, the default budget stops the
    optimizer before convergence (checked on several test days). Raising it reaches the least-squares
    estimate (it is an optimizer setting, not a model hyperparameter).
    """

    def __init__(self, window: int = FIT_WINDOW_HOURS, season: int = HW_SEASON, maxfun: int = 1_000_000):
        if window < 2 * season:
            raise ValueError(f"The fit window ({window} h) must cover at least two seasons of {season} h")
        self.window, self.season, self.maxfun = window, season, maxfun

    def fit(self, history: pd.DataFrame):
        """Fit on the last `window` hours of prices; returns the statsmodels results."""
        if len(history) < self.window:
            raise ValueError(f"history has {len(history)} rows, the fit window needs {self.window}")
        y = history[TARGET].iloc[-self.window :]
        model = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=self.season)
        return model.fit(minimize_kwargs={"options": {"maxfun": self.maxfun}})

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, None]:
        forecast = self.fit(history).forecast(HORIZON)
        assert forecast.index.equals(future_cov.index), f"Forecast hours do not match day D: {forecast.index[0]} …"
        return forecast.to_numpy(dtype=float), None
