"""SARIMA and SARIMAX forecasters for the common rolling loop (CLAUDE.md section 6, lecture 4).

Orders frozen in `src.config` (selected on the pre-test period in 02_exploration.ipynb, section 7).
`statsmodels.tsa.statespace.sarimax.SARIMAX`, as in the course.

Weekly schedule: the parameters used for day D are those estimated on the **Monday of the week of D**,
on the 8 weeks (1,344 h) before that Monday 00:00. On the other days of the week, the prices (and, for
SARIMAX, the exogenous variables) from Monday 00:00 to D-1 23:00 are added to the model state with
`MLEResults.append(..., refit=False)`: the Kalman filter is run on the new observations with the
Monday parameters, which are not re-estimated. The forecasts of a day therefore do not depend on the
first day of the run (e.g. a test period starting on a Wednesday uses the fit of the Monday before).

Estimation: maximum likelihood with the `statsmodels` defaults (L-BFGS, Kalman filter), except the
iteration budget `maxiter` (default 50). With the 4 exogenous variables, SARIMAX needs 54 to 69
iterations: at 50, the optimizer stops with a `ConvergenceWarning` on most weekly fits, although the
log-likelihood is within 0.01 of its maximum (checked on 3 fits; 04_sarima.ipynb, section 3). Raising
the budget lets the default estimator reach its optimum; it is an optimizer setting, not a model
hyperparameter (same reasoning as `maxfun` for Holt-Winters in `src.baselines`).
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import EXOG, FIT_WINDOW_HOURS, HORIZON, SARIMA_ORDER, SARIMA_SEASONAL_ORDER, TARGET

ONE_HOUR = pd.Timedelta(hours=1)


class Sarima:
    """SARIMA (`exog=None`) or SARIMAX (`exog` = list of columns, by default `config.EXOG`).

    SARIMAX is a regression with SARIMA errors: price_t = x_t' beta + u_t, u_t ~ SARIMA with the same
    orders as SARIMA. For day D, x_t is read from `history` over the fit window and from `future_cov`
    (the day-ahead forecasts and calendar of D) for the 24 forecast hours.

    Internal state: the last weekly fit, reused while the Monday and the fit window data are unchanged.
    Every fit is logged in `fit_log` (fit day, convergence, warnings, parameters).
    Returns no quantiles.
    """

    def __init__(self, exog: list[str] | None = None, window: int = FIT_WINDOW_HOURS,
                 order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER, maxiter: int = 1_000):
        self.exog = list(exog) if exog is not None else None
        self.maxiter = maxiter
        self.window, self.order, self.seasonal_order = window, tuple(order), tuple(seasonal_order)
        self._fit_monday = None  # Monday of the current weekly fit
        self._fit_window = None  # data of the current fit window (to detect a different history)
        self._fitted = None  # statsmodels results of the current weekly fit
        self.fit_log: list[dict] = []

    def _exog(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        return None if self.exog is None else frame[self.exog].astype(float)

    def fit(self, history: pd.DataFrame):
        """Estimate the parameters on the last `window` hours of `history`; returns the statsmodels results.

        Convergence and estimation warnings are recorded in `fit_log`, not raised.
        """
        if len(history) < self.window:
            raise ValueError(f"history has {len(history)} rows, the fit window needs {self.window}")
        data = history.iloc[-self.window :]
        model = SARIMAX(data[TARGET], exog=self._exog(data), order=self.order, seasonal_order=self.seasonal_order)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = model.fit(disp=False, maxiter=self.maxiter)
        self.fit_log.append({
            "fit_day": data.index[-1].normalize() + pd.Timedelta(days=1),
            "converged": bool(results.mle_retvals["converged"]),
            "iterations": int(results.mle_retvals["iterations"]),
            "n_warnings": len(caught),
            "warnings": "; ".join(sorted({f"{type(w.message).__name__}: {w.message}" for w in caught})),
            **results.params.to_dict(),
        })
        return results

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, None]:
        day = future_cov.index[0]
        assert day == day.normalize() and history.index[-1] == day - ONE_HOUR, "history must end at D-1 23:00"
        monday = day - pd.Timedelta(days=day.day_of_week)

        # Weekly fit, on the window that ends on Sunday 23:00 before the Monday of the week of D
        before_monday = history.loc[: monday - ONE_HOUR]
        window_data = before_monday.iloc[-self.window :][[TARGET, *(self.exog or [])]]
        if self._fit_monday != monday or not window_data.equals(self._fit_window):
            self._fitted = self.fit(before_monday)
            self._fit_monday, self._fit_window = monday, window_data

        # Tuesday to Sunday: add the observations from Monday 00:00 to D-1 23:00, same parameters
        results = self._fitted
        if day > monday:
            new = history.loc[monday:]
            results = results.append(new[TARGET], exog=self._exog(new), refit=False)

        forecast = results.forecast(HORIZON, exog=self._exog(future_cov))
        assert forecast.index.equals(future_cov.index), f"Forecast hours do not match day D: {forecast.index[0]} …"
        return forecast.to_numpy(dtype=float), None

    def fit_table(self) -> pd.DataFrame:
        """`fit_log` as a DataFrame, one row per weekly fit."""
        return pd.DataFrame(self.fit_log)
