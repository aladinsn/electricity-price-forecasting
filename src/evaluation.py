"""Rolling evaluation loop shared by all models, and metrics."""

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import stats

from src.config import HORIZON, QUANTILE_COLUMNS, QUANTILES, RESULTS_DIR, ROOT, TARGET
from src.data import split_for_day

RESULT_COLUMNS = ["model", "target_date", "hour", "y_true", "y_pred", *QUANTILE_COLUMNS]


def _display_path(path: Path) -> str:
    """Path relative to the project root when possible, so that printed outputs do not show the local machine."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


class Forecaster(Protocol):
    """Common interface. A model may keep an internal state as long as it only sees `history`."""

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
        """Return 24 point forecasts and, if available, the quantiles (columns q10 … q90)."""
        ...


def run_rolling(
    model: Forecaster, name: str, df: pd.DataFrame, days: pd.DatetimeIndex, results_dir: Path = RESULTS_DIR
) -> pd.DataFrame:
    """Forecast each day of `days`, moving forward one day at a time; long-format results (RESULT_COLUMNS).

    For each day D, the model only receives `split_for_day(df, D)`; the actual prices of D are read
    afterwards, to fill `y_true`. Results are saved to <results_dir>/<name>.parquet.

    Reads <results_dir>/<name>.parquet if it exists instead of recomputing. The cached file must cover
    exactly `days`: evaluate another period in another directory (e.g. RESULTS_DIR / "control").
    """
    days = pd.DatetimeIndex(days)
    if not (days.is_monotonic_increasing and days.is_unique):
        raise ValueError("days must be strictly increasing (rolling origin)")
    path = Path(results_dir) / f"{name}.parquet"

    if path.exists():
        results = pd.read_parquet(path)
        cached_days = pd.DatetimeIndex(results["target_date"].unique())
        if list(results.columns) != RESULT_COLUMNS or not cached_days.equals(days):
            raise ValueError(f"{_display_path(path)} does not cover the requested days: delete it or use another results_dir")
        print(f"{name}: loaded from {_display_path(path)}")
        return results

    start = time.perf_counter()
    frames = []
    for day in days:
        history, future_cov = split_for_day(df, day)
        y_pred, quantiles = model.predict(history, future_cov)
        y_true = df.loc[future_cov.index, TARGET].to_numpy()
        frames.append(_day_results(name, day, future_cov["hour"].to_numpy(), y_true, y_pred, quantiles))
    results = pd.concat(frames, ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(path, index=False)
    print(f"{name}: {len(days)} days in {time.perf_counter() - start:.1f} s, saved to {_display_path(path)}")
    return results


def _day_results(name, day, hours, y_true, y_pred, quantiles) -> pd.DataFrame:
    """Long-format results of one day; checks the shape and finiteness of the model output."""
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != (HORIZON,) or not np.isfinite(y_pred).all():
        raise ValueError(f"{name}, {day.date()}: expected {HORIZON} finite point forecasts, got {y_pred}")

    if quantiles is None:
        q = np.full((HORIZON, len(QUANTILE_COLUMNS)), np.nan)
    else:
        q = np.asarray(quantiles[QUANTILE_COLUMNS], dtype=float)
        if q.shape != (HORIZON, len(QUANTILE_COLUMNS)) or not np.isfinite(q).all():
            raise ValueError(f"{name}, {day.date()}: expected {HORIZON} rows of finite quantiles {QUANTILE_COLUMNS}")

    results = pd.DataFrame({"model": name, "target_date": day, "hour": hours, "y_true": y_true, "y_pred": y_pred})
    results[QUANTILE_COLUMNS] = q
    return results


def _as_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays of the same shape (positional, no pandas index alignment)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in EUR/MWh."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in EUR/MWh."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmae(y_true, y_pred, y_naive) -> float:
    """Model MAE / reference naive MAE (D-7 on Monday, Saturday, Sunday; D-1 otherwise).

    `y_naive` holds the reference naive forecasts of the same hours as `y_true`. Below 1: better than the naive.
    """
    return mae(y_true, y_pred) / mae(y_true, y_naive)


def pinball(y_true, quantiles: pd.DataFrame) -> float:
    """Mean pinball loss over the 10th … 90th percentile levels, in EUR/MWh.

    For level tau and quantile forecast q: max(tau * (y - q), (tau - 1) * (y - q)), averaged over the
    hours and the 9 levels (columns q10 … q90 of `quantiles`, one row per value of `y_true`).
    """
    q = np.asarray(quantiles[QUANTILE_COLUMNS], dtype=float)
    y = np.asarray(y_true, dtype=float)
    if q.shape != (len(y), len(QUANTILES)):
        raise ValueError(f"Expected quantiles of shape {(len(y), len(QUANTILES))}, got {q.shape}")
    if np.isnan(q).any():
        raise ValueError("Missing quantiles: pinball loss is only defined for probabilistic models")
    tau = np.asarray(QUANTILES)
    diff = y[:, None] - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def daily_errors(results: pd.DataFrame) -> pd.DataFrame:
    """Errors y_true - y_pred of one model as a (days, 24) table: index target_date, columns hour 0 … 23.

    Input: long-format results of a single model (RESULT_COLUMNS), as returned by `run_rolling`.
    """
    if results["model"].nunique() != 1:
        raise ValueError(f"Expected the results of a single model, got {sorted(results['model'].unique())}")
    errors = (results["y_true"] - results["y_pred"]).to_numpy()
    table = pd.DataFrame({"target_date": results["target_date"], "hour": results["hour"], "error": errors})
    table = table.pivot(index="target_date", columns="hour", values="error").sort_index()
    if list(table.columns) != list(range(HORIZON)) or table.isna().any().any():
        raise ValueError(f"Expected {HORIZON} hourly errors for every day")
    return table


def diebold_mariano(errors_a, errors_b, norm: int = 1) -> float:
    """One-sided multivariate Diebold-Mariano test; returns the p-value.

    H0: the forecasts of model B are not more accurate than those of model A.
    H1: model B is more accurate. A small p-value (e.g. < 0.05) means B is significantly better than A;
    swap the arguments to test the other direction.

    `errors_a`, `errors_b`: (days, 24) errors of the two models on the same days (see `daily_errors`).
    The loss differential is computed on the daily vector of 24 errors,
        d_t = mean_h |e_A[t, h]|^norm - mean_h |e_B[t, h]|^norm,     norm = 1 (MAE) or 2 (MSE),
    and DM = mean(d) / sqrt(var(d) / N) (population variance, N days) is compared with a standard normal:
    p = P(Z > DM). Unlike the original DM test, there is no autocorrelation (HAC) correction: d_t is a
    daily series of one-step-ahead (day D) forecasts, as in epftoolbox.

    Reimplementation of the `version="multivariate"` branch of `epftoolbox.evaluation.DM`
    (J. Lago, https://github.com/jeslago/epftoolbox, Apache-2.0; method in Lago, Marcjasz, De Schutter
    and Weron, "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best
    practices and an open-access benchmark", Applied Energy 293, 2021). epftoolbox takes the actual prices
    and the two forecasts; this version takes the errors directly (the losses only use |e| or e^2).
    One deliberate difference: the p-value uses `norm.sf(DM)` instead of `1 - norm.cdf(DM)`, which is
    the same number but does not round to 0 for DM > 8.

    Checked once (2026-11-19) against epftoolbox 1.0 (commit 47d6e06 of 2025-10-25, installed without its
    dependencies in a separate environment; numpy 2.5.4, scipy 1.18.1, Python 3.12.13) on 2,000 random
    error matrices (both norms, 5 to 400 days, heavy-tailed and biased errors, p-values from 0 to 1):
    max absolute p-value difference 1.1e-16. Script: tests/check_dm_against_epftoolbox.py.
    """
    if norm not in (1, 2):
        raise ValueError("norm must be 1 or 2")
    if isinstance(errors_a, pd.DataFrame) and isinstance(errors_b, pd.DataFrame):
        if not errors_a.index.equals(errors_b.index):
            raise ValueError("The two models must be evaluated on the same days")
    errors_a, errors_b = _as_pair(errors_a, errors_b)
    if errors_a.ndim != 2 or errors_a.shape[1] != HORIZON:
        raise ValueError(f"Errors must have shape (days, {HORIZON}), got {errors_a.shape}")
    if not (np.isfinite(errors_a).all() and np.isfinite(errors_b).all()):
        raise ValueError("Errors must be finite")

    d = np.mean(np.abs(errors_a) ** norm, axis=1) - np.mean(np.abs(errors_b) ** norm, axis=1)
    var_d = np.var(d, ddof=0)
    if var_d == 0:
        raise ValueError("Constant loss differential (e.g. identical forecasts): the DM test is undefined")
    dm_stat = np.mean(d) / np.sqrt(var_d / d.size)
    return float(stats.norm.sf(dm_stat))
