"""Rolling evaluation loop shared by all models, and metrics."""

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

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


def diebold_mariano(errors_a: pd.DataFrame, errors_b: pd.DataFrame, norm: int = 1) -> float:
    """One-sided multivariate Diebold-Mariano test (H1: model B is more accurate than model A).

    Errors are (days, 24) arrays; the loss differential is computed on the daily vector of
    24 errors. Reimplementation of `epftoolbox.evaluation.DM` (Apache-2.0, Lago et al. 2021).
    Returns the p-value.
    """
    raise NotImplementedError
