from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import EXOG, HORIZON, QUANTILE_COLUMNS, QUANTILES, TARGET
from src.data import split_for_day
from src.evaluation import RESULT_COLUMNS, run_rolling
from src.foundation import TimesFM3, fm_inputs

CONTEXT = 48  # short context for the 10-day toy dataset
DAY = pd.Timestamp("2026-04-06")


class FakeTimesFM3:
    """Stands in for TimesFM3Forecaster: records the calls, returns quantiles 10 * level + hour."""

    config = SimpleNamespace(quantiles=list(QUANTILES), median_quantile_index=4)

    def __init__(self):
        self.calls = []

    def predict(self, context, horizon, past_future_covariates=None, return_quantiles=False):
        self.calls.append((context, horizon, past_future_covariates, return_quantiles))
        quantiles = 100 * np.asarray(QUANTILES)[None, :] + np.arange(horizon)[:, None]
        return SimpleNamespace(forecast=quantiles[:, 4], quantiles=quantiles)


def test_fm_inputs_shapes_and_alignment(toy_df):
    history, future_cov = split_for_day(toy_df, DAY)
    context, covariates = fm_inputs(history, future_cov, CONTEXT)

    assert context.shape == (CONTEXT,) and context.dtype == np.float32
    assert covariates.shape == (len(EXOG), CONTEXT + HORIZON) and covariates.dtype == np.float32
    # Context = the last prices up to D-1 23:00, in order
    np.testing.assert_array_equal(context, history[TARGET].iloc[-CONTEXT:].to_numpy())
    # Covariates over context + D, one row per EXOG column, aligned on the same hours
    expected = toy_df.loc[DAY - pd.Timedelta(hours=CONTEXT) : DAY + pd.Timedelta(hours=HORIZON - 1), EXOG]
    np.testing.assert_array_equal(covariates, expected.to_numpy().T)


def test_fm_inputs_rejects_short_history(toy_df):
    history, future_cov = split_for_day(toy_df, pd.Timestamp("2026-03-31"))  # 24 h of history
    with pytest.raises(ValueError):
        fm_inputs(history, future_cov, CONTEXT)


@pytest.mark.parametrize("with_covariates", [False, True])
def test_timesfm3_wrapper(toy_df, with_covariates):
    model = FakeTimesFM3()
    y_pred, quantiles = TimesFM3(model, with_covariates, CONTEXT).predict(*split_for_day(toy_df, DAY))

    context, horizon, covariates, return_quantiles = model.calls[0]
    assert context.shape == (CONTEXT,) and horizon == HORIZON and return_quantiles
    assert (covariates is not None) == with_covariates
    if with_covariates:
        assert covariates.shape == (len(EXOG), CONTEXT + HORIZON)
    assert list(quantiles.columns) == QUANTILE_COLUMNS and quantiles.shape == (HORIZON, len(QUANTILES))
    np.testing.assert_array_equal(y_pred, quantiles["q50"])


def test_timesfm3_rejects_other_quantile_levels():
    model = FakeTimesFM3()
    model.config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        TimesFM3(model, with_covariates=False)


def test_timesfm3_in_rolling_loop(toy_df, tmp_path):
    days = pd.date_range("2026-04-06", "2026-04-08", freq="D")
    results = run_rolling(TimesFM3(FakeTimesFM3(), True, CONTEXT), "timesfm_cov", toy_df, days, tmp_path)
    assert list(results.columns) == RESULT_COLUMNS and len(results) == HORIZON * len(days)
    assert (results["y_pred"] == results["q50"]).all()
