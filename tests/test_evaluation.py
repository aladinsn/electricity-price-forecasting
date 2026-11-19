import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import QUANTILE_COLUMNS, QUANTILES, TARGET
from src.evaluation import RESULT_COLUMNS, daily_errors, diebold_mariano, mae, pinball, rmae, rmse, run_rolling

DAYS = pd.date_range("2026-04-06", "2026-04-08", freq="D")  # Monday to Wednesday


class NaiveD1:
    """Dummy model: price at the same hour the previous day."""

    def __init__(self):
        self.calls = []

    def predict(self, history, future_cov):
        self.calls.append((history.index[-1], list(future_cov.columns)))
        return history[TARGET].iloc[-24:].to_numpy(), None


class ConstantQuantiles:
    """Dummy probabilistic model: quantile qXX of every hour equals XX (10, 20, ..., 90)."""

    def predict(self, history, future_cov):
        quantiles = pd.DataFrame([[100 * q for q in QUANTILES]] * 24, columns=QUANTILE_COLUMNS)
        return quantiles["q50"].to_numpy(), quantiles


# Metrics, checked by hand on y_true = [1, 2, 3, 4]


def test_mae_rmse_rmae_by_hand():
    y_true, y_pred, y_naive = [1, 2, 3, 4], [2, 2, 5, 0], [1, 1, 1, 1]
    # |errors| = [1, 0, 2, 4], squared = [1, 0, 4, 16], naive |errors| = [0, 1, 2, 3]
    assert mae(y_true, y_pred) == pytest.approx(7 / 4)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(21 / 4))
    assert rmae(y_true, y_pred, y_naive) == pytest.approx((7 / 4) / (6 / 4))


def test_metrics_ignore_pandas_index():
    # Positional comparison: misaligned indexes must not trigger pandas alignment
    y_true = pd.Series([1.0, 2.0], index=[0, 1])
    y_pred = pd.Series([1.0, 2.0], index=[5, 6])
    assert mae(y_true, y_pred) == 0


def test_pinball_by_hand():
    # Quantiles q10 ... q90 = 1 ... 9, y = 5: loss per level tau is
    # tau = 0.1 ... 0.4: tau * (5 - 10 tau) = 0.4, 0.6, 0.6, 0.4; tau = 0.5: 0;
    # tau = 0.6 ... 0.9: (1 - tau) * (10 tau - 5) = 0.4, 0.6, 0.6, 0.4. Sum 4, mean over 9 levels 4/9.
    quantiles = pd.DataFrame([[10 * q for q in QUANTILES]], columns=QUANTILE_COLUMNS)
    assert pinball([5.0], quantiles) == pytest.approx(4 / 9)
    # Perfect quantiles: zero loss; all quantiles 1 above y: mean of (1 - tau) = 0.5
    assert pinball([3.0], quantiles * 0 + 3.0) == 0
    assert pinball([3.0], quantiles * 0 + 4.0) == pytest.approx(0.5)


def test_pinball_rejects_missing_quantiles():
    quantiles = pd.DataFrame([[np.nan] * len(QUANTILES)], columns=QUANTILE_COLUMNS)
    with pytest.raises(ValueError):
        pinball([1.0], quantiles)


# Rolling loop


def test_run_rolling_naive_end_to_end(toy_df, tmp_path):
    model = NaiveD1()
    results = run_rolling(model, "naive_d1", toy_df, DAYS, results_dir=tmp_path)

    assert list(results.columns) == RESULT_COLUMNS
    assert len(results) == 24 * len(DAYS)
    assert (results["model"] == "naive_d1").all()
    assert list(results["target_date"].unique()) == list(DAYS)
    assert list(results["hour"]) == list(range(24)) * len(DAYS)
    assert results[QUANTILE_COLUMNS].isna().all().all()
    # y_true is the actual price of D: day 7 of the toy data (2026-04-06) costs 700 + h
    first_day = results[results["target_date"] == DAYS[0]]
    assert list(first_day["y_true"]) == [700.0 + h for h in range(24)]
    # Naive D-1 is off by exactly 100 at every hour
    assert mae(results["y_true"], results["y_pred"]) == pytest.approx(100)
    assert rmse(results["y_true"], results["y_pred"]) == pytest.approx(100)
    # The model only saw prices up to D-1 23:00 and no price for D
    assert [last for last, _ in model.calls] == list(DAYS - pd.Timedelta(hours=1))
    assert all(TARGET not in columns for _, columns in model.calls)
    assert (tmp_path / "naive_d1.parquet").exists()


def test_run_rolling_reads_cache_instead_of_recomputing(toy_df, tmp_path):
    first = run_rolling(NaiveD1(), "naive_d1", toy_df, DAYS, results_dir=tmp_path)
    model = NaiveD1()
    second = run_rolling(model, "naive_d1", toy_df, DAYS, results_dir=tmp_path)
    assert model.calls == []
    pd.testing.assert_frame_equal(first, second)
    # A cached file covering other days is not silently reused
    with pytest.raises(ValueError):
        run_rolling(model, "naive_d1", toy_df, DAYS[:2], results_dir=tmp_path)


def test_run_rolling_stores_quantiles(toy_df, tmp_path):
    results = run_rolling(ConstantQuantiles(), "quantiles", toy_df, DAYS, results_dir=tmp_path)
    assert list(results.loc[0, QUANTILE_COLUMNS]) == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert (results["y_pred"] == results["q50"]).all()
    assert pinball(results["y_true"], results[QUANTILE_COLUMNS]) > 0


@pytest.mark.parametrize("bad_output", [np.zeros(23), np.full(24, np.nan)])
def test_run_rolling_rejects_bad_forecasts(toy_df, tmp_path, bad_output):
    class Bad:
        def predict(self, history, future_cov):
            return bad_output, None

    with pytest.raises(ValueError):
        run_rolling(Bad(), "bad", toy_df, DAYS, results_dir=tmp_path)
    assert not (tmp_path / "bad.parquet").exists()


def test_run_rolling_requires_increasing_days(toy_df, tmp_path):
    with pytest.raises(ValueError):
        run_rolling(NaiveD1(), "naive_d1", toy_df, DAYS[::-1], results_dir=tmp_path)


# Diebold-Mariano test (checked against epftoolbox once, see tests/check_dm_against_epftoolbox.py)


def test_diebold_mariano_by_hand():
    # Model B is perfect; model A has |error| 1, 2, 3 on days 1, 2, 3 (signs must not matter).
    errors_a = np.array([[-1.0] * 24, [2.0] * 24, [-3.0] * 24])
    errors_b = np.zeros((3, 24))
    # norm 1: d = [1, 2, 3], mean 2, population variance 2/3, DM = 2 / sqrt(2/9) = 3 sqrt(2)
    assert diebold_mariano(errors_a, errors_b, norm=1) == pytest.approx(stats.norm.sf(3 * np.sqrt(2)))
    # norm 2: d = [1, 4, 9], mean 14/3, population variance 98/9, DM = (14/3) / sqrt(98/27) = sqrt(6)
    assert diebold_mariano(errors_a, errors_b, norm=2) == pytest.approx(stats.norm.sf(np.sqrt(6)))


def test_diebold_mariano_direction():
    rng = np.random.default_rng(0)
    errors_good, errors_bad = rng.normal(0, 1, (100, 24)), rng.normal(0, 3, (100, 24))
    # H1 "B more accurate than A": rejected when B is the good model, not the other way round
    assert diebold_mariano(errors_bad, errors_good) < 0.01
    assert diebold_mariano(errors_good, errors_bad) > 0.99
    # One-sided p-values of the two directions sum to 1
    assert diebold_mariano(errors_bad, errors_good) + diebold_mariano(errors_good, errors_bad) == pytest.approx(1)


def test_diebold_mariano_rejects_bad_inputs():
    errors = np.random.default_rng(0).normal(size=(10, 24))
    with pytest.raises(ValueError):
        diebold_mariano(errors, errors[:9])  # different number of days
    with pytest.raises(ValueError):
        diebold_mariano(errors.ravel(), errors.ravel())  # not (days, 24)
    with pytest.raises(ValueError):
        diebold_mariano(errors, errors)  # identical forecasts: zero variance
    with pytest.raises(ValueError):
        diebold_mariano(errors, errors * 2, norm=3)
    a = pd.DataFrame(errors, index=pd.date_range("2026-04-01", periods=10))
    with pytest.raises(ValueError):
        diebold_mariano(a, a.set_axis(a.index + pd.Timedelta(days=1)) * 2)  # different days


def test_daily_errors_from_rolling_results(toy_df, tmp_path):
    results = run_rolling(NaiveD1(), "naive_d1", toy_df, DAYS, results_dir=tmp_path)
    errors = daily_errors(results.sample(frac=1, random_state=0))  # row order must not matter
    assert errors.shape == (len(DAYS), 24)
    assert list(errors.index) == list(DAYS)
    assert list(errors.columns) == list(range(24))
    assert (errors == 100).all().all()  # naive D-1 is 100 below the actual price at every hour
    with pytest.raises(ValueError):
        daily_errors(pd.concat([results, results.assign(model="other")]))
    with pytest.raises(ValueError):
        daily_errors(results.iloc[:-1])  # one hour missing
