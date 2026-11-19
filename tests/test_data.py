import pandas as pd
import pytest

from src.config import DATASET_PATH, TARGET
from src.data import COLUMNS, load_dataset, split_for_day


def test_split_for_day_information_set(toy_df):
    day = pd.Timestamp("2026-04-02")
    history, future_cov = split_for_day(toy_df, day)

    assert history.index[-1] == pd.Timestamp("2026-04-01 23:00")
    assert history.index[0] == toy_df.index[0]
    assert list(history.columns) == COLUMNS
    assert TARGET not in future_cov.columns
    assert list(future_cov.index) == list(pd.date_range(day, periods=24, freq="h"))
    assert list(future_cov["hour"]) == list(range(24))


@pytest.mark.parametrize("day", ["2026-03-30", "2026-04-09", "2026-04-02 12:00"])
def test_split_for_day_rejects_days_without_full_information(toy_df, day):
    # First day (no history), day after the end of the data (no covariates), not a day start
    with pytest.raises(AssertionError):
        split_for_day(toy_df, pd.Timestamp(day))


def test_load_dataset_round_trip(toy_df, tmp_path):
    path = tmp_path / "dataset.parquet"
    toy_df.to_parquet(path)
    df = load_dataset(path)
    pd.testing.assert_frame_equal(df, toy_df, check_freq=False)
    assert df.index.freq == "h"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda df: df.drop(columns="solar_fc"),  # missing column
        lambda df: df.drop(df.index[30]),  # 23 h day
        lambda df: df.iloc[:-1],  # incomplete last day
        lambda df: df.assign(dow=0),  # calendar inconsistent with the index
        lambda df: df.assign(price=df["price"].where(df.index.hour != 5)),  # missing values
        lambda df: df.tz_localize("Europe/Berlin"),  # tz-aware index
    ],
)
def test_load_dataset_rejects_bad_layout(toy_df, tmp_path, corrupt):
    path = tmp_path / "dataset.parquet"
    corrupt(toy_df).to_parquet(path)
    with pytest.raises(ValueError):
        load_dataset(path)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="data/dataset.parquet not downloaded (01_download.ipynb)")
def test_real_dataset_layout():
    df = load_dataset()
    assert len(df) % 24 == 0
