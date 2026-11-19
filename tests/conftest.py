import numpy as np
import pandas as pd
import pytest

from src.config import TARGET


def make_toy(start="2026-03-30", n_days=10) -> pd.DataFrame:
    """Synthetic dataset with the layout of data/dataset.parquet.

    The price of day d (0, 1, ...) at hour h is 100 * d + h, so a naive D-1 forecast is off by exactly
    100 EUR/MWh at every hour and a naive D-7 by exactly 700.
    """
    idx = pd.date_range(start, periods=24 * n_days, freq="h", name="datetime")
    day_number = np.repeat(np.arange(n_days), 24)
    return pd.DataFrame(
        {
            TARGET: 100.0 * day_number + idx.hour,
            "load_fc": 50.0,
            "wind_fc": 20.0,
            "solar_fc": 10.0,
            "hour": idx.hour,
            "dow": idx.day_of_week,
            "non_working_day": (idx.day_of_week >= 5).astype(int),
        },
        index=idx,
    )


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return make_toy()
