"""Dataset loading and split of the information available for a target day D."""

import pandas as pd

from src.config import CALENDAR, COVARIATES, DATASET_PATH, HORIZON, TARGET

COLUMNS = [TARGET, *COVARIATES, *CALENDAR]
ONE_HOUR = pd.Timedelta(hours=1)


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Load data/dataset.parquet (hourly index in Berlin time, 24 h per day).

    Columns: price, load_fc, wind_fc, solar_fc, hour, dow, non_working_day.
    Raises ValueError if the columns, the index or the calendar do not follow this layout.
    """
    df = pd.read_parquet(path)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[COLUMNS]

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None:
        raise ValueError("The index must be a tz-naive DatetimeIndex (Berlin wall-clock time)")
    # Exactly 24 h per day, no gap and no duplicate, from 00:00 of the first day to 23:00 of the last
    expected = pd.date_range(idx[0].normalize(), idx[-1].normalize() + 23 * ONE_HOUR, freq="h")
    if not idx.equals(expected):
        raise ValueError("The index must cover whole days with exactly 24 hourly rows per day")
    if not (df["hour"].to_numpy() == idx.hour).all() or not (df["dow"].to_numpy() == idx.day_of_week).all():
        raise ValueError("Columns hour/dow are inconsistent with the index")
    if df.isna().any().any():
        raise ValueError(f"Missing values: {df.isna().sum()[lambda s: s > 0].to_dict()}")

    return df.asfreq("h")  # attach the hourly frequency (statsmodels expects it)


def split_for_day(df: pd.DataFrame, day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the information available at the forecast origin for target day `day`.

    Returns:
    - history: all columns up to the end of D-1 inclusive (D-1 23:00);
    - future_cov: covariates (forecasts) and calendar for the 24 h of D, without the price column.
    """
    day = pd.Timestamp(day)
    assert day == day.normalize(), f"The target day must be a date at 00:00, got {day}"
    last_hour = day + (HORIZON - 1) * ONE_HOUR

    history = df.loc[: day - ONE_HOUR]
    future_cov = df.loc[day:last_hour].drop(columns=TARGET)

    # Information set at the origin (CLAUDE.md section 4)
    assert len(history) > 0 and history.index[-1] == day - ONE_HOUR, (
        f"history must end at D-1 23:00 ({day - ONE_HOUR}), ends at {history.index[-1] if len(history) else None}"
    )
    assert TARGET not in future_cov.columns, "future_cov must not contain the price of D"
    assert len(future_cov) == HORIZON and future_cov.index[0] == day and future_cov.index[-1] == last_hour, (
        f"future_cov must contain the {HORIZON} hours of {day.date()}, got {len(future_cov)} rows"
    )
    return history, future_cov
