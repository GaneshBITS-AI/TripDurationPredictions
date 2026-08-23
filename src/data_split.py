"""
data_split.py
---------------
Week 1 / Module M2: Split cleaned raw trips into train/test BEFORE any
data-derived feature engineering (e.g., location bin edges) is fit.

Why split this early rather than after feature engineering:
  - Row-wise features (hour_of_day, trip_distance_km, ...) don't care when
    you split -- they only look at their own row.
  - Data-derived features (location bins, and later any scaler/encoder/
    target-encoding in Week 2) DO care: fitting them on the full dataset
    lets statistics from the test set leak into training. Splitting first
    and fitting only on train (then applying to test) avoids this.

A time-based split is used by default (train = earlier trips, test = later
trips) rather than a random split, because ETA prediction is a forecasting-
style problem: at serving time the model only ever sees trips that happen
after training data was collected. A random split would let the model
"see the future" during training in a way it never could in production.
Random splitting is also supported for cases where that's an explicit
course requirement.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split as _sk_train_test_split

logger = logging.getLogger(__name__)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    method: str = "time",
    time_col: str = "pickup_datetime",
    random_state: int = 42,
) -> tuple:
    """
    Splits `df` into (train_df, test_df).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned (post-validation) trip data, BEFORE feature engineering.
    test_size : float
        Fraction of rows held out for testing.
    method : {"time", "random"}
        "time"   -- sort by `time_col`, take the earliest (1 - test_size)
                    as train and the latest test_size as test. Recommended
                    for ETA/forecasting problems.
        "random" -- standard random split (sklearn train_test_split).
    time_col : str
        Column to sort on for a time-based split.
    random_state : int
        Seed for reproducibility (random split, or tie-breaking).

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    if method == "time":
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
        test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
        logger.info(
            "Time-based split: train=%d rows (up to %s), test=%d rows (from %s)",
            len(train_df), train_df[time_col].max(), len(test_df), test_df[time_col].min(),
        )
    elif method == "random":
        train_df, test_df = _sk_train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        logger.info("Random split: train=%d rows, test=%d rows", len(train_df), len(test_df))
    else:
        raise ValueError(f"Unknown split method: {method!r}. Use 'time' or 'random'.")

    return train_df, test_df
