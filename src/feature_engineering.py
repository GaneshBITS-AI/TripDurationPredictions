"""
feature_engineering.py
------------------------
Week 1 / Module M2: Engineer time- and location-based features.

IMPORTANT (leakage): row-wise features (hour_of_day, day_of_week,
trip_distance_km, etc.) are pure functions of a single row and are safe to
compute at any point. The location bin features are NOT row-wise -- pd.cut
derives bin edges from the data's own min/max, so fitting them on
train+test combined would leak test-set distribution info into training
features. To avoid this:

    1. Split raw data into train/test FIRST (see src/data_split.py).
    2. fit_location_bins(train_df) to get bin edges from TRAIN ONLY.
    3. apply_location_bins(df, edges) on both train and test using the
       SAME fitted edges.

engineer_features() below takes an optional `location_bin_edges` argument
so this fit-on-train/apply-to-both pattern is enforced by the function
signature, not just a comment.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise -- safe to compute before or after a train/test split."""
    df = df.copy()
    pickup = pd.to_datetime(df["pickup_datetime"])
    df["hour_of_day"] = pickup.dt.hour
    df["day_of_week"] = pickup.dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = pickup.dt.month
    df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (haversine between two points in the SAME row) -- safe either way."""
    df = df.copy()
    df["trip_distance_km"] = _haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    return df


def fit_location_bins(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """
    Computes grid bin edges for pickup/dropoff lat/lon from `df`.

    Call this on the TRAINING split only. Reuse the returned edges (via
    apply_location_bins) on the test split so both sets are binned on the
    exact same grid -- fitting separately on test would both leak test
    distribution info and produce bins that aren't comparable to train's.
    """
    edges = {}
    for prefix in ["pickup", "dropoff"]:
        for axis, col in [("lat", f"{prefix}_latitude"), ("lon", f"{prefix}_longitude")]:
            _, bin_edges = pd.cut(df[col], bins=n_bins, retbins=True)
            edges[f"{prefix}_{axis}"] = bin_edges.tolist()
    return edges


def apply_location_bins(df: pd.DataFrame, edges: dict) -> pd.DataFrame:
    """Bins pickup/dropoff coordinates using pre-fitted edges (see fit_location_bins)."""
    df = df.copy()
    for prefix in ["pickup", "dropoff"]:
        lat_col, lon_col = f"{prefix}_latitude", f"{prefix}_longitude"
        lat_edges = edges[f"{prefix}_lat"]
        lon_edges = edges[f"{prefix}_lon"]
        df[f"{prefix}_lat_bin"] = pd.cut(
            df[lat_col], bins=lat_edges, labels=False, include_lowest=True
        )
        df[f"{prefix}_lon_bin"] = pd.cut(
            df[lon_col], bins=lon_edges, labels=False, include_lowest=True
        )
        # Values outside the train-fitted range (can happen on test) fall
        # outside all bins and come back as NaN from pd.cut -- clip them
        # into the nearest edge bin instead of silently losing rows.
        lat_bin = df[f"{prefix}_lat_bin"]
        n_lat_bins = len(lat_edges) - 1
        lat_bin = lat_bin.where(~lat_bin.isna(), np.where(df[lat_col] < lat_edges[0], 0, n_lat_bins - 1))
        df[f"{prefix}_lat_bin"] = lat_bin.astype(int)

        lon_bin = df[f"{prefix}_lon_bin"]
        n_lon_bins = len(lon_edges) - 1
        lon_bin = lon_bin.where(~lon_bin.isna(), np.where(df[lon_col] < lon_edges[0], 0, n_lon_bins - 1))
        df[f"{prefix}_lon_bin"] = lon_bin.astype(int)
    return df


def engineer_features(df: pd.DataFrame, location_bin_edges: dict = None, n_bins: int = 10):
    """
    Runs the full Week 1 feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
    location_bin_edges : dict or None
        If None, bin edges are FIT on `df` (only correct when `df` is the
        training split). If provided, `df` is transformed using those
        pre-fitted edges (correct usage for the test split).
    n_bins : int
        Only used when fitting (location_bin_edges is None).

    Returns
    -------
    (features_df, location_bin_edges) : tuple[pd.DataFrame, dict]
        Always returns the edges used, so callers can pass them on to the
        next split.
    """
    df = add_time_features(df)
    df = add_distance_features(df)

    if location_bin_edges is None:
        location_bin_edges = fit_location_bins(df, n_bins=n_bins)
        logger.info("Fitted new location bin edges from this dataframe (expected: train split only).")
    else:
        logger.info("Applying pre-fitted location bin edges (expected: test split).")

    df = apply_location_bins(df, location_bin_edges)
    logger.info("Feature engineering complete. Columns now: %s", list(df.columns))
    return df, location_bin_edges
