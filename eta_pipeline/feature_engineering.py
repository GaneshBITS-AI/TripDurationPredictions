"""
Feature Engineering
====================

Transforms clean raw trips into a rich feature set:

Time-based
----------
    * hour_of_day    (0-23)
    * day_of_week    (0=Mon ... 6=Sun)
    * month          (1-12)
    * is_weekend     (bool -> int)
    * is_rush_hour   (7-9 AM, 4-7 PM weekdays)
    * part_of_day    (morning / afternoon / evening / night)

Location-based (derived from raw lat/lon)
------------------------------------------
    * haversine_km   - great-circle distance (cross-check vs. GPS distance)
    * bearing_deg    - compass direction pickup -> dropoff
    * pickup_grid_x / pickup_grid_y   - 500 m grid cell indices (sparse area proxy)
    * dropoff_grid_x / dropoff_grid_y

Interaction
-----------
    * log_haversine_km   - log-scaled haversine distance (reduces right skew)
    * store_and_fwd_flag encoding (Y/N -> 1/0)

Design
------
- Functional, stateless transformations.
- A single `build_features(df)` entry point returns the augmented DataFrame.
- Input integrity is validated before any operation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eta_pipeline import config
from eta_pipeline.logger import get_logger

log = get_logger(__name__)

# Grid cell size in degrees (~ 500 m at NYC latitude)
GRID_CELL_DEG = 0.005


# ------------------------------------------------------------------ Validators --
def _assert_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature engineering: missing columns {missing}")


# --------------------------------------------------------------- Time features --
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from pickup_datetime."""
    _assert_columns(df, ["pickup_datetime"])

    dt = df["pickup_datetime"]

    df = df.copy()
    df["hour_of_day"] = dt.dt.hour.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")  # 0=Mon, 6=Sun
    df["month"]       = dt.dt.month.astype("int8")
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype("int8")

    # Rush hour: 7-9 AM or 16-19 on weekdays
    rush = (
        (df["is_weekend"] == 0)
        & (
            df["hour_of_day"].between(7, 9)
            | df["hour_of_day"].between(16, 19)
        )
    )
    df["is_rush_hour"] = rush.astype("int8")

    # Part-of-day ordinal
    def _part_of_day(h: pd.Series) -> pd.Series:
        conditions = [
            h.between(5, 11),
            h.between(12, 16),
            h.between(17, 20),
        ]
        choices = ["morning", "afternoon", "evening"]
        return np.select(conditions, choices, default="night")

    df["part_of_day"] = _part_of_day(df["hour_of_day"])

    # Cyclical encoding of hour and day (preserves periodicity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24).round(6)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7).round(6)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7).round(6)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12).round(6)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12).round(6)

    log.debug("Time features added: hour_of_day, day_of_week, month, is_weekend, is_rush_hour, cyclicals.")
    return df


# ----------------------------------------------------------- Location features --
def _haversine(lat1: np.ndarray, lon1: np.ndarray,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine formula -> kilometres."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))


def _bearing(lat1: np.ndarray, lon1: np.ndarray,
             lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Forward azimuth (bearing) in degrees [0, 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geospatial features derived from lat/lon coordinates."""
    _assert_columns(df, ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"])

    df = df.copy()
    p_lat = df["pickup_lat"].to_numpy()
    p_lon = df["pickup_lon"].to_numpy()
    d_lat = df["dropoff_lat"].to_numpy()
    d_lon = df["dropoff_lon"].to_numpy()

    df["haversine_km"] = _haversine(p_lat, p_lon, d_lat, d_lon).round(4)
    df["bearing_deg"]  = _bearing(p_lat, p_lon, d_lat, d_lon).round(2)

    # Bearing cyclical encoding
    df["bearing_sin"] = np.sin(np.radians(df["bearing_deg"])).round(6)
    df["bearing_cos"] = np.cos(np.radians(df["bearing_deg"])).round(6)

    # Grid-cell indices (proxy for zone/neighbourhood)
    REF_LAT, REF_LON = 40.50, -74.05  # NYC bounding box origin
    df["pickup_grid_x"]  = ((p_lat - REF_LAT) / GRID_CELL_DEG).astype(int)
    df["pickup_grid_y"]  = ((p_lon - REF_LON) / GRID_CELL_DEG).astype(int)
    df["dropoff_grid_x"] = ((d_lat - REF_LAT) / GRID_CELL_DEG).astype(int)
    df["dropoff_grid_y"] = ((d_lon - REF_LON) / GRID_CELL_DEG).astype(int)

    # Absolute lat/lon deltas
    df["delta_lat"] = (d_lat - p_lat).round(6)
    df["delta_lon"] = (d_lon - p_lon).round(6)

    log.debug("Location features added: haversine_km, bearing_deg, grid cells, deltas.")
    return df


# ------------------------------------------------------ Distance / interaction features --
def add_distance_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived distance and interaction features."""
    _assert_columns(df, ["haversine_km"])

    df = df.copy()
    df["log_haversine_km"] = np.log1p(df["haversine_km"]).round(6)

    log.debug("Distance/interaction features added.")
    return df


# --------------------------------------------------------------- Weather features --
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary weather condition flags from the merged weather columns.

    Expects columns already joined by merge_weather_with_trips():
        avg_temp_f, precipitation, snow_depth

    Adds:
        is_snowing  (snow_depth > 0)
        is_raining  (precipitation > 0.1 inches, not snowing)
    """
    df = df.copy()

    if "snow_depth" in df.columns:
        df["is_snowing"] = (df["snow_depth"] > 0).astype("int8")
    else:
        df["is_snowing"] = 0

    if "precipitation" in df.columns:
        df["is_raining"] = (
            (df["precipitation"] > 0.1) & (df.get("is_snowing", 0) == 0)
        ).astype("int8")
    else:
        df["is_raining"] = 0

    # Ensure weather numeric columns are float32 and present
    for col in ["avg_temp_f", "precipitation", "snow_depth"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].astype("float32")

    log.debug("Weather features added: is_snowing, is_raining.")
    return df


# ----------------------------------------------------------------- Orchestrator --
def encode_store_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Encode store_and_fwd_flag Y/N -> 1/0 if the column is present."""
    if "store_and_fwd_flag" in df.columns:
        df = df.copy()
        df["store_and_fwd_flag"] = (
            df["store_and_fwd_flag"].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
        ).fillna(0).astype("int8")
        log.debug("store_and_fwd_flag encoded Y-1, N-0.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a single split DataFrame.

    IMPORTANT: call this AFTER the train/test split so that no information
    from the test set leaks into training-time feature construction.

    Parameters
    ----------
    df : a single split (train, val, or test) DataFrame from the ingestion module

    Returns
    -------
    Augmented DataFrame with all engineered features.
    """
    log.info("=" * 60)
    log.info("STARTING FEATURE ENGINEERING")
    log.info("=" * 60)
    initial_cols = set(df.columns)

    df = add_time_features(df)
    df = add_location_features(df)
    df = add_distance_interaction_features(df)
    df = encode_store_flag(df)
    df = add_weather_features(df)

    new_cols = sorted(set(df.columns) - initial_cols)
    log.info(f"Feature engineering complete. {len(new_cols)} new features added.")
    for c in new_cols:
        log.info(f"  + {c}")

    return df