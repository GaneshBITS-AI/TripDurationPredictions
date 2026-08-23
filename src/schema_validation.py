"""
schema_validation.py
---------------------
Week 1 / Module M2: Validate schema before features are engineered.

Checks:
  1. All required columns are present.
  2. GPS pings are present and within a plausible NYC bounding box.
  3. Timestamps are valid (pickup < dropoff, duration within plausible range).
  4. Passenger counts are plausible.

Produces a validation report (dict) and a cleaned DataFrame with bad rows
flagged/removed, so every drop is auditable rather than silent.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame, required_columns: list) -> dict:
    """Checks that all required columns exist. Raises if any are missing."""
    missing = [c for c in required_columns if c not in df.columns]
    report = {"missing_columns": missing, "passed": len(missing) == 0}
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    logger.info("Schema check passed: all %d required columns present.", len(required_columns))
    return report


def _flag_missing_gps(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Returns a boolean mask of rows with missing or out-of-bounds GPS pings."""
    lat_cols = ["pickup_latitude", "dropoff_latitude"]
    lon_cols = ["pickup_longitude", "dropoff_longitude"]

    null_mask = df[lat_cols + lon_cols].isna().any(axis=1)

    in_bounds = (
        df["pickup_latitude"].between(cfg["nyc_lat_min"], cfg["nyc_lat_max"])
        & df["dropoff_latitude"].between(cfg["nyc_lat_min"], cfg["nyc_lat_max"])
        & df["pickup_longitude"].between(cfg["nyc_lon_min"], cfg["nyc_lon_max"])
        & df["dropoff_longitude"].between(cfg["nyc_lon_min"], cfg["nyc_lon_max"])
    )
    out_of_bounds_mask = ~in_bounds.fillna(False)

    return null_mask | out_of_bounds_mask


def _flag_invalid_timestamps(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Returns a boolean mask of rows with invalid pickup/dropoff timestamps or duration."""
    pickup = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    null_ts_mask = pickup.isna() | dropoff.isna()
    order_mask = (dropoff <= pickup).fillna(True)

    duration = df["trip_duration"]
    duration_mask = (duration < cfg["min_trip_duration_sec"]) | (duration > cfg["max_trip_duration_sec"])

    return null_ts_mask | order_mask | duration_mask


def _flag_invalid_passengers(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return (df["passenger_count"] <= 0) | (df["passenger_count"] > cfg["max_passenger_count"])


def validate_and_clean(df: pd.DataFrame, validation_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Runs all row-level validation checks, drops bad rows, and returns
    (clean_df, report) where report captures exactly what was dropped and why
    -- this audit trail matters for a data-versioning step later.
    """
    n_start = len(df)

    missing_gps = _flag_missing_gps(df, validation_cfg)
    bad_timestamps = _flag_invalid_timestamps(df, validation_cfg)
    bad_passengers = _flag_invalid_passengers(df, validation_cfg)

    bad_any = missing_gps | bad_timestamps | bad_passengers

    report = {
        "rows_in": n_start,
        "missing_or_out_of_bounds_gps": int(missing_gps.sum()),
        "invalid_timestamps_or_duration": int(bad_timestamps.sum()),
        "invalid_passenger_count": int(bad_passengers.sum()),
        "total_rows_dropped": int(bad_any.sum()),
        "rows_out": int(n_start - bad_any.sum()),
        "drop_rate_pct": round(100 * bad_any.sum() / n_start, 2) if n_start else 0,
    }

    clean_df = df.loc[~bad_any].reset_index(drop=True)

    logger.info(
        "Validation complete: %d/%d rows dropped (%.2f%%). GPS=%d, timestamps=%d, passengers=%d",
        report["total_rows_dropped"], n_start, report["drop_rate_pct"],
        report["missing_or_out_of_bounds_gps"], report["invalid_timestamps_or_duration"],
        report["invalid_passenger_count"],
    )

    return clean_df, report
