"""
NYC Taxi Trip Duration — Data Ingestion Pipeline
=================================================

Loads the raw Kaggle NYC Taxi trip dataset and the NYC daily weather dataset,
adapts/renames columns to the pipeline's internal schema, merges weather
features onto each trip, validates schema and row-level data quality, and
splits the result into "clean" and "quarantine" DataFrames.

Typical usage
-------------
    from ingestion import (
        load_nyc_kaggle_data,
        load_weather_data,
        merge_weather_with_trips,
        validate_and_ingest,
    )

    trips_df   = load_nyc_kaggle_data()
    weather_df = load_weather_data()
    trips_df   = merge_weather_with_trips(trips_df, weather_df)
    clean_df, quarantine_df, report = validate_and_ingest(trips_df)

NOTE: The imports/header block below and a few lines inside the schema
validators were not visible in the source photos and have been
reconstructed to match the style of the rest of the file. Search for
"RECONSTRUCTED" to find those spots and adjust if your real `config`
module differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from eta_pipeline import config  # RECONSTRUCTED: local project config module
from eta_pipeline.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------- Validation report ----------------------------------------------------------------

@dataclass
class ValidationReport:
    total_rows:             int       = 0
    missing_gps_rows:       int       = 0
    invalid_timestamp_rows: int       = 0
    out_of_range_rows:      int       = 0
    negative_eta_rows:      int       = 0
    invalid_passenger_rows: int       = 0
    duplicate_trip_ids:     int       = 0
    clean_rows:              int      = 0
    quarantine_rows:         int      = 0
    schema_errors:           List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.clean_rows / self.total_rows if self.total_rows else 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def log_summary(self) -> None:
        log.info("=" * 60)
        log.info("VALIDATION REPORT")
        log.info("=" * 60)
        log.info(f"  Total rows              : {self.total_rows:>10,}")
        log.info(f"  Missing GPS rows        : {self.missing_gps_rows:>10,}")
        log.info(f"  Invalid timestamp rows  : {self.invalid_timestamp_rows:>10,}")
        log.info(f"  Out-of-range rows       : {self.out_of_range_rows:>10,}")
        log.info(f"  Invalid ETA rows        : {self.negative_eta_rows:>10,}")
        log.info(f"  Invalid passenger rows  : {self.invalid_passenger_rows:>10,}")
        log.info(f"  Duplicate trip IDs      : {self.duplicate_trip_ids:>10,}")
        log.info(f"  Quarantined rows        : {self.quarantine_rows:>10,}")
        log.info(f"  Clean rows              : {self.clean_rows:>10,}")
        log.info(f"  Pass rate               : {self.pass_rate:>10.2%}")
        if self.schema_errors:
            log.warning("  Schema errors:")
            for err in self.schema_errors:
                log.warning(f"    * {err}")
        log.info("=" * 60)


# ---------------------------------------------------------------- Kaggle loader & adapter ----------------------------------------------------------------

def load_nyc_kaggle_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the NYC Taxi trip CSV and adapt it to the pipeline's internal schema.

    Tries data/raw/NYC.csv first, falls back to data/raw/train.csv (Kaggle).

    Parameters
    ----------
    path : explicit path override. Defaults to NYC.csv then train.csv.

    Returns
    -------
    DataFrame with pipeline column names and derived eta_minutes column.
    """
    if path is None:
        # Prefer NYC.csv; fall back to Kaggle train.csv
        nyc_path = config.RAW_DIR / config.NYC_RAW_FILENAME
        kaggle_path = config.RAW_DIR / config.KAGGLE_RAW_FILENAME
        path = nyc_path if nyc_path.exists() else kaggle_path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at: {path}\n"
            "Download it from https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration\n"
            f"and place train.csv in:  {config.RAW_DIR}"
        )

    log.info(f"Loading Kaggle NYC Taxi dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Raw dataset shape: {df.shape}")
    log.info(f"Raw columns: {df.columns.tolist()}")

    # --- Rename Kaggle columns → pipeline names ---
    present_map = {k: v for k, v in config.KAGGLE_COLUMN_MAP.items() if k in df.columns}
    missing_raw = [k for k in config.KAGGLE_COLUMN_MAP if k not in df.columns]
    if missing_raw:
        log.warning(f"Raw Kaggle columns not found (will be skipped): {missing_raw}")

    df = df.rename(columns=present_map)

    # --- Parse datetimes ---
    for col in ["pickup_datetime", "dropoff_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Derive target: eta_minutes from trip_duration_sec ---
    if "trip_duration_sec" in df.columns:
        df["eta_minutes"] = (df["trip_duration_sec"] / 60.0).round(4)
        log.info("Derived eta_minutes from trip_duration_sec.")
    else:
        log.warning("trip_duration_sec not found — eta_minutes must already exist.")

    # --- Coerce numeric types ---
    for col in ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "eta_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["passenger_count", "vendor_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    log.info(f"Post-mapping shape: {df.shape}")
    return df


# ---------------------------------------------------------------- Weather loader & merger ----------------------------------------------------------------

def load_weather_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the NYC daily weather CSV and normalise it for merging.

    Source columns:
      date, maximum temperature, minimum temperature, average temperature,
      precipitation, snow fall, snow depth

    Returns a DataFrame indexed by `date` (date only, no time) with
    normalised column names ready for left-joining to the trip DataFrame.
    """
    if path is None:
        path = config.RAW_DIR / config.WEATHER_RAW_FILENAME

    path = Path(path)
    if not path.exists():
        log.warning(
            f"Weather file not found at: {path} — trips will proceed without weather features."
        )
        return pd.DataFrame()

    log.info(f"Loading weather data from: {path}")
    wx = pd.read_csv(path, low_memory=False)
    log.info(f"Raw weather shape: {wx.shape}  columns: {wx.columns.tolist()}")

    # Normalise column names (strip spaces, lower-case)
    wx.columns = [c.strip().lower() for c in wx.columns]

    # Parse date — format is D-M-YYYY (e.g. 1-1-2016)
    wx["date"] = pd.to_datetime(wx["date"], format="%d-%m-%Y", errors="coerce")
    wx = wx.dropna(subset=["date"])
    wx["date"] = wx["date"].dt.normalize()  # keep date only (no time component)

    # Rename to pipeline-internal names
    rename_map = {
        "average temperature": "avg_temp_f",
        "maximum temperature": "max_temp_f",
        "minimum temperature": "min_temp_f",
        "precipitation":       "precipitation",
        "snow fall":           "snow_fall",
        "snow depth":          "snow_depth",
    }
    wx = wx.rename(columns={k: v for k, v in rename_map.items() if k in wx.columns})

    # Coerce numeric
    for col in ["avg_temp_f", "max_temp_f", "min_temp_f", "precipitation", "snow_fall", "snow_depth"]:
        if col in wx.columns:
            wx[col] = pd.to_numeric(wx[col], errors="coerce")

    wx = wx.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    log.info(f"Weather data ready: {len(wx):,} days  ({wx['date'].min().date()} to {wx['date'].max().date()})")
    return wx


def merge_weather_with_trips(
    trips_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join daily weather onto the trips DataFrame via pickup date.

    Adds columns: avg_temp_f, precipitation, snow_depth
    Missing weather rows (no matching date) are filled with column medians.

    Parameters
    ----------
    trips_df   : trip DataFrame after load_nyc_kaggle_data()
    weather_df : weather DataFrame after load_weather_data()

    Returns
    -------
    trips_df enriched with weather columns.
    """
    if weather_df.empty:
        log.warning("Weather DataFrame is empty — skipping merge. Weather features will be 0.")
        for col in ["avg_temp_f", "precipitation", "snow_depth"]:
            trips_df[col] = 0.0
        return trips_df

    log.info("Merging weather data with trips on pickup date …")
    trips_df = trips_df.copy()

    # Create a date-only join key from pickup_datetime
    trips_df["_pickup_date"] = pd.to_datetime(
        trips_df["pickup_datetime"], errors="coerce"
    ).dt.normalize()

    weather_keep = ["date"] + [
        c for c in ["avg_temp_f", "precipitation", "snow_depth", "snow_fall"]
        if c in weather_df.columns
    ]
    wx_slim = weather_df[weather_keep].rename(columns={"date": "_pickup_date"})

    trips_df = trips_df.merge(wx_slim, on="_pickup_date", how="left")
    trips_df = trips_df.drop(columns=["_pickup_date"])

    # Fill NaNs: temperature → median of weather table; precip/snow → 0 (missing = none)
    for col in ["avg_temp_f", "max_temp_f", "min_temp_f"]:
        if col in trips_df.columns:
            fill_val = weather_df[col].median() if col in weather_df.columns else 55.0
            trips_df[col] = trips_df[col].fillna(fill_val)
    for col in ["precipitation", "snow_depth", "snow_fall"]:
        if col in trips_df.columns:
            trips_df[col] = trips_df[col].fillna(0.0)

    matched = trips_df["avg_temp_f"].notna().sum() if "avg_temp_f" in trips_df.columns else 0
    log.info(f"Weather merge complete: {matched:,} trips enriched with daily weather.")
    return trips_df


# ---------------------------------------------------------------- Schema validators ----------------------------------------------------------------

def _validate_schema(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Missing required columns after mapping: {missing_cols}"
        report.schema_errors.append(msg)
        log.error(msg)
    return df


def _flag_missing_gps(df: pd.DataFrame, report: ValidationReport) -> pd.Series:
    gps_cols = ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"]
    mask = df[gps_cols].isna().any(axis=1)
    report.missing_gps_rows = int(mask.sum())
    if report.missing_gps_rows:
        log.warning(f"Missing GPS pings in {report.missing_gps_rows:,} rows.")
    return mask


def _flag_invalid_timestamps(df: pd.DataFrame, report: ValidationReport) -> pd.Series:
    nat_mask   = df["pickup_datetime"].isna() | df["dropoff_datetime"].isna()
    order_mask = ~nat_mask & (df["dropoff_datetime"] <= df["pickup_datetime"])
    mask = nat_mask | order_mask
    report.invalid_timestamp_rows = int(mask.sum())
    if report.invalid_timestamp_rows:
        log.warning(f"Invalid timestamps in {report.invalid_timestamp_rows:,} rows.")
    return mask


def _flag_out_of_range(df: pd.DataFrame, report: ValidationReport) -> pd.Series:
    lat_ok = (
        df["pickup_lat"].between(*config.LAT_RANGE)
        & df["dropoff_lat"].between(*config.LAT_RANGE)
    )
    # RECONSTRUCTED: longitude check not visible in source photo, added to mirror lat_ok
    lon_ok = (
        df["pickup_lon"].between(*config.LON_RANGE)
        & df["dropoff_lon"].between(*config.LON_RANGE)
    )
    mask = ~(lat_ok & lon_ok)
    report.out_of_range_rows = int(mask.sum())
    if report.out_of_range_rows:
        log.warning(f"Out-of-range GPS coordinates in {report.out_of_range_rows:,} rows.")
    return mask


# RECONSTRUCTED: the following three flag functions were not visible in any
# photo (the source jumps from _flag_out_of_range at line ~303 to the public
# API at line 341). They are written to match the calling signature used in
# validate_and_ingest() and the style of the functions above — check them
# against your real source.

def _flag_invalid_eta(df: pd.DataFrame, report: ValidationReport) -> pd.Series:
    mask = df["eta_minutes"].isna() | (df["eta_minutes"] <= 0)
    if hasattr(config, "MAX_ETA_MINUTES"):
        mask = mask | (df["eta_minutes"] > config.MAX_ETA_MINUTES)
    report.negative_eta_rows = int(mask.sum())
    if report.negative_eta_rows:
        log.warning(f"Invalid ETA values in {report.negative_eta_rows:,} rows.")
    return mask


def _flag_invalid_passengers(df: pd.DataFrame, report: ValidationReport) -> pd.Series:
    mask = df["passenger_count"].isna() | (df["passenger_count"] <= 0)
    if hasattr(config, "MAX_PASSENGERS"):
        mask = mask | (df["passenger_count"] > config.MAX_PASSENGERS)
    report.invalid_passenger_rows = int(mask.sum())
    if report.invalid_passenger_rows:
        log.warning(f"Invalid passenger counts in {report.invalid_passenger_rows:,} rows.")
    return mask


def _flag_duplicate_trip_ids(df: pd.DataFrame, report: ValidationReport) -> pd.Series:
    if "trip_id" in df.columns:
        mask = df["trip_id"].duplicated(keep="first")
    else:
        mask = pd.Series(False, index=df.index)
    report.duplicate_trip_ids = int(mask.sum())
    if report.duplicate_trip_ids:
        log.warning(f"Duplicate trip IDs in {report.duplicate_trip_ids:,} rows.")
    return mask


# ---------------------------------------------------------------- Public API ----------------------------------------------------------------

def validate_and_ingest(
    df: pd.DataFrame,
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """
    Validate schema and row-level quality; split into clean and quarantine sets.

    Parameters
    ----------
    df       : DataFrame returned by load_nyc_kaggle_data()
    save_raw : persist clean + quarantine Parquet + validation JSON to data/raw/

    Returns
    -------
    (clean_df, quarantine_df, ValidationReport)
    """
    log.info(f"Starting validation on {len(df):,} rows …")
    report = ValidationReport(total_rows=len(df))

    df = _validate_schema(df, report)

    bad_gps        = _flag_missing_gps(df, report)
    bad_ts         = _flag_invalid_timestamps(df, report)
    bad_range      = _flag_out_of_range(df, report)
    bad_eta        = _flag_invalid_eta(df, report)
    bad_passengers = _flag_invalid_passengers(df, report)
    bad_dupes      = _flag_duplicate_trip_ids(df, report)

    quarantine_mask = bad_gps | bad_ts | bad_range | bad_eta | bad_passengers | bad_dupes
    clean_df      = df[~quarantine_mask].copy().reset_index(drop=True)
    quarantine_df = df[quarantine_mask].copy().reset_index(drop=True)

    report.quarantine_rows = len(quarantine_df)
    report.clean_rows      = len(clean_df)
    report.log_summary()

    if save_raw:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_df.to_parquet(config.RAW_DIR / f"trips_clean_{ts}.parquet", index=False)
        quarantine_df.to_parquet(config.RAW_DIR / f"trips_quarantine_{ts}.parquet", index=False)
        report_path = config.RAW_DIR / f"validation_report_{ts}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        log.info(f"Saved clean ({len(clean_df):,} rows) and quarantine ({len(quarantine_df):,} rows).")

    return clean_df, quarantine_df, report