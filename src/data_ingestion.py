"""
data_ingestion.py
------------------
Week 1 / Module M2: Ingest historical trip data.

Primary path: reads the Kaggle "NYC Taxi Trip Duration" file
(https://www.kaggle.com/c/nyc-taxi-trip-duration/data) from data/raw/train.csv.

Because that file must be downloaded manually (Kaggle requires auth), this
module also ships a synthetic-data fallback so the rest of the pipeline can
be developed and tested end-to-end before the real file is in place. Swap in
the real CSV at any time -- no other code needs to change.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_synthetic_trips(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mirrors the NYC Taxi Trip Duration
    schema exactly, so downstream validation/feature-engineering code can be
    built and tested without the real Kaggle file.

    NOTE: This is a stand-in for development only. Replace with the real
    Kaggle CSV in data/raw/train.csv for the actual submission run.
    """
    rng = np.random.default_rng(seed)

    n = n_rows
    start = pd.Timestamp("2016-01-01")
    end = pd.Timestamp("2016-06-30")
    pickup_ts = start + pd.to_timedelta(
        rng.integers(0, int((end - start).total_seconds()), size=n), unit="s"
    )

    # NYC bounding box for plausible pickup/drop-off coordinates
    pickup_lat = rng.uniform(40.60, 40.85, size=n)
    pickup_lon = rng.uniform(-74.03, -73.75, size=n)
    dropoff_lat = rng.uniform(40.60, 40.85, size=n)
    dropoff_lon = rng.uniform(-74.03, -73.75, size=n)

    # Rough duration model: base + distance-driven noise, so it correlates
    # sensibly with the coordinates generated above (haversine-ish proxy).
    approx_dist_deg = np.sqrt((dropoff_lat - pickup_lat) ** 2 + (dropoff_lon - pickup_lon) ** 2)
    # ~111 km per degree of latitude near NYC; assume ~20 km/h average city speed
    approx_dist_km = approx_dist_deg * 111
    trip_duration = (120 + (approx_dist_km / 20) * 3600 * rng.uniform(0.7, 1.3, size=n)).astype(int)
    trip_duration = np.clip(trip_duration, 60, 5400)
    dropoff_ts = pickup_ts + pd.to_timedelta(trip_duration, unit="s")

    df = pd.DataFrame({
        "id": [f"id{i:07d}" for i in range(n)],
        "vendor_id": rng.choice([1, 2], size=n),
        "pickup_datetime": pickup_ts,
        "dropoff_datetime": dropoff_ts,
        "passenger_count": rng.integers(1, 6, size=n),
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "store_and_fwd_flag": rng.choice(["N", "Y"], size=n, p=[0.995, 0.005]),
        "trip_duration": trip_duration,
    })

    # Deliberately inject a few dirty rows so schema validation has
    # something real to catch (missing GPS, bad timestamps, etc.)
    dirty_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[dirty_idx[: len(dirty_idx) // 3], "pickup_latitude"] = np.nan
    df.loc[dirty_idx[len(dirty_idx) // 3: 2 * len(dirty_idx) // 3], "dropoff_datetime"] = df.loc[
        dirty_idx[len(dirty_idx) // 3: 2 * len(dirty_idx) // 3], "pickup_datetime"
    ] - pd.Timedelta(minutes=5)  # dropoff before pickup
    df.loc[dirty_idx[2 * len(dirty_idx) // 3:], "trip_duration"] = 5  # implausibly short

    logger.warning(
        "Using SYNTHETIC fallback data (%d rows). Place the real Kaggle "
        "NYC Taxi Trip Duration CSV at data/raw/train.csv to use real data.",
        n,
    )
    return df


def load_raw_trips(raw_path: str, synthetic_cfg: dict) -> pd.DataFrame:
    """
    Loads the raw trip dataset.

    Parameters
    ----------
    raw_path : str
        Full path to the raw CSV file (data/raw/train.csv).
    synthetic_cfg : dict
        Config block controlling the synthetic fallback (enabled, n_rows, seed).

    Returns
    -------
    pd.DataFrame
    """
    if os.path.exists(raw_path):
        logger.info("Loading raw trip data from %s", raw_path)
        df = pd.read_csv(raw_path, parse_dates=["pickup_datetime", "dropoff_datetime"])
        logger.info("Loaded %d rows, %d columns", *df.shape)
        return df

    if not synthetic_cfg.get("enabled", True):
        raise FileNotFoundError(
            f"Raw data file not found at {raw_path} and synthetic fallback is disabled. "
            f"Download the NYC Taxi Trip Duration dataset from Kaggle and place it there."
        )

    logger.warning("Raw data file not found at %s -- generating synthetic data instead.", raw_path)
    return _generate_synthetic_trips(
        n_rows=synthetic_cfg.get("n_rows", 5000),
        seed=synthetic_cfg.get("seed", 42),
    )


def attach_synthetic_weather(df: pd.DataFrame, date_col: str = "pickup_datetime", seed: int = 7) -> pd.DataFrame:
    """
    The Kaggle NYC Taxi dataset has no weather column. In production this
    step would join against a real weather source (e.g., NOAA daily
    observations for Central Park, or a weather API keyed on date/hour).

    For this project, we simulate a plausible daily weather series
    (temperature in Celsius, precipitation in mm, and a simple condition
    label) keyed on calendar date, and left-join it onto the trips.
    Replace `_simulate_daily_weather` with a real data source when available.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col]).dt.date
    unique_dates = pd.Series(sorted(dates.unique()))

    rng = np.random.default_rng(seed)
    # Simple seasonal temperature curve + noise
    day_of_year = pd.DatetimeIndex(pd.to_datetime(unique_dates)).dayofyear
    seasonal_temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_c = seasonal_temp + rng.normal(0, 3, size=len(unique_dates))
    precip_mm = np.clip(rng.exponential(1.5, size=len(unique_dates)) - 1.0, 0, None)
    condition = np.where(precip_mm > 3, "rain", np.where(temp_c < 2, "snow_risk", "clear"))

    weather = pd.DataFrame({
        "trip_date": unique_dates,
        "temp_c": temp_c.round(1),
        "precip_mm": precip_mm.round(1),
        "weather_condition": condition,
    })

    df["trip_date"] = dates
    df = df.merge(weather, on="trip_date", how="left")
    return df
