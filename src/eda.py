"""
eda.py
-------
Week 1 / Module M2: Exploratory Data Analysis on the engineered feature set.

Produces:
  - A dataset overview (shape, dtypes, missing values, summary stats)
  - Target variable (trip_duration) distribution + outlier view
  - Time-based patterns (hour-of-day, day-of-week, rush hour vs. off-peak)
  - Distance vs. duration relationship (with an implied-speed sanity check)
  - Weather impact on duration
  - Pickup/drop-off spatial density
  - A correlation heatmap of numeric features against the target

All plots are saved as PNG files; a markdown report ties them together with
the numeric findings so it can be dropped straight into a project report.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless -- safe for scripts/CI, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return name


def dataset_overview(df: pd.DataFrame) -> dict:
    """Shape, dtypes, missing values, basic describe() -- the 'what am I working with' step."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    overview = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in missing.items()},
        "numeric_summary": df.select_dtypes(include=[np.number]).describe().round(2).to_dict(),
    }
    logger.info("EDA overview: %d rows, %d cols, %d columns with missing values",
                overview["n_rows"], overview["n_columns"], len(overview["missing_values"]))
    return overview


def plot_target_distribution(df: pd.DataFrame, out_dir: str) -> dict:
    """Distribution of trip_duration (raw + log) and a boxplot to expose outliers/skew."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(df["trip_duration"], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Trip Duration (seconds) - Raw")

    sns.histplot(np.log1p(df["trip_duration"]), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("Trip Duration (log1p) - reduces right-skew")

    sns.boxplot(x=df["trip_duration"], ax=axes[2], color="#C44E52")
    axes[2].set_title("Trip Duration - Boxplot (outlier view)")

    fig.tight_layout()
    fname = _save(fig, out_dir, "01_target_distribution.png")

    stats = df["trip_duration"].describe().round(1).to_dict()
    stats["skewness"] = round(float(df["trip_duration"].skew()), 2)
    return {"plot": fname, "stats": stats}


def plot_time_patterns(df: pd.DataFrame, out_dir: str) -> dict:
    """Trip volume and average duration by hour-of-day and day-of-week."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    trips_by_hour = df.groupby("hour_of_day").size()
    sns.barplot(x=trips_by_hour.index, y=trips_by_hour.values, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Trip Volume by Hour of Day")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Trip count")

    dur_by_hour = df.groupby("hour_of_day")["trip_duration"].mean()
    sns.lineplot(x=dur_by_hour.index, y=dur_by_hour.values, ax=axes[0, 1], marker="o", color="#DD8452")
    axes[0, 1].set_title("Avg Trip Duration by Hour of Day")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Avg duration (s)")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    trips_by_day = df.groupby("day_of_week").size()
    sns.barplot(x=[day_labels[i] for i in trips_by_day.index], y=trips_by_day.values,
                ax=axes[1, 0], color="#55A868")
    axes[1, 0].set_title("Trip Volume by Day of Week")

    rush_compare = df.groupby("is_rush_hour")["trip_duration"].mean()
    sns.barplot(x=["Off-peak", "Rush hour"], y=rush_compare.reindex([0, 1]).values,
                ax=axes[1, 1], color="#8172B2")
    axes[1, 1].set_title("Avg Duration: Rush Hour vs Off-Peak")
    axes[1, 1].set_ylabel("Avg duration (s)")

    fig.tight_layout()
    fname = _save(fig, out_dir, "02_time_patterns.png")

    return {
        "plot": fname,
        "peak_hour_by_volume": int(trips_by_hour.idxmax()),
        "slowest_hour_by_avg_duration": int(dur_by_hour.idxmax()),
        "avg_duration_rush_vs_offpeak_sec": {
            "off_peak": round(float(rush_compare.get(0, np.nan)), 1),
            "rush_hour": round(float(rush_compare.get(1, np.nan)), 1),
        },
    }


def plot_distance_vs_duration(df: pd.DataFrame, out_dir: str) -> dict:
    """Core relationship the model will lean on, plus an implied-speed check to catch bad rows."""
    sample = df.sample(min(3000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.scatterplot(data=sample, x="trip_distance_km", y="trip_duration",
                     hue="is_rush_hour", alpha=0.4, ax=axes[0], palette=["#4C72B0", "#C44E52"])
    axes[0].set_title("Distance vs Duration (by rush hour)")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Duration (s)")

    implied_speed_kmh = (df["trip_distance_km"] / (df["trip_duration"] / 3600)).replace(
        [np.inf, -np.inf], np.nan
    )
    sns.histplot(implied_speed_kmh.clip(upper=100), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("Implied Speed (km/h) - sanity check")
    axes[1].set_xlabel("km/h")

    fig.tight_layout()
    fname = _save(fig, out_dir, "03_distance_vs_duration.png")

    corr = float(df["trip_distance_km"].corr(df["trip_duration"]))
    return {
        "plot": fname,
        "distance_duration_correlation": round(corr, 3),
        "implied_speed_kmh_median": round(float(implied_speed_kmh.median()), 1),
        "implied_speed_kmh_p99": round(float(implied_speed_kmh.quantile(0.99)), 1),
    }


def plot_weather_impact(df: pd.DataFrame, out_dir: str) -> dict:
    """How weather condition and temperature relate to trip duration."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, x="weather_condition", y="trip_duration", hue="weather_condition",
                ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Trip Duration by Weather Condition")

    sample = df.sample(min(3000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="temp_c", y="trip_duration", alpha=0.35, ax=axes[1], color="#4C72B0")
    axes[1].set_title("Temperature vs Duration")
    axes[1].set_xlabel("Temp (C)")

    fig.tight_layout()
    fname = _save(fig, out_dir, "04_weather_impact.png")

    avg_by_condition = df.groupby("weather_condition")["trip_duration"].mean().round(1).to_dict()
    return {"plot": fname, "avg_duration_by_condition_sec": avg_by_condition}


def plot_spatial_density(df: pd.DataFrame, out_dir: str) -> dict:
    """Pickup and drop-off density across the NYC bounding box."""
    sample = df.sample(min(5000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].scatter(sample["pickup_longitude"], sample["pickup_latitude"],
                     s=3, alpha=0.3, color="#4C72B0")
    axes[0].set_title("Pickup Location Density")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    axes[1].scatter(sample["dropoff_longitude"], sample["dropoff_latitude"],
                     s=3, alpha=0.3, color="#C44E52")
    axes[1].set_title("Drop-off Location Density")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    fig.tight_layout()
    fname = _save(fig, out_dir, "05_spatial_density.png")
    return {"plot": fname}


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> dict:
    """Correlation of numeric features against each other and the target."""
    numeric_cols = [
        "trip_duration", "trip_distance_km", "passenger_count", "hour_of_day",
        "day_of_week", "is_weekend", "is_rush_hour", "temp_c", "precip_mm",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fname = _save(fig, out_dir, "06_correlation_heatmap.png")

    target_corr = corr["trip_duration"].drop("trip_duration").sort_values(key=abs, ascending=False)
    return {"plot": fname, "correlation_with_target": target_corr.to_dict()}


def run_full_eda(df: pd.DataFrame, out_dir: str) -> dict:
    """Runs every EDA step and returns a single results dict (also used to render the report)."""
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Running EDA, saving plots to %s", out_dir)

    results = {
        "overview": dataset_overview(df),
        "target_distribution": plot_target_distribution(df, out_dir),
        "time_patterns": plot_time_patterns(df, out_dir),
        "distance_vs_duration": plot_distance_vs_duration(df, out_dir),
        "weather_impact": plot_weather_impact(df, out_dir),
        "spatial_density": plot_spatial_density(df, out_dir),
        "correlation": plot_correlation_heatmap(df, out_dir),
    }
    logger.info("EDA complete: %d plots saved to %s", 6, out_dir)
    return results
