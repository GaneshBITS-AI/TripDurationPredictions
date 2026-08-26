"""
Exploratory Data Analysis (EDA) Module
=======================================

Produces:
    * Univariate distributions (target + numeric features)
    * Categorical frequency plots
    * Correlation heatmap
    * ETA vs distance scatter (coloured by traffic level)
    * Temporal patterns (hourly, daily, monthly mean ETA)
    * Geographic density hex-bin for pickup locations
    * A consolidated HTML/PNG report saved to reports/

All plotting functions are side-effect free (no plt.show()) and return
the matplotlib Figure so they can be composited or tested independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / pipeline use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from eta_pipeline import config
from eta_pipeline.logger import get_logger

log = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
REPORT_DIR = config.REPORT_DIR / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- Helper --
def _save_figure(fig: plt.Figure, name: str) -> Path:
    path = REPORT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")
    return path


# ------------------------------------------------------------ Individual plots --
def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ETA Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(df[config.TARGET_COLUMN], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("ETA (minutes)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw")

    log_eta = np.log1p(df[config.TARGET_COLUMN])
    axes[1].hist(log_eta, bins=60, color="#DD8452", edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("log(1 + ETA)")
    axes[1].set_title("Log-transformed")

    _save_figure(fig, "01_target_distribution")
    return fig


def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in df.columns and c != "is_weekend"]
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=40, color="#55A868", edgecolor="white", linewidth=0.3)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, "02_numeric_distributions")
    return fig


def plot_categorical_distributions(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Categorical Feature Frequencies", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, config.CATEGORICAL_FEATURES):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color=sns.color_palette("muted", len(counts)))
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)

    _save_figure(fig, "03_categorical_distributions")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_cols = [c for c in config.NUMERIC_FEATURES if c in df.columns] + [config.TARGET_COLUMN]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, "04_correlation_heatmap")
    return fig


def plot_eta_vs_distance(df: pd.DataFrame) -> plt.Figure:
    """ETA vs haversine distance, coloured by rush-hour flag."""
    dist_col = "haversine_km" if "haversine_km" in df.columns else None
    if dist_col is None:
        # Compute on the fly from raw coordinates
        import math
        if not {"pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"}.issubset(df.columns):
            log.warning("plot_eta_vs_distance: no distance or coordinate columns - skipping.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No distance data available", ha="center", va="center")
            _save_figure(fig, "05_eta_vs_distance")
            return fig
        df = df.copy()
        R = 6371.0
        lat1 = np.radians(df["pickup_lat"].to_numpy())
        lat2 = np.radians(df["dropoff_lat"].to_numpy())
        dlat = lat2 - lat1
        dlon = np.radians(df["dropoff_lon"].to_numpy() - df["pickup_lon"].to_numpy())
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df["haversine_km"] = 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))
        dist_col = "haversine_km"

    sample = df.sample(min(5_000, len(df)), random_state=config.RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(10, 6))
    if "is_rush_hour" in sample.columns:
        palette = {0: "#4C72B0", 1: "#DD8452"}
        labels = {0: "Off-peak", 1: "Rush hour"}
        for flag, grp in sample.groupby("is_rush_hour"):
            ax.scatter(
                grp[dist_col], grp[config.TARGET_COLUMN],
                alpha=0.35, s=15,
                color=palette.get(int(flag), "#555"),
                label=labels.get(int(flag), str(flag)),
            )
        ax.legend(title="Period")
    else:
        ax.scatter(sample[dist_col], sample[config.TARGET_COLUMN], alpha=0.35, s=15)

    ax.set_xlabel("Haversine Distance (km)")
    ax.set_ylabel("ETA (minutes)")
    ax.set_title("ETA vs Distance (coloured by Rush Hour)", fontweight="bold")
    _save_figure(fig, "05_eta_vs_distance")
    return fig


def plot_temporal_patterns(df: pd.DataFrame) -> plt.Figure:
    df = df.copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.day_name()
    df["month"] = df["pickup_datetime"].dt.month

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Temporal ETA Patterns", fontsize=14, fontweight="bold")

    # Hourly
    hourly = df.groupby("hour")[config.TARGET_COLUMN].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o", color="#4C72B0", linewidth=2)
    axes[0].fill_between(hourly.index, hourly.values, alpha=0.15, color="#4C72B0")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Mean ETA (min)")
    axes[0].set_title("Mean ETA by Hour")
    axes[0].xaxis.set_major_locator(mticker.MultipleLocator(4))

    # Day-of-week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("weekday")[config.TARGET_COLUMN].mean().reindex(day_order)
    colors = ["#DD8452" if d in ("Saturday", "Sunday") else "#4C72B0" for d in day_order]
    axes[1].bar(day_order, daily.values, color=colors)
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Mean ETA (min)")
    axes[1].set_title("Mean ETA by Weekday")
    axes[1].tick_params(axis="x", rotation=30)

    # Monthly
    monthly = df.groupby("month")[config.TARGET_COLUMN].mean()
    axes[2].plot(monthly.index, monthly.values, marker="s", color="#55A868", linewidth=2)
    axes[2].fill_between(monthly.index, monthly.values, alpha=0.15, color="#55A868")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Mean ETA (min)")
    axes[2].set_title("Mean ETA by Month")
    axes[2].xaxis.set_major_locator(mticker.MultipleLocator(1))

    fig.tight_layout()
    _save_figure(fig, "06_temporal_patterns")
    return fig


def plot_weather_traffic_eta(df: pd.DataFrame) -> plt.Figure:
    """ETA by real weather signals: temperature band, precipitation, and snow."""
    df = df.copy()

    has_temp = "avg_temp_f" in df.columns
    has_precip = "precipitation" in df.columns
    has_snow = "is_snowing" in df.columns

    if not (has_temp or has_precip or has_snow):
        log.warning("plot_weather_traffic_eta: no weather columns - skipping.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No weather data available", ha="center", va="center")
        _save_figure(fig, "07_weather_traffic_eta")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ETA by Weather Conditions (NYC Data)", fontsize=14, fontweight="bold")

    # --- Panel 1: Mean ETA by temperature band ---
    if has_temp:
        df["temp_band"] = pd.cut(
            df["avg_temp_f"],
            bins=[0, 32, 50, 65, 80, 120],
            labels=["Freezing (<32F)", "Cold (32-50F)", "Cool (50-65F)",
                    "Warm (65-80F)", "Hot (>80F)"],
        )
        band_order = ["Freezing (<32F)", "Cold (32-50F)", "Cool (50-65F)",
                      "Warm (65-80F)", "Hot (>80F)"]
        temp_means = df.groupby("temp_band", observed=True)[config.TARGET_COLUMN].mean().reindex(band_order)
        axes[0].bar(temp_means.index, temp_means.values,
                    color=sns.color_palette("coolwarm", len(band_order)))
        axes[0].set_xlabel("Temperature Band")
        axes[0].set_ylabel("Mean ETA (min)")
        axes[0].set_title("Mean ETA by Temperature")
        axes[0].tick_params(axis="x", rotation=25)
    else:
        axes[0].set_visible(False)

    # --- Panel 2: Mean ETA - rainy vs dry ---
    if has_precip:
        df["is_rainy"] = (df["precipitation"] > 0.1).map({True: "Rainy", False: "Dry"})
        rain_means = df.groupby("is_rainy")[config.TARGET_COLUMN].mean()
        axes[1].bar(rain_means.index, rain_means.values,
                    color=["#4C72B0", "#DD8452"][:len(rain_means)])
        axes[1].set_xlabel("Precipitation")
        axes[1].set_ylabel("Mean ETA (min)")
        axes[1].set_title("Mean ETA: Rainy vs Dry")
    else:
        axes[1].set_visible(False)

    # --- Panel 3: Mean ETA - snowy vs clear ---
    if has_snow:
        snow_means = df.groupby("is_snowing")[config.TARGET_COLUMN].mean()
        labels = {0: "No Snow", 1: "Snowing"}
        x_labels = [labels.get(int(k), str(k)) for k in snow_means.index]
        axes[2].bar(x_labels, snow_means.values, color=["#55A868", "#C44E52"][:len(snow_means)])
        axes[2].set_xlabel("Snow")
        axes[2].set_ylabel("Mean ETA (min)")
        axes[2].set_title("Mean ETA: Snowy vs Clear")
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, "07_weather_traffic_eta")
    return fig


def plot_pickup_density(df: pd.DataFrame) -> plt.Figure:
    sample = df.sample(min(10_000, len(df)), random_state=config.RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(
        sample["pickup_lon"], sample["pickup_lat"],
        gridsize=50, cmap="YlOrRd", mincnt=1,
    )
    plt.colorbar(hb, ax=ax, label="Trip count")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Pickup Location Density", fontweight="bold")
    _save_figure(fig, "08_pickup_density")
    return fig


def plot_boxplot_eta_by_hour(df: pd.DataFrame) -> plt.Figure:
    df = df.copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    sample = df.sample(min(15_000, len(df)), random_state=config.RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(16, 5))
    sample.boxplot(
        column=config.TARGET_COLUMN,
        by="hour",
        ax=ax,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("ETA (minutes)")
    ax.set_title("ETA Distribution by Hour of Day", fontweight="bold")
    fig.suptitle("")
    fig.tight_layout()
    _save_figure(fig, "09_boxplot_eta_by_hour")
    return fig


# --------------------------------------------------------- Summary statistics --
def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    num_cols = [c for c in df.select_dtypes(include="number").columns]
    desc = df[num_cols].describe().round(4).to_dict()

    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "descriptive_stats": desc,
        "target_skewness": float(df[config.TARGET_COLUMN].skew()),
        "target_kurtosis": float(df[config.TARGET_COLUMN].kurtosis()),
    }

    return summary


# --------------------------------------------------------------- Orchestrator --
def _safe_plot(fn, *args, **kwargs):
    """Call a plot function; log and continue on any error."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        log.warning(f"{fn.__name__} failed and was skipped: {exc}")


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute the full EDA suite on a clean DataFrame.

    Parameters
    ----------
    df : clean trip DataFrame (output of ingestion.validate_and_ingest)

    Returns
    -------
    summary dict with descriptive statistics
    """
    log.info("=" * 60)
    log.info("STARTING EDA")
    log.info("=" * 60)

    summary = compute_summary_stats(df)
    log.info(f"Dataset shape : {summary['n_rows']:,} rows x {summary['n_columns']} columns")
    log.info(f"Target skewness  : {summary['target_skewness']:.3f}")
    log.info(f"Target kurtosis  : {summary['target_kurtosis']:.3f}")

    missing = {k: v for k, v in summary["missing_values"].items() if v > 0}
    if missing:
        log.info(f"Columns with missing values: {missing}")
    else:
        log.info("No missing values in clean dataset.")

    log.info("Generating plots ...")

    # Enrich with hour/weekday/month only if datetime cols are present
    enriched = df.copy()
    if "pickup_datetime" in df.columns:
        enriched["hour_of_day"] = df["pickup_datetime"].dt.hour
        enriched["day_of_week"] = df["pickup_datetime"].dt.dayofweek
        enriched["month"] = df["pickup_datetime"].dt.month
        enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)

    plot_target_distribution(df)
    plot_numeric_distributions(enriched)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(enriched)
    _safe_plot(plot_eta_vs_distance, df)
    _safe_plot(plot_temporal_patterns, df)
    _safe_plot(plot_weather_traffic_eta, df)
    _safe_plot(plot_pickup_density, df)
    _safe_plot(plot_boxplot_eta_by_hour, df)

    # Persist summary
    stats_path = REPORT_DIR / "summary_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary stats saved: {stats_path}")

    log.info("EDA complete. All outputs in: %s", REPORT_DIR)
    return summary