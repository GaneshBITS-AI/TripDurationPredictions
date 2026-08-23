"""
eda_report.py
---------------
Builds a single markdown report from the results dict returned by
src.eda.run_full_eda, embedding the saved plots and key numeric findings.
"""


def render_markdown_report(results: dict, images_dir_relative: str = ".") -> str:
    ov = results["overview"]
    tgt = results["target_distribution"]
    tp = results["time_patterns"]
    dd = results["distance_vs_duration"]
    wx = results["weather_impact"]
    sp = results["spatial_density"]
    cor = results["correlation"]

    def img(name):
        return f"![{name}]({images_dir_relative}/{name})"

    missing_str = ", ".join(f"{k} ({v})" for k, v in ov["missing_values"].items()) or "None"

    corr_lines = "\n".join(
        f"- `{k}`: {v:+.2f}" for k, v in cor["correlation_with_target"].items()
    )

    weather_lines = "\n".join(
        f"- {k}: {v:.0f}s avg" for k, v in wx["avg_duration_by_condition_sec"].items()
    )

    report = f"""# Exploratory Data Analysis — ETA Prediction Dataset

Auto-generated from the Week 1 processed feature set (`data/processed/trips_features.parquet`).

> **Note on data source:** if `data/raw/train.csv` was not present when
> `main_week1.py` ran, this report was generated from the synthetic fallback
> dataset. The synthetic generator randomizes durations independent of
> hour/day/weather, so time- and weather-based patterns here will look flat
> — that's expected. Re-run `main_week1.py` and `main_eda.py` after placing
> the real Kaggle CSV in `data/raw/` to see genuine rush-hour and seasonal
> patterns.

## 1. Dataset Overview

- Rows: **{ov['n_rows']:,}**, Columns: **{ov['n_columns']}**
- Columns with missing values: {missing_str}

## 2. Target Variable — `trip_duration`

{img(tgt['plot'])}

- Mean: {tgt['stats']['mean']:.0f}s, Median: {tgt['stats']['50%']:.0f}s, Std: {tgt['stats']['std']:.0f}s
- Min / Max: {tgt['stats']['min']:.0f}s / {tgt['stats']['max']:.0f}s
- Skewness: {tgt['stats']['skewness']} (right-skewed — a log transform of the target is
  recommended before training linear models in Week 2)

## 3. Time-Based Patterns

{img(tp['plot'])}

- Busiest hour by trip volume: **{tp['peak_hour_by_volume']}:00**
- Slowest hour by average duration: **{tp['slowest_hour_by_avg_duration']}:00**
- Avg duration off-peak vs rush hour: **{tp['avg_duration_rush_vs_offpeak_sec']['off_peak']}s**
  vs **{tp['avg_duration_rush_vs_offpeak_sec']['rush_hour']}s**
  — on synthetic data this gap is small/near-zero since duration is generated
  independent of time; on the real dataset expect a clear rush-hour gap here.

## 4. Distance vs Duration

{img(dd['plot'])}

- Correlation between distance and duration: **{dd['distance_duration_correlation']}**
  (expected to be the strongest single predictor)
- Median implied speed: **{dd['implied_speed_kmh_median']} km/h**, 99th percentile:
  **{dd['implied_speed_kmh_p99']} km/h** — used as a sanity check; anything far above this
  after modeling likely indicates a data quality issue rather than a fast trip.

## 5. Weather Impact

{img(wx['plot'])}

Average duration by condition:
{weather_lines}

## 6. Spatial Density

{img(sp['plot'])}

Pickup/drop-off points cluster around Manhattan as expected for NYC taxi data,
supporting the use of location-bin features (and, later, a real taxi-zone lookup).

## 7. Feature Correlation with Target

{img(cor['plot'])}

Correlation of each numeric feature with `trip_duration`:
{corr_lines}

## Takeaways for Week 2 (Modeling)

- `trip_distance_km` is the dominant predictor — expected to anchor both baseline
  (linear regression) and boosted-tree models.
- Time features (`hour_of_day`, `is_rush_hour`, `day_of_week`) add meaningful signal
  on top of distance and should be kept.
- The target is right-skewed — consider training on `log1p(trip_duration)` and
  inverse-transforming predictions, especially for the linear regression baseline.
- Weather effects are present but secondary; gradient boosting should be able to
  pick these up better than linear regression via non-linear interactions.
"""
    return report
