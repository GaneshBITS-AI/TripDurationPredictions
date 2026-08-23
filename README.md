# ETA Prediction Pipeline — Mini Project (PG AI & ML)

End-to-end ML pipeline that predicts trip ETA (delivery time / ride duration)
from trip distance, time-of-day, weather, and pickup/drop-off location.

**Problem statement:** A logistics/ride-hailing platform wants to predict
delivery time based on trip distance, time of day, weather, traffic
patterns, and pickup/drop-off location — ingest data, engineer features,
train and compare models, deploy the best model as a service, and monitor
it for accuracy drift.

Built across 4 weekly modules:

| Week | Module | Scope |
|------|--------|-------|
| **1** | M2 | Data ingestion, schema validation, feature engineering, dataset versioning ✅ *(this delivery)* |
| 2 | M3 | Train/compare models (linear regression vs. gradient boosting), experiment tracking |
| 3 | M4 | Package best model, serve via REST API |
| 4 | M5 | Log predictions vs. actuals, simulate drift, monitoring + retraining trigger |

---

## Week 1 (Module M2) — what's implemented

1. **Ingestion** (`src/data_ingestion.py`) — loads the Kaggle **NYC Taxi Trip
   Duration** dataset from `data/raw/train.csv`. If that file isn't present,
   it automatically falls back to a synthetic dataset with the *identical
   schema* (with some deliberately dirty rows) so the whole pipeline can be
   run and demoed without the real file.
2. **Schema validation** (`src/schema_validation.py`) — confirms all
   required columns exist, then flags/drops rows with missing or
   out-of-NYC-bounds GPS pings, invalid timestamps (dropoff before pickup,
   implausible durations), and invalid passenger counts. Every drop is
   counted and logged for auditability.
3. **Feature engineering** (`src/feature_engineering.py`) — adds
   `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `is_rush_hour`,
   haversine `trip_distance_km`, and coarse pickup/drop-off location bins.
   Synthetic daily weather (`temp_c`, `precip_mm`, `weather_condition`) is
   joined in via `attach_synthetic_weather` in `data_ingestion.py` — swap
   this for a real weather API/NOAA dataset when available.
4. **Train/test split** (`src/data_split.py`) — splits the cleaned data
   into train/test **before** any data-derived feature engineering. Default
   is a **time-based split** (train = earlier trips, test = later trips),
   which matches how the model will actually be used in production (predict
   on future trips, never trained on them) and avoids the "seeing the
   future" leakage a random split can introduce here — a random split is
   also available via `config.yaml` if your course requires one.
5. **Feature engineering** (`src/feature_engineering.py`) — adds
   `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `is_rush_hour`,
   haversine `trip_distance_km`, and coarse pickup/drop-off location bins.
   Synthetic daily weather (`temp_c`, `precip_mm`, `weather_condition`) is
   joined in via `attach_synthetic_weather` in `data_ingestion.py` — swap
   this for a real weather API/NOAA dataset when available.
   **Leakage note:** the location-bin edges are *fit on the train split
   only* (`fit_location_bins`) and then *applied* to test with the exact
   same edges (`apply_location_bins`) — fitting them on the full dataset
   would leak test-set distribution info into the training features. Any
   data-derived transform added in Week 2+ (scalers, target encoders,
   imputation statistics) should follow the same fit-on-train/apply-to-test
   pattern.
6. **Dataset versioning** (`src/data_versioning.py`) — writes a timestamped
   Parquet snapshot for **each split** to `data/versions/` and appends
   metadata (row/column counts, content hash, validation report, source) to
   `data/versions/manifest.json`. Downstream weeks can fetch the latest
   train/test version via `latest_version()`.
7. **Exploratory Data Analysis** (`src/eda.py`, `src/eda_report.py`,
   run via `main_eda.py`) — runs on the **train split only** (EDA on test
   data is itself a form of leakage — decisions informed by test data
   shouldn't influence modeling). Covers target distribution & skew, time
   patterns, distance-vs-duration relationship with an implied-speed sanity
   check, weather impact, spatial density, and a correlation heatmap. Saves
   6 plots plus `EDA_REPORT.md` to `eda_outputs/`.

## Getting the real dataset

Download **NYC Taxi Trip Duration** from Kaggle:
https://www.kaggle.com/c/nyc-taxi-trip-duration/data

Place `train.csv` at:
```
data/raw/train.csv
```

If this file is absent, `main_week1.py` automatically uses synthetic data
(same schema) so you can still run and evaluate the pipeline.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main_week1.py     # ingest -> validate -> engineer features -> version
python main_eda.py       # exploratory data analysis on the processed features
```

**`main_week1.py`** will:
- Load raw data (real or synthetic fallback)
- Validate + clean it
- **Split into train/test** (time-based by default — see `config.yaml` to switch to random)
- Engineer features (location bins fit on train, applied to test — no leakage)
- Save `data/processed/trips_features_train.parquet` and `trips_features_test.parquet`
- Save the fitted location bin edges to `data/processed/location_bin_edges.json`
- Save versioned snapshots + manifest entries (one per split) in `data/versions/`
- Print a summary and write full logs to `logs/week1_pipeline.log`

**`main_eda.py`** will (run this after `main_week1.py`):
- Load the **processed training split only** (`trips_features_train.parquet`)
- Generate 6 plots (target distribution, time patterns, distance vs.
  duration, weather impact, spatial density, correlation heatmap) into
  `eda_outputs/`
- Write `eda_outputs/EDA_REPORT.md` summarizing the numeric findings and
  linking each plot

## Project structure

```
eta_prediction_pipeline/
├── config.yaml                 # paths, schema, validation thresholds
├── requirements.txt
├── main_week1.py                # Week 1: ingest/validate/split/features/version
├── main_eda.py                  # EDA orchestrator (train split only)
├── src/
│   ├── data_ingestion.py       # load real/synthetic data + weather join
│   ├── schema_validation.py    # column/GPS/timestamp/passenger checks
│   ├── data_split.py           # train/test split (time-based by default)
│   ├── feature_engineering.py  # time, distance, location features
│   ├── data_versioning.py      # Parquet snapshots + JSON manifest
│   ├── eda.py                  # EDA plots + stats
│   └── eda_report.py           # renders EDA_REPORT.md from eda.py results
├── data/
│   ├── raw/                    # put train.csv here
│   ├── processed/
│   │   ├── trips_features_train.parquet
│   │   ├── trips_features_test.parquet
│   │   └── location_bin_edges.json   # bin edges fit on train, reused on test
│   └── versions/               # versioned snapshots (train + test) + manifest.json
├── eda_outputs/                 # EDA plots + EDA_REPORT.md (train split only)
└── logs/                       # pipeline run logs
```

## Design notes / what to mention in your report

- **Why split before feature engineering?** Row-wise features
  (`hour_of_day`, `trip_distance_km`, etc.) only look at their own row, so
  split timing doesn't affect them. But the location-bin features derive
  their bin edges from the data's own min/max — fitting that on the full
  dataset would leak test-set information into training features. The
  pipeline splits first, fits bin edges on train only
  (`fit_location_bins`), and reuses those exact edges on test
  (`apply_location_bins`). Any transform you add later that's *fit* on data
  (scalers, target encoders, imputers) should follow the same pattern.
- **Why a time-based split, not random?** ETA prediction is a forecasting
  problem — in production the model only ever sees trips after it was
  trained. A random split would mix future trips into training and make
  test performance look better than it will in production. Set
  `split.method: random` in `config.yaml` if your assignment specifically
  wants a random split instead.
- **Why synthetic fallback for weather?** The Kaggle NYC Taxi dataset has no
  weather column. A production system would join real hourly/daily weather
  (e.g., NOAA Central Park station, or a weather API keyed by date). The
  synthetic version here uses a seasonal sine curve + noise so the feature
  behaves realistically for demoing the rest of the pipeline; it's clearly
  isolated in one function (`attach_synthetic_weather`) for easy replacement.
- **Why bucket lat/lon instead of a real zone lookup?** A production system
  would join against the official NYC taxi zone shapefile/lookup table for
  borough/neighborhood features. Grid-binning is a placeholder that keeps
  the pipeline dependency-free; swapping in real zones only touches
  `add_location_bucket_features`.
- **Versioning approach:** implemented as lightweight Parquet snapshots + a
  JSON manifest rather than DVC/MLflow, to keep the submission
  self-contained. The manifest schema (version_id, hash, row/col counts,
  validation report, source) maps directly onto DVC or MLflow's dataset
  tracking if you want to extend it in Week 2+.

## Next steps (Weeks 2–4)

- **Week 2:** load the latest versioned dataset via
  `src/data_versioning.latest_version()`, train linear regression and
  gradient boosting (e.g., XGBoost/LightGBM) baselines, track experiments
  (MLflow or a simple results CSV), compare RMSE/MAE.
- **Week 3:** serialize the winning model, wrap it in a FastAPI service with
  a `/predict` endpoint accepting trip details.
- **Week 4:** log predictions vs. actuals, simulate drift (e.g., resample
  rush-hour/festival periods), and set up a monitoring + retraining trigger.
