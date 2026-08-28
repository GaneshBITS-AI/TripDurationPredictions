import pathlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VERSIONED_DIR = PROCESSED_DIR / "versioned"
REPORT_DIR = PROCESSED_DIR / "report"
MODELS_DIR = PROCESSED_DIR / "models"
LOGS_DIR = PROCESSED_DIR / "logs"

MLRUNS_DIR = PROCESSED_DIR / "mlruns"

for _d in [RAW_DIR, DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, MLRUNS_DIR, VERSIONED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

NYC_RAW_FILENAME = "NYC.csv"
KAGGLE_RAW_FILENAME = "train.csv"
WEATHER_RAW_FILENAME = "weather_data_nyc.csv"

KAGGLE_COLUMN_MAP = {
    "id": "trip_id",
    "vendor_id" : "vendor_id",
    "pickup_datetime" : "pickup_datetime",
    "dropoff_datetime" : "dropoff_datetime",
    "passenger_count": "passenger_count",
    "pickup_longitude": "pickup_lon",
    "pickup_latitude": "pickup_lat",
    "dropoff_latitude": "dropoff_lat",
    "dropoff_longitude": "dropoff_lon",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "trip_duration": "trip_duration_sec"
}

REQUIRED_COLUMNS = [
    "trip_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_lon",
    "pickup_lat",
    "dropoff_lat",
    "dropoff_lon",
    "vendor_id",
    "eta_minutes",
]

# ---------------------------- Validation thresholds (tight NYC bounding box) ----------------------------
LAT_RANGE        = (40.50, 40.92)      # NYC tight bounding box
LON_RANGE        = (-74.26, -73.68)
MAX_ETA_MINUTES  = 120.0               # > 2 hours is an outlier for NYC taxi
MIN_ETA_MINUTES  = 1.0
MAX_PASSENGER    = 6                   # NYC taxi legal max
MAX_DISTANCE_KM  = 120.0               # allow airport runs

# ---------------------------- Feature engineering ----------------------------
CATEGORICAL_FEATURES = ["vendor_id", "store_and_fwd_flag"]

# Weather column names (post-normalisation in ingestion)
WEATHER_TEMP_COL   = "avg_temp_f"
WEATHER_PRECIP_COL = "precipitation"
WEATHER_SNOW_COL   = "snow_depth"

NUMERIC_FEATURES = [
    "passenger_count",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "haversine_km",
    "bearing_deg",
    "pickup_lat",
    "pickup_lon",
    "dropoff_lat",
    "dropoff_lon",
    "delta_lat",
    "delta_lon",
    "log_haversine_km",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "bearing_sin",
    "bearing_cos",
    # weather features
    "avg_temp_f",
    "precipitation",
    "snow_depth",
    "is_snowing",
    "is_raining",
]
TARGET_COLUMN = "eta_minutes"

# ---------------------------- Preprocessing ----------------------------
SCALER_TYPE = "standard"   # "standard" | "minmax" | "robust"
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10

# ---------------------------- Dataset versioning ----------------------------
DATASET_VERSION_PREFIX = "v"

# ---------------------------- MLflow ----------------------------
MLFLOW_TRACKING_URI     = f"sqlite:///{ROOT_DIR / 'mlruns' / 'mlflow.db'}"  # SQLite backend (enables Model Registry on Windows)
MLFLOW_EXPERIMENT_NAME  = "eta-prediction-pipeline"
MLFLOW_REGISTERED_MODEL = "eta-best-model"
MLFLOW_PRODUCTION_STAGE = "Production"

# Models to train and compare
MODEL_CANDIDATES = ["linear_regression", "random_forest", "gradient_boosting", "xgboost"]

# Metric used to select the best model (lower is better)
CHAMPION_METRIC = "val_rmse"

# ---------------------------- Hyperparameter search ----------------------------
# Number of random parameter combinations to try per model family
HP_SEARCH_ITER = 20
# Cross-validation folds used during search (applied on train split only)
HP_CV_FOLDS = 3
