"""
main_week1.py
--------------
Week 1 (Module M2) entry point:

    Ingest -> Validate/Clean -> Split (train/test) -> Feature Engineer
    (fit on train, apply to test) -> Version (train + test separately)

Splitting happens BEFORE feature engineering because the location-bin
features are fit from the data itself (pd.cut derives bin edges from
min/max). Fitting those on the full dataset would leak test-set
distribution info into training features -- see src/data_split.py and
src/feature_engineering.py for details.

Usage:
    python main_week1.py

Reads config.yaml for all paths and thresholds. Place the real Kaggle NYC
Taxi Trip Duration CSV at data/raw/train.csv before running for the actual
submission; otherwise a synthetic dataset with the same schema is used so
the pipeline can be demoed end-to-end.
"""

import os
import sys
import json
import logging
import yaml

sys.path.append(os.path.dirname(__file__))

from src.data_ingestion import load_raw_trips, attach_synthetic_weather
from src.schema_validation import validate_schema, validate_and_clean
from src.data_split import split_train_test
from src.feature_engineering import engineer_features
from src.data_versioning import save_version

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_LOG_DIR, "week1_pipeline.log")),
    ],
)
logger = logging.getLogger("main_week1")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.join(base_dir, cfg["paths"]["logs_dir"]), exist_ok=True)
    os.makedirs(os.path.join(base_dir, cfg["paths"]["processed_data_dir"]), exist_ok=True)

    raw_path = os.path.join(base_dir, cfg["paths"]["raw_data_dir"], cfg["paths"]["raw_data_file"])

    # 1. Ingest
    logger.info("STEP 1/5: Ingesting raw trip data")
    df_raw = load_raw_trips(raw_path, cfg["synthetic_fallback"])
    used_real_data = os.path.exists(raw_path)

    # 2. Validate schema + clean bad rows
    logger.info("STEP 2/5: Validating schema and cleaning data")
    validate_schema(df_raw, cfg["schema"]["required_columns"])
    df_clean, validation_report = validate_and_clean(df_raw, cfg["validation"])

    # 3. Split train/test BEFORE feature engineering (avoids leakage into
    #    data-derived features like location bins; see src/data_split.py)
    logger.info("STEP 3/5: Splitting into train/test")
    split_cfg = cfg["split"]
    train_raw, test_raw = split_train_test(
        df_clean,
        test_size=split_cfg["test_size"],
        method=split_cfg["method"],
        random_state=split_cfg["random_state"],
    )

    # 4. Feature engineering: fit location bins on TRAIN ONLY, apply the
    #    same fitted edges to TEST. Weather is a deterministic function of
    #    calendar date (external data), so joining it separately per split
    #    introduces no leakage.
    logger.info("STEP 4/5: Engineering features (fit on train, apply to test)")
    train_weather = attach_synthetic_weather(train_raw)
    test_weather = attach_synthetic_weather(test_raw)

    train_features, location_bin_edges = engineer_features(
        train_weather, location_bin_edges=None, n_bins=split_cfg["location_bins"]
    )
    test_features, _ = engineer_features(
        test_weather, location_bin_edges=location_bin_edges
    )

    processed_dir = os.path.join(base_dir, cfg["paths"]["processed_data_dir"])
    train_path = os.path.join(processed_dir, "trips_features_train.parquet")
    test_path = os.path.join(processed_dir, "trips_features_test.parquet")
    edges_path = os.path.join(processed_dir, "location_bin_edges.json")

    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)
    with open(edges_path, "w") as f:
        json.dump(location_bin_edges, f, indent=2)

    logger.info("Train features saved to %s (%d rows, %d cols)", train_path, *train_features.shape)
    logger.info("Test features saved to %s (%d rows, %d cols)", test_path, *test_features.shape)
    logger.info("Location bin edges (fit on train) saved to %s", edges_path)

    # 5. Version train and test datasets separately, so Week 2 can pull an
    #    exact, matched (train, test) pair by version id.
    logger.info("STEP 5/5: Versioning datasets")
    versions_dir = os.path.join(base_dir, cfg["paths"]["versions_dir"])
    manifest_path = os.path.join(base_dir, cfg["paths"]["version_manifest"])
    source = "kaggle_nyc_taxi_trip_duration" if used_real_data else "synthetic_fallback"

    train_version = save_version(
        train_features, versions_dir, manifest_path, validation_report,
        source_description=f"{source}_train_split",
    )
    test_version = save_version(
        test_features, versions_dir, manifest_path, validation_report,
        source_description=f"{source}_test_split",
    )

    print("\n===== WEEK 1 PIPELINE SUMMARY =====")
    print(f"Data source            : {source}")
    print(f"Split method            : {split_cfg['method']} (test_size={split_cfg['test_size']})")
    print(f"Rows in / cleaned       : {validation_report['rows_in']} -> {validation_report['rows_out']}")
    print(f"Rows dropped            : {validation_report['total_rows_dropped']} "
          f"({validation_report['drop_rate_pct']}%)")
    print(f"Train rows / test rows  : {len(train_features)} / {len(test_features)}")
    print(f"Feature columns         : {train_features.shape[1]}")
    print(f"Train dataset version   : {train_version['version_id']}")
    print(f"Test dataset version    : {test_version['version_id']}")
    print(f"Train features saved    : {train_path}")
    print(f"Test features saved     : {test_path}")
    print(f"Version manifest        : {manifest_path}")
    print("====================================\n")


if __name__ == "__main__":
    main()
