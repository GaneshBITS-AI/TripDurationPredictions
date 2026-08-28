from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow

from eta_pipeline import config
from eta_pipeline.logger import get_logger
from eta_pipeline.mlflow_tracking import (
    setup_mlflow,
    start_pipeline_run,
    log_ingestion_metrics,
    log_dataset_artifact,
    log_eda_summary,
    log_preprocessing_params,
    log_eda_artifacts
)

log = get_logger("main")

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ETA Prediction Pipeline - NYC Taxi(Kaggle)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, default=None, help="Path to Kaggle train.csv"
    )

    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--scaler", choices=["standard", "minmax", "robust"], default=config.SCALER_TYPE
    )
    parser.add_argument("--encoder", choices=["ordinal", "onehot"], default="ordinal")
    return parser.parse_args(argv)

def _banner(title: str)->None:
    line= "=" * 64
    log.info(line)
    log.info(f"Title: {title}")
    log.info(line)

def main(argv=None):
    args = _parse_args(argv)
    t0  = time.perf_counter()

    client = setup_mlflow()
    run_name = f"eta-nyc-{datetime.now().strftime("%Y%m%d-%H%M%S")}"
    print(run_name)

    _banner("ETA Prediction Pipeline - NYC Taxi(Kaggle)")
    log.info(f"MLflow Experiment started: {config.MLFLOW_EXPERIMENT_NAME}")
    log.info(f"MLflow store: {config.MLFLOW_TRACKING_URI}")

    with start_pipeline_run(run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        mlflow.log_params({
            "pipeline.scaler" : args.scaler,
            "pipeline.encoder" : args.encoder,
            "pipeline.run_id" : parent_run_id,
            "pipeline.experiment_name" : config.MLFLOW_EXPERIMENT_NAME,
            "pipeline.mlflow_tracking_uri" : config.MLFLOW_TRACKING_URI,
            "pipeline.skip_eda" : args.skip_eda,
            "pipeline.skip_train" : args.skip_train,
        })

        _banner("STAGE 1 - LOAD NYC TAXI + WEATHER DATA & VALIDATION")
        from eta_pipeline.ingestion import (
            load_nyc_kaggle_data,
            load_weather_data,
            merge_weather_with_trips,
            validate_and_ingest
        )

        input_path = Path(args.input) if args.input else None
        raw_df = load_nyc_kaggle_data(path=input_path)
        weather_df = load_weather_data()
        raw_df = merge_weather_with_trips(raw_df, weather_df)
        mlflow.log_metric("ingestion.weather_days", len(weather_df))

        clean_df, quarantine_df , val_report = validate_and_ingest(raw_df, save_raw=True)

        log_ingestion_metrics(val_report)

        if val_report.pass_rate < 0.80:
            mlflow.set_tag("pipeline.status", "FAILED_VALIDATION")
            log.error(f"Pass Rate {val_report.pass_rate: .1%} is below 80% threshold")
            sys.exit(1)

        log.info(
            f"Ingestion complete: {val_report.clean_rows:,} clean rows "
            f"({val_report.pass_rate:.1%} pass rate)."
        )

        # --- STAGE 2: EDA -----------------------------------------------------
        eda_summary = {}
        if args.skip_eda:
            log.info("STAGE 2 - EDA skipped (--skip-eda flag set).")
            mlflow.set_tag("eda.skipped", "true")
        else:
            _banner("STAGE 2 - EXPLORATORY DATA ANALYSIS")
            from eta_pipeline.eda import run_eda
            eda_summary = run_eda(clean_df)
            log_eda_summary(eda_summary)
            log_eda_artifacts()
            log.info(
                f"EDA complete - target mean ETA: "
                f"{eda_summary['descriptive_stats'][config.TARGET_COLUMN]['mean']:.1f} min"
            )

        # --- STAGE 3: Split -> Feature Engineer -> Encode -> Scale -> Version ------
        _banner("STAGE 3 - PREPROCESSING  (split -> engineer -> encode -> scale)")
        from eta_pipeline.preprocessing import preprocess_and_version, _compute_sha256

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, encoder, version,
        ) = preprocess_and_version(
            clean_df,
            encoder_type=args.encoder,
            scaler_type=args.scaler,
        )

        sha256 = _compute_sha256(clean_df)
        log_preprocessing_params(
            version=version,
            scaler_type=args.scaler,
            encoder_type=args.encoder,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            n_features=X_train.shape[1],
            sha256=sha256,
        )
        log_dataset_artifact(version)
        log.info(f"Preprocessing complete - dataset version: {version}")
        log.info(
            f"Splits - train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}"
        )
        log.info(f"Feature count after engineering: {X_train.shape[1]}")

        # --- STAGE 4: Model Training & Registry --------------------------------
        comparison = None
        if args.skip_train:
            log.info("STAGE 4 - Model training skipped (--skip-train flag set).")
            mlflow.set_tag("training.skipped", "true")
        else:
            _banner("STAGE 4 - MODEL TRAINING, COMPARISON & MODEL REGISTRY")
            from eta_pipeline.model_training import train_all_models
            comparison = train_all_models(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                dataset_version=version,
            )

            champ = comparison.champion
            mlflow.log_params({
                "champion.model_name": champ.name,
                "champion.registry_ver": champ.registered_ver,
            })
            mlflow.log_metrics({
                "champion.val_rmse": champ.val_metrics.get("val_rmse", 0),
                "champion.val_mae": champ.val_metrics.get("val_mae", 0),
                "champion.val_r2": champ.val_metrics.get("val_r2", 0),
                "champion.test_rmse": champ.test_metrics.get("test_rmse", 0),
                "champion.test_mae": champ.test_metrics.get("test_mae", 0),
                "champion.test_r2": champ.test_metrics.get("test_r2", 0),
            })
            mlflow.set_tag("pipeline.status", "SUCCESS")

        # --- Final summary -------------------------------------------------------
        elapsed = time.perf_counter() - t0
        mlflow.log_metric("pipeline.elapsed_sec", round(elapsed, 2))
        mlflow.set_tag("pipeline.run_name", run_name)

        _banner("PIPELINE COMPLETE")
        log.info(f"  Parent run ID        : {parent_run_id}")
        log.info(f"  Dataset version      : {version}")
        log.info(f"  Train / Val / Test   : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
        log.info(f"  Features             : {X_train.shape[1]}")
        log.info(f"  Scaler               : {type(scaler).__name__}")
        log.info(f"  Encoder              : {type(encoder).__name__ if encoder else 'None'}")
        if comparison:
            log.info(f"  Champion model       : {comparison.champion.name}")
            log.info(f"  Champion val RMSE    : {comparison.champion.val_rmse:.3f} min")
            log.info(f"  Registry version     : {comparison.champion.registered_ver}")
        log.info(f"  Versioned artifacts  -> {config.VERSIONED_DIR / version}")
        log.info(f"  Total elapsed time   : {elapsed:.1f}s")
        log.info("")
        log.info("  MLflow UI: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db"
                 "                -> http://localhost:5000")

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "scaler": scaler, "encoder": encoder,
            "version": version,
            "val_report": val_report,
            "comparison": comparison,
            "parent_run_id": parent_run_id,
        }

if __name__ == "__main__":
    main()