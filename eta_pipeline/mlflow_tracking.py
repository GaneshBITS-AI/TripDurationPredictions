from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from eta_pipeline import config
from eta_pipeline.config import REPORT_DIR
from eta_pipeline.logger import get_logger

log = get_logger(__name__)

def setup_mlflow() -> mlflow.tracking.MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    log.info(f"MLFlow tracking URI: {config.MLFLOW_TRACKING_URI}")

    client = MlflowClient()
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)

    if exp is None:
        exp = client.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
        log.info(f"Created experiment {config.MLFLOW_EXPERIMENT_NAME} - {exp}")
    else:
        log.info(f"Experiment {config.MLFLOW_EXPERIMENT_NAME} already exists")

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    return client

@contextmanager
def start_pipeline_run(run_name: str) -> Generator[mlflow.ActiveRun, None, None]:

    with mlflow.start_run(run_name=run_name, nested=False) as run:
        mlflow.set_tag("pipeline.stage", "orchestration")
        mlflow.set_tag("pipeline.version", "1.0.0")
        log.info(f"MLflow parent run started: {run.info.run_id}")
        yield run

def log_ingestion_metrics(report) -> None:
    mlflow.set_tag("stage", "ingestion")
    mlflow.log_metrics({
        "ingestion.total_rows": report.total_rows,
        "ingestion.clean_rows" : report.clean_rows,
        "ingestion.quarantine_rows" : report.quarantine_rows,
        "ingestion.missing_gps_rows" : report.missing_gps_rows,
        "ingestion.invalid_timestamp_rows" : report.invalid_timestamp_rows,
        "ingestion.out_of_range_rows" : report.out_of_range_rows,
        "ingestion.negative_eta_rows" : report.negative_eta_rows,
        "ingestion.duplicate_trip_ids" : report.duplicate_trip_ids,
        "ingestion.pass_rate" : round(report.pass_rate,4),
    })
    log.info("log ingestion metrics completed")

def log_eda_summary(summary : Dict[str, Any]) -> None:
    mlflow.set_tag("pipeline.stage", "eda")
    eda_stats = summary.get("descriptive_stats", {}).get(config.TARGET_COLUMN, {})
    metrics = {
        "eda.n_rows" : summary.get("n_rows", 0),
        "eda.target_mean" : round(eda_stats.get("mean", 0),4),
        'eda.target_std' : round(eda_stats.get("std", 0),4),
        "eda.target_min" : round(eda_stats.get("min", 0),4),
        "eda.target_max" : round(eda_stats.get("max", 0),4),
        "eda.target_skewness" : round(eda_stats.get("skewness", 0),4),
        "eda.target_kurtosis" : round(eda_stats.get("kurtosis", 0),4)
    }
    mlflow.log_metrics(metrics)
    log.info("log_eda_summary completed")

def log_preprocessing_params(
    version: str,
    scaler_type: str,
    encoder_type: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    sha256: str,
) -> None:
    """Log preprocessing configuration and dataset metadata to MLflow."""

    mlflow.set_tag("stage", "preprocessing")

    mlflow.log_params(
        {
            "dataset.version": version,
            "dataset.sha256": sha256,
            "dataset.n_train": n_train,
            "dataset.n_val": n_val,
            "dataset.n_test": n_test,
            "dataset.n_features": n_features,
            "preprocessing.scaler": scaler_type,
            "preprocessing.encoder": encoder_type,
            "preprocessing.test_split": config.TEST_SIZE,
            "preprocessing.val_split": config.VAL_SIZE,
        }
    )

    log.info("Logged preprocessing params to MLflow.")


def log_dataset_artifact(version: str) -> None:
    """Log the entire versioned dataset folder as an MLflow artifact."""

    versioned_dir = Path(config.VERSIONED_DIR) / version

    if versioned_dir.exists():
        mlflow.log_artifacts(
            str(versioned_dir),
            artifact_path=f"dataset/{version}",
        )
        log.info(
            "Logged dataset artifacts for version '%s' to MLflow.",
            version,
        )
    else:
        log.warning(
            "Versioned dataset directory not found: %s",
            versioned_dir,
        )


def log_eda_artifacts() -> None:
    """Log all EDA PNG plots and summary JSON as MLflow artifacts."""

    eda_dir = Path(config.REPORT_DIR) / "eda"

    if eda_dir.exists():
        mlflow.log_artifacts(
            str(eda_dir),
            artifact_path="eda",
        )
        log.info("Logged EDA artifacts to MLflow.")
    else:
        log.warning("EDA directory not found: %s", eda_dir)


# ---------------------------------------------------------------------------
# Model Registry helpers
# ---------------------------------------------------------------------------

def register_model(
    run_id: str,
    model_artifact_path: str = "model",
    registered_name: str = config.MLFLOW_REGISTERED_MODEL,
) -> str:
    """
    Register a model from an existing MLflow run into the Model Registry.

    Returns:
        The registered model version as a string.
    """

    model_uri = f"runs:/{run_id}/{model_artifact_path}"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=registered_name,
    )

    version = str(result.version)

    log.info(
        "Model registered: name='%s' version=%s from run=%s",
        registered_name,
        version,
        run_id,
    )

    return version


def promote_to_production(
    registered_name: str = config.MLFLOW_REGISTERED_MODEL,
    version: Optional[str] = None,
    archive_existing: bool = True,
) -> None:
    """
    Transition the given (or latest) model version to Production.

    Archives any existing Production version when archive_existing=True.

    Note:
        This uses the legacy stage API, which works with MLflow 2.x
        model registries that support model stages.
    """

    client = MlflowClient()

    if version is None:
        versions = client.get_latest_versions(registered_name)

        if not versions:
            raise RuntimeError(
                f"No versions found for model '{registered_name}'."
            )

        version = str(max(int(v.version) for v in versions))

    if archive_existing:
        try:
            prod_versions = client.get_latest_versions(
                registered_name,
                stages=["Production"],
            )

            for pv in prod_versions:
                client.transition_model_version_stage(
                    name=registered_name,
                    version=pv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

                log.info(
                    "Archived previous Production model version %s.",
                    pv.version,
                )

        except Exception as exc:
            # Do not hide the promotion failure, but allow the archive
            # lookup to fail when the registry/backend does not support it.
            log.warning(
                "Could not archive existing Production versions: %s",
                exc,
            )

    client.transition_model_version_stage(
        name=registered_name,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )

    log.info(
        "Model '%s' version %s promoted to Production.",
        registered_name,
        version,
    )


def get_production_model(
    registered_name: str = config.MLFLOW_REGISTERED_MODEL,
):
    """
    Load the current Production model from the registry as a sklearn model.

    Safe to call at inference time.
    """

    model_uri = f"models:/{registered_name}/Production"

    model = mlflow.sklearn.load_model(model_uri)

    log.info(
        "Loaded Production model from registry: %s",
        model_uri,
    )

    return model
