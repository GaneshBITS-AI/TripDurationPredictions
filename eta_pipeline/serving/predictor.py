"""
ETA Prediction — Model Loading & Inference
============================================

Loads the current Production model from the MLflow Model Registry plus the
scaler/encoder/feature-list from the matching versioned dataset, and turns a
single raw trip into a prediction using the same feature-engineering
pipeline used at training time (eta_pipeline.feature_engineering.build_features).
"""

from __future__ import annotations

import json
import pickle
import threading
from dataclasses import dataclass
from typing import Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from eta_pipeline import config
from eta_pipeline.feature_engineering import build_features
from eta_pipeline.logger import get_logger
from eta_pipeline.mlflow_tracking import get_production_model

log = get_logger(__name__)

# The training pipeline points MLflow at its SQLite registry via
# setup_mlflow(); the serving process never calls that, so without this the
# registry lookups below silently hit MLflow's default local store instead
# and "Registered Model ... not found" even when the model is registered.
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


@dataclass
class TripInput:
    pickup_datetime:     str
    pickup_lat:           float
    pickup_lon:           float
    dropoff_lat:          float
    dropoff_lon:          float
    passenger_count:      int              = 1
    vendor_id:            int              = 1
    store_and_fwd_flag:   str              = "N"
    dropoff_datetime:     Optional[str]    = None
    avg_temp_f:           Optional[float]  = None
    precipitation:        Optional[float]  = None
    snow_depth:           Optional[float]  = None

    def to_raw_frame(self) -> pd.DataFrame:
        """Build a one-row DataFrame with the raw columns build_features() expects."""
        row = {
            "pickup_datetime":    self.pickup_datetime,
            # dropoff_datetime is only used by validation, not by any feature -
            # a placeholder after pickup is fine when the caller doesn't know it yet.
            "dropoff_datetime":   self.dropoff_datetime or self.pickup_datetime,
            "pickup_lat":         self.pickup_lat,
            "pickup_lon":         self.pickup_lon,
            "dropoff_lat":        self.dropoff_lat,
            "dropoff_lon":        self.dropoff_lon,
            "passenger_count":    self.passenger_count,
            "vendor_id":          self.vendor_id,
            "store_and_fwd_flag": self.store_and_fwd_flag,
            # Weather defaults to 0 when the caller doesn't supply it, matching
            # merge_weather_with_trips()'s behaviour when no weather data is available.
            "avg_temp_f":         self.avg_temp_f if self.avg_temp_f is not None else 0.0,
            "precipitation":      self.precipitation if self.precipitation is not None else 0.0,
            "snow_depth":         self.snow_depth if self.snow_depth is not None else 0.0,
        }
        df = pd.DataFrame([row])
        df["pickup_datetime"]  = pd.to_datetime(df["pickup_datetime"])
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
        return df


def _latest_dataset_version() -> str:
    """Highest 'vN' folder under data/processed/versioned/ (same numbering as preprocessing._next_version)."""
    prefix = config.DATASET_VERSION_PREFIX
    versions = [
        p.name for p in config.VERSIONED_DIR.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    if not versions:
        raise FileNotFoundError(
            f"No versioned dataset found under {config.VERSIONED_DIR}. "
            "Run `python main.py` to train a model first."
        )
    return max(versions, key=lambda v: int(v[len(prefix):]))


class Predictor:
    """Loads the Production model + matching preprocessing artifacts once, then serves predictions."""

    def __init__(self):
        self.dataset_version = _latest_dataset_version()
        ver_dir = config.VERSIONED_DIR / self.dataset_version

        with open(ver_dir / "metadata.json") as f:
            self._metadata = json.load(f)

        with open(ver_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        enc_path = ver_dir / "encoder.pkl"
        self.encoder = pickle.load(open(enc_path, "rb")) if enc_path.exists() else None

        self.numeric_features     = list(self._metadata["numeric_features"])
        self.categorical_features = list(self._metadata["categorical_features"])
        self.feature_columns      = self.numeric_features + self.categorical_features

        self.model = get_production_model()

        self.model_version = "unknown"
        try:
            client = MlflowClient()
            prod = client.get_latest_versions(config.MLFLOW_REGISTERED_MODEL, stages=["Production"])
            if prod:
                self.model_version = prod[0].version
        except Exception as exc:
            log.warning(f"Could not resolve registry version number: {exc}")

        self._metadata["model_name"] = config.MLFLOW_REGISTERED_MODEL
        self._metadata["version"]    = self.dataset_version

        log.info(
            f"Predictor ready: model='{config.MLFLOW_REGISTERED_MODEL}' "
            f"version={self.model_version} dataset={self.dataset_version} "
            f"features={len(self.feature_columns)}"
        )

    def predict(self, trip: TripInput) -> dict:
        raw  = trip.to_raw_frame()
        feat = build_features(raw)

        missing = [c for c in self.feature_columns if c not in feat.columns]
        if missing:
            raise ValueError(f"Feature engineering did not produce expected columns: {missing}")

        X = feat[self.feature_columns].copy()

        if self.categorical_features and self.encoder is not None:
            X[self.categorical_features] = self.encoder.transform(X[self.categorical_features])

        if self.numeric_features:
            X[self.numeric_features] = X[self.numeric_features].fillna(0.0)
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])

        eta_minutes = float(self.model.predict(X)[0])
        eta_minutes = max(eta_minutes, 0.1)  # guard against a stray negative prediction

        return {
            "eta_minutes":     round(eta_minutes, 2),
            "eta_seconds":     int(round(eta_minutes * 60)),
            "model_name":      config.MLFLOW_REGISTERED_MODEL,
            "model_version":   str(self.model_version),
            "dataset_version": self.dataset_version,
            "input_features":  len(self.feature_columns),
        }


_predictor: Optional[Predictor] = None
_lock = threading.Lock()


def get_predictor() -> Predictor:
    """Lazily create and cache a single Predictor instance (thread-safe)."""
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                _predictor = Predictor()
    return _predictor
