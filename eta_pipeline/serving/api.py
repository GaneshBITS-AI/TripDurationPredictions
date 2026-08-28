"""
ETA Prediction REST API
========================

FastAPI application that exposes the trained Production model as an HTTP service.

Endpoints
---------
GET  /health       - liveness check; returns model version info
GET  /model-info   - detailed model + dataset metadata
POST /predict      - accept trip details, return predicted ETA

Quick start
-----------
    python serve.py
    # or
    uvicorn eta_pipeline.serving.api:app --host 0.0.0.0 --port 8000 --reload

Example request
----------------
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
            "pickup_datetime":  "2016-06-01 08:30:00",
            "pickup_lat":       40.7614,
            "pickup_lon":       -73.9776,
            "dropoff_lat":      40.6413,
            "dropoff_lon":      -73.7781,
            "passenger_count":  1,
            "vendor_id":        2,
            "store_and_fwd_flag": "N"
          }'
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from eta_pipeline import config
from eta_pipeline.logger import get_logger
from eta_pipeline.serving.predictor import TripInput, get_predictor

log = get_logger(__name__)

# ---------------------------- App setup ----------------------------

app = FastAPI(
    title="ETA Prediction API",
    description=(
        "Predict ride / delivery ETA from NYC Taxi trip details. "
        "Model served from the MLflow Model Registry (Production stage)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_error: str | None = None

# Eagerly warm up the predictor so the first request isn't slow
@app.on_event("startup")
async def _warmup():
    global _startup_error
    try:
        get_predictor()
        _startup_error = None
        log.info("Predictor warm-up complete.")
    except Exception as exc:
        _startup_error = str(exc)
        log.error(
            f"Predictor warm-up failed: {exc}\n"
            "  -> Run `python main.py` to train a model, then restart the server."
        )

# ---------------------- Request / Response schemas ----------------------

class PredictRequest(BaseModel):
    pickup_datetime: str = Field(
        ...,
        description="Trip pickup time. ISO-8601 or 'YYYY-MM-DD HH:MM:SS'.",
        examples=["2016-06-01 08:30:00"],
    )
    pickup_lat: float = Field(
        ..., ge=40.50, le=40.92, description="Pickup latitude (NYC range)."
    )
    pickup_lon: float = Field(
        ..., ge=-74.26, le=-73.68, description="Pickup longitude (NYC range)."
    )
    dropoff_lat: float = Field(
        ..., ge=40.50, le=40.92, description="Dropoff latitude (NYC range)."
    )
    dropoff_lon: float = Field(
        ..., ge=-74.26, le=-73.68, description="Dropoff longitude (NYC range)."
    )
    passenger_count: int = Field(
        default=1, ge=1, le=6, description="Number of passengers (1-6)."
    )
    vendor_id: int = Field(
        default=1, ge=1, le=2, description="Taxi vendor ID (1 or 2)."
    )
    store_and_fwd_flag: str = Field(
        default="N", description="Store-and-forward flag: 'Y' or 'N'."
    )
    dropoff_datetime: Optional[str] = Field(
        default=None, description="Optional dropoff time (informational only)."
    )
    avg_temp_f: Optional[float] = Field(
        default=None,
        description="Average daily temperature (\u00b0F). Auto-looked-up from weather data if omitted.",
    )
    precipitation: Optional[float] = Field(
        default=None, ge=0.0, description="Daily precipitation (inches). Auto-looked-up if omitted."
    )
    snow_depth: Optional[float] = Field(
        default=None, ge=0.0, description="Snow depth (inches). Auto-looked-up if omitted."
    )

    @field_validator("store_and_fwd_flag")
    @classmethod
    def _validate_flag(cls, v: str) -> str:
        # NOTE: this photo of the screen was obstructed by glare, so this
        # method body is reconstructed to match the style of the validator
        # below rather than transcribed directly - please double check it
        # against your original file.
        v = (v or "N").strip().upper()
        if v not in {"Y", "N"}:
            raise ValueError("store_and_fwd_flag must be 'Y' or 'N'.")
        return v

    @field_validator("dropoff_datetime")
    @classmethod
    def _validate_datetime(cls, v):
        if v is None:
            return v
        try:
            datetime.fromisoformat(str(v).replace(" ", "T"))
            return str(v)
        except ValueError:
            raise ValueError(f"Invalid datetime format: '{v}'. Use 'YYYY-MM-DD HH:MM:SS'.")


class PredictResponse(BaseModel):
    eta_minutes:     float = Field(..., description="Predicted ETA in minutes.")
    eta_seconds:     int   = Field(..., description="Predicted ETA in seconds.")
    model_name:      str   = Field(..., description="Registered model name.")
    model_version:   str   = Field(..., description="MLflow model registry version.")
    dataset_version: str   = Field(..., description="Training dataset version used.")
    input_features:  int   = Field(..., description="Number of features fed to the model.")
    latency_ms:      float = Field(..., description="Server-side inference latency (ms).")


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    dataset_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    dataset_version: str
    input_features: int
    metadata: dict


# --------------------------- Endpoints ---------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """Liveness check. Returns model identity without running inference."""
    if _startup_error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded: {_startup_error} | "
                "Run `python main.py` to train, then restart the server."
            ),
        )
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="ok",
            model_name=predictor._metadata.get("model_name", config.MLFLOW_REGISTERED_MODEL),
            model_version=str(predictor.model_version),
            dataset_version=predictor._metadata.get("version", "unknown"),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Operations"])
def model_info():
    """Returns detailed metadata about the loaded model and training dataset."""
    try:
        predictor = get_predictor()
        return ModelInfoResponse(
            model_name=predictor._metadata.get("model_name", "eta-best-model"),
            model_version=str(predictor.model_version),
            dataset_version=predictor._metadata.get("version", "unknown"),
            input_features=len(predictor.feature_columns),
            metadata=predictor._metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Predict ETA for a single NYC taxi trip.

    Applies the same feature-engineering pipeline used at training time
    (time features, haversine distance, bearing, cyclical encodings),
    then runs the Production model from the MLflow registry.
    """
    if _startup_error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded: {_startup_error} | "
                "Run `python main.py` to train, then restart the server."
            ),
        )

    t0 = time.perf_counter()

    trip = TripInput(
        pickup_datetime=request.pickup_datetime,
        pickup_lat=request.pickup_lat,
        pickup_lon=request.pickup_lon,
        dropoff_lat=request.dropoff_lat,
        dropoff_lon=request.dropoff_lon,
        passenger_count=request.passenger_count,
        vendor_id=request.vendor_id,
        store_and_fwd_flag=request.store_and_fwd_flag,
        dropoff_datetime=request.dropoff_datetime,
        avg_temp_f=request.avg_temp_f,
        precipitation=request.precipitation,
        snow_depth=request.snow_depth,
    )

    try:
        predictor = get_predictor()
        result    = predictor.predict(trip)
    except Exception as exc:
        log.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    result["latency_ms"] = latency_ms

    log.info(
        f"Predicted ETA={result['eta_minutes']} min "
        f"({latency_ms} ms) "
        f"pickup=({request.pickup_lat},{request.pickup_lon}) "
        f"dropoff=({request.dropoff_lat},{request.dropoff_lon})"
    )

    return PredictResponse(**result)


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(requests: list[PredictRequest]):
    """
    Predict ETA for multiple trips in a single HTTP call (max 100).
    Returns a list of predictions in the same order as the input.
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds limit of 100 trips per request.",
        )

    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    results = []
    for req in requests:
        t0 = time.perf_counter()
        trip = TripInput(
            pickup_datetime=req.pickup_datetime,
            pickup_lat=req.pickup_lat,
            pickup_lon=req.pickup_lon,
            dropoff_lat=req.dropoff_lat,
            dropoff_lon=req.dropoff_lon,
            passenger_count=req.passenger_count,
            vendor_id=req.vendor_id,
            store_and_fwd_flag=req.store_and_fwd_flag,
            dropoff_datetime=req.dropoff_datetime,
        )
        try:
            result = predictor.predict(trip)
        except Exception as exc:
            result = {"error": str(exc)}

        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        results.append(result)

    return {"predictions": results, "count": len(results)}