from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from eta_pipeline import config
from eta_pipeline.logger import get_logger

log = get_logger(__name__)

CATEGORICAL_ENCODER = "ordinal"  # "ordinal" | "onehot"


# ----------------------------- Scaler factory -----------------------------

def _build_scaler(scaler_type: str):
    options = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if scaler_type not in options:
        raise ValueError(f"Unknown scaler '{scaler_type}'. Choose from {list(options)}.")
    return options[scaler_type]


# ----------------------------- Column selection -----------------------------

EXCLUDE_COLS = {
    config.TARGET_COLUMN,
    "trip_id",
    "pickup_datetime",
    "dropoff_datetime",
    "trip_duration_sec",
    "part_of_day",       # string ordinal - not used directly as a feature
}


def _select_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) present in df, excluding non-feature cols."""
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in EXCLUDE_COLS
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in EXCLUDE_COLS
    ]
    return num_cols, cat_cols


# ----------------------------- Versioning -----------------------------

def _compute_sha256(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _next_version(versioned_dir: Path, prefix: str = config.DATASET_VERSION_PREFIX) -> str:
    existing = [p.name for p in versioned_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    nums = []
    for e in existing:
        try:
            nums.append(int(e.replace(prefix, "")))
        except ValueError:
            pass
    return f"{prefix}{(max(nums) + 1) if nums else 1}"


def _persist_version(
    version: str,
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,     y_val: pd.Series,     y_test: pd.Series,
    scaler, encoder,
    num_cols: list[str], cat_cols: list[str], sha256: str,
) -> Path:
    ver_dir = config.VERSIONED_DIR / version
    ver_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X, y in [("train", X_train, y_train),
                              ("val",   X_val,   y_val),
                              ("test",  X_test,  y_test)]:
        out = X.copy()
        out[config.TARGET_COLUMN] = y.values
        out.to_parquet(ver_dir / f"{split_name}.parquet", index=False)

    with open(ver_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    if encoder is not None:
        with open(ver_dir / "encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

    meta = {
        "version":             version,
        "sha256":               sha256,
        "created_at":           datetime.now().isoformat(),
        "n_train":              len(X_train),
        "n_val":                len(X_val),
        "n_test":               len(X_test),
        "numeric_features":     num_cols,
        "categorical_features": cat_cols,
        "scaler_type":          config.SCALER_TYPE,
        "encoder_type":         CATEGORICAL_ENCODER,
        "target":               config.TARGET_COLUMN,
        "pipeline_order":       "split -> feature_engineering -> encode -> scale",
    }
    with open(ver_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Dataset version '{version}' persisted at {ver_dir}")
    return ver_dir


# ----------------------------- Public API -----------------------------

def preprocess_and_version(
    clean_df:      pd.DataFrame,
    encoder_type:  str   = CATEGORICAL_ENCODER,
    scaler_type:   str   = config.SCALER_TYPE,
    test_size:     float = config.TEST_SIZE,
    val_size:      float = config.VAL_SIZE,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,    pd.Series,    pd.Series,
    object,       object,       str
]:
    """
    Split -> Feature Engineer -> Encode -> Scale -> Version.

    Parameters
    ----------
    clean_df      : validated clean DataFrame from ingestion (raw features only)
    encoder_type  : "ordinal" | "onehot"
    scaler_type   : "standard" | "minmax" | "robust"
    test_size     : fraction held out for final test evaluation
    val_size      : fraction of remaining data used for validation

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test,
    fitted_scaler, fitted_encoder, dataset_version_string
    """
    from eta_pipeline.feature_engineering import build_features

    log.info("=" * 60)
    log.info("PREPROCESSING PIPELINE  (split -> engineer -> encode -> scale)")
    log.info("=" * 60)

    if config.TARGET_COLUMN not in clean_df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found.")

    sha256 = _compute_sha256(clean_df)
    log.info(f"Input dataset SHA-256: {sha256}")

    # --- STEP 1: Raw split on CLEAN data (before any feature engineering) ------
    y_raw = clean_df[config.TARGET_COLUMN].astype("float64")
    y_bin = pd.qcut(y_raw, q=5, labels=False, duplicates="drop")

    df_temp, df_test, _, _, yb_temp, _ = train_test_split(
        clean_df, y_raw, y_bin,
        test_size=test_size,
        random_state=config.RANDOM_SEED,
        stratify=y_bin,
    )

    val_frac_adj = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_frac_adj,
        random_state=config.RANDOM_SEED,
        stratify=yb_temp,
    )

    log.info(
        f"Raw split sizes -> train: {len(df_train):,} | "
        f"val: {len(df_val):,} | test: {len(df_test):,}"
    )

    # --- STEP 2: Feature engineering applied independently to each split ------
    log.info("Applying feature engineering to train split ...")
    df_train = build_features(df_train)
    log.info("Applying feature engineering to val split ...")
    df_val   = build_features(df_val)
    log.info("Applying feature engineering to test split ...")
    df_test  = build_features(df_test)

    # --- STEP 3: Separate target after engineering ------------------------
    y_train = df_train.pop(config.TARGET_COLUMN).astype("float64")
    y_val   = df_val.pop(config.TARGET_COLUMN).astype("float64")
    y_test  = df_test.pop(config.TARGET_COLUMN).astype("float64")

    # Use train columns as the authoritative feature set
    num_cols, cat_cols = _select_feature_columns(df_train)
    log.info(f"Numeric features    : {len(num_cols)}")
    log.info(f"Categorical features: {len(cat_cols)}")

    X_train = df_train[num_cols + cat_cols].copy()
    X_val   = df_val[[c for c in num_cols + cat_cols if c in df_val.columns]].copy()
    X_test  = df_test[[c for c in num_cols + cat_cols if c in df_test.columns]].copy()

    # --- STEP 4: Encode categoricals (fit on train only) -------------------
    fitted_encoder = None
    present_cat = [c for c in cat_cols if c in X_train.columns]

    if present_cat:
        if encoder_type == "ordinal":
            fitted_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            X_train[present_cat] = fitted_encoder.fit_transform(X_train[present_cat])
            X_val[present_cat]   = fitted_encoder.transform(X_val[present_cat])
            X_test[present_cat]  = fitted_encoder.transform(X_test[present_cat])
            log.info(f"OrdinalEncoder fitted on train -> applied to val/test: {present_cat}")

        elif encoder_type == "onehot":
            fitted_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                drop="first",
            )
            ohe_train = fitted_encoder.fit_transform(X_train[present_cat])
            ohe_cols  = fitted_encoder.get_feature_names_out(present_cat).tolist()
            for X_split, raw_arr in [(X_train, ohe_train),
                                       (X_val,   fitted_encoder.transform(X_val[present_cat])),
                                       (X_test,  fitted_encoder.transform(X_test[present_cat]))]:
                X_split.drop(columns=present_cat, inplace=True)
                X_split[ohe_cols] = raw_arr
            log.info(f"OneHotEncoder: {len(ohe_cols)} dummy columns created from {present_cat}")

        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}'.")

    # --- STEP 5a: Impute NaNs in numeric cols (fit on train only) ----------
    present_num = [c for c in num_cols if c in X_train.columns]
    nan_counts = X_train[present_num].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
    if cols_with_nan:
        log.warning(
            f"NaNs in {len(cols_with_nan)} numeric col(s) before scaling - "
            f"imputing with median: {cols_with_nan}"
        )
        imputer = SimpleImputer(strategy="median")
        X_train[present_num] = imputer.fit_transform(X_train[present_num])
        X_val[present_num]   = imputer.transform(X_val[present_num])
        X_test[present_num]  = imputer.transform(X_test[present_num])
    else:
        imputer = None
        log.info("No NaNs in numeric features - imputation skipped.")

    # --- STEP 5b: Scale numerics (fit on train only) -----------------------
    fitted_scaler = _build_scaler(scaler_type)
    X_train[present_num] = fitted_scaler.fit_transform(X_train[present_num])
    X_val[present_num]   = fitted_scaler.transform(X_val[present_num])
    X_test[present_num]  = fitted_scaler.transform(X_test[present_num])
    log.info(f"{scaler_type.capitalize()}Scaler fitted on train -> applied to val/test: {len(present_num)} cols")

    # --- STEP 6: Persist versioned dataset ----------------------------------
    version = _next_version(config.VERSIONED_DIR)
    _persist_version(
        version,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        fitted_scaler, fitted_encoder,
        present_num, present_cat, sha256,
    )

    log.info("Preprocessing complete.")
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        fitted_scaler, fitted_encoder, version,
    )


# ----------------------------- Load versioned split -----------------------------

def load_versioned_dataset(version: str) -> Dict:
    """Reload a previously versioned dataset from disk."""
    ver_dir = config.VERSIONED_DIR / version
    if not ver_dir.exists():
        raise FileNotFoundError(f"Version '{version}' not found at {ver_dir}")

    result = {}
    for split in ["train", "val", "test"]:
        df_split = pd.read_parquet(ver_dir / f"{split}.parquet")
        result[f"y_{split}"] = df_split.pop(config.TARGET_COLUMN)
        result[f"X_{split}"] = df_split

    with open(ver_dir / "scaler.pkl", "rb") as f:
        result["scaler"] = pickle.load(f)

    enc_path = ver_dir / "encoder.pkl"
    result["encoder"] = pickle.load(open(enc_path, "rb")) if enc_path.exists() else None

    with open(ver_dir / "metadata.json") as f:
        result["metadata"] = json.load(f)

    log.info(f"Loaded versioned dataset '{version}' from {ver_dir}")
    return result