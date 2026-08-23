"""
data_versioning.py
--------------------
Week 1 / Module M2: Version the processed dataset.

Lightweight, dependency-free versioning: each pipeline run writes a
timestamped Parquet snapshot into data/versions/ and appends an entry to a
JSON manifest recording row/column counts, a content hash, the validation
report, and the config used. This gives reproducibility and an audit trail
without requiring DVC/MLflow to be installed for a course project -- but the
manifest is structured so it maps cleanly onto DVC/MLflow later if needed.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Stable content hash so identical datasets are easy to recognize."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:16]


def save_version(
    df: pd.DataFrame,
    versions_dir: str,
    manifest_path: str,
    validation_report: dict,
    source_description: str,
) -> dict:
    """
    Saves a Parquet snapshot of `df` and appends metadata to the manifest.

    Returns the manifest entry that was written.
    """
    os.makedirs(versions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    content_hash = _hash_dataframe(df)
    version_id = f"v_{timestamp}_{content_hash}"
    file_path = os.path.join(versions_dir, f"{version_id}.parquet")

    df.to_parquet(file_path, index=False)

    entry = {
        "version_id": version_id,
        "timestamp_utc": timestamp,
        "file_path": os.path.relpath(file_path, start=os.path.dirname(manifest_path)),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "content_hash": content_hash,
        "source_description": source_description,
        "validation_report": validation_report,
    }

    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    manifest.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved dataset version %s (%d rows) to %s", version_id, len(df), file_path)
    return entry


def latest_version(manifest_path: str) -> dict | None:
    """Convenience helper for downstream weeks (Week 2 training) to fetch the latest dataset version."""
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return manifest[-1] if manifest else None
