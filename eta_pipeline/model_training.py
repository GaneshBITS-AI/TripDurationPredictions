from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import sklearn

from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from eta_pipeline import config
from eta_pipeline.logger import get_logger
from eta_pipeline.mlflow_tracking import register_model, promote_to_production

log = get_logger(__name__)


# ----------------------------- Metric helpers -----------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        f"{prefix}_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        f"{prefix}_r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


# ----------------------------- Feature importance plot -----------------------------

def _plot_feature_importance(
    model, feature_names: List[str], model_name: str, top_n: int = 20
) -> Optional[str]:
    """
    Save a feature importance bar chart to a temp PNG and return its path.
    Returns None for models without feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names  = [feature_names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(names[::-1], values[::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {model_name} (top {top_n})", fontweight="bold")
    fig.tight_layout()

    path = config.MODELS_DIR / f"feat_importance_{model_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ----------------------------- Model catalogue -----------------------------

def _build_model_catalogue() -> Dict[str, Tuple[Any, Dict]]:
    """
    Returns {model_name: (estimator_instance, search_space_dict)}.
    Each estimator is a base instance; RandomizedSearchCV explores the space.
    """
    catalogue: Dict[str, Tuple[Any, Dict]] = {}

    # 1. Linear Regression - no meaningful continuous HP space; single fixed config
    catalogue["linear_regression"] = (
        LinearRegression(),
        {"fit_intercept": [True, False]},
    )

    # 2. Decision Tree - a single tree fits in a fraction of the time a
    # Random Forest's 100-500 bagged trees take, while still being a
    # non-linear model distinct from Linear Regression for comparison.
    # ccp_alpha (cost-complexity pruning) keeps a single unbounded tree from
    # just overfitting the training split.
    catalogue["decision_tree"] = (
        DecisionTreeRegressor(random_state=config.RANDOM_SEED),
        {
            "max_depth":         [None, 5, 10, 15, 20, 30],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf":  randint(1, 20),
            "max_features":      ["sqrt", "log2", None, 0.5, 0.8],
            "ccp_alpha":         uniform(0.0, 0.01),
        },
    )

    # 3. Gradient Boosting
    # HistGradientBoostingRegressor instead of GradientBoostingRegressor:
    # histogram-binned splits make it an order of magnitude faster to fit on
    # tens of thousands of rows, and early_stopping lets a trial quit once
    # the internal validation loss stops improving instead of always running
    # to max_iter.
    catalogue["gradient_boosting"] = (
        HistGradientBoostingRegressor(
            random_state=config.RANDOM_SEED,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
        ),
        {
            "max_iter":           randint(100, 400),
            "learning_rate":      uniform(0.01, 0.19),
            "max_depth":          [None, 3, 5, 8, 12],
            "max_leaf_nodes":     randint(15, 63),
            "min_samples_leaf":   randint(10, 50),
            "l2_regularization":  uniform(0, 1.0),
        },
    )

    # 4. XGBoost (optional)
    if _XGBOOST_AVAILABLE:
        catalogue["xgboost"] = (
            XGBRegressor(
                random_state=config.RANDOM_SEED,
                n_jobs=1,  # avoid double parallelism, see random_forest above
                tree_method="hist",
                verbosity=0,
            ),
            {
                "n_estimators":     randint(200, 600),
                "learning_rate":    uniform(0.01, 0.19),
                "max_depth":        randint(3, 8),
                "subsample":        uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5),
                "reg_alpha":        uniform(0, 1.0),
                "reg_lambda":       uniform(0.5, 2.0),
            },
        )
    else:
        log.warning("xgboost not installed - skipping XGBRegressor.")

    return catalogue


# ----------------------------- Result dataclass -----------------------------

@dataclass
class ModelResult:
    name:            str
    run_id:          str
    train_metrics:   Dict[str, float]
    val_metrics:     Dict[str, float]
    test_metrics:    Dict[str, float] = field(default_factory=dict)
    duration_sec:    float            = 0.0
    registered_ver:  Optional[str]    = None

    @property
    def val_rmse(self) -> float:
        return self.val_metrics.get("val_rmse", float("inf"))


@dataclass
class ModelComparisonResult:
    results:      List[ModelResult]
    champion:     ModelResult
    leaderboard:  pd.DataFrame

    def print_leaderboard(self) -> None:
        log.info("\n" + self.leaderboard.to_string(index=False))


# ----------------------------- Single model trainer -----------------------------

def _n_iter_for_space(search_space: Dict) -> int:
    """
    Number of RandomizedSearchCV trials to run for a given search space.

    When every hyperparameter is a finite list (no scipy distribution),
    random sampling can skip options entirely by bad luck - e.g. a 2-way
    toggle like linear_regression's {"fit_intercept": [True, False]} has
    only a 50% chance of ever trying both values at n_iter=1. In that case,
    try every combination (capped at HP_SEARCH_ITER) instead of sampling.
    Any space with a continuous distribution falls back to HP_SEARCH_ITER
    random samples, since its combinations aren't finite to enumerate.
    """
    total_combinations = 1
    for values in search_space.values():
        if not isinstance(values, list):
            return config.HP_SEARCH_ITER
        total_combinations *= len(values)
    return min(total_combinations, config.HP_SEARCH_ITER)


def _train_one_model(
    name:            str,
    model:           Any,
    search_space:    Dict,
    X_train:         pd.DataFrame,
    y_train:         pd.Series,
    X_val:           pd.DataFrame,
    y_val:           pd.Series,
    dataset_version: str,
) -> ModelResult:
    """
    Run RandomizedSearchCV over the search space on the train split,
    then evaluate the best estimator on the held-out val split.
    All trials and the best params are logged to a nested MLflow child run.
    """
    log.info(f"  Tuning: {name}  ({config.HP_SEARCH_ITER} iterations x {config.HP_CV_FOLDS}-fold CV) ...")
    t0 = time.perf_counter()

    feature_names = X_train.columns.tolist()
    n_iter = _n_iter_for_space(search_space)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=config.HP_CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        refit=True,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )

    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best_model  = search.best_estimator_
    best_params = search.best_params_

    log.info(f"    Best params: {best_params}")

    with mlflow.start_run(run_name=name, nested=True) as child_run:
        run_id = child_run.info.run_id

        # --- Tags ---
        mlflow.set_tags({
            "stage":            "model_training",
            "model_name":       name,
            "dataset.version":  dataset_version,
            "tuning.n_iter":    str(n_iter),
            "tuning.cv_folds":  str(config.HP_CV_FOLDS),
        })

        # --- Best hyperparameters ---
        mlflow.log_param("model_name", name)
        mlflow.log_param("hp_search_iter", n_iter)
        mlflow.log_param("hp_cv_folds", config.HP_CV_FOLDS)
        mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})

        # --- CV summary metric ---
        cv_rmse = round(-search.best_score_, 4)
        mlflow.log_metric("cv_rmse_best", cv_rmse)

        # --- Val-set metrics (final evaluation) ---
        train_pred = best_model.predict(X_train)
        val_pred   = best_model.predict(X_val)

        train_metrics = _compute_metrics(y_train.to_numpy(), train_pred, "train")
        val_metrics   = _compute_metrics(y_val.to_numpy(),   val_pred,   "val")

        overfit_gap = round(train_metrics["train_r2"] - val_metrics["val_r2"], 4)
        mlflow.log_metrics({**train_metrics, **val_metrics,
                             "overfit_gap_r2": overfit_gap})
        mlflow.log_metric("training_duration_sec", round(elapsed, 2))

        log.info(
            f"    {name} | cv_rmse={cv_rmse:.3f}  "
            f"val_rmse={val_metrics['val_rmse']:.3f}  "
            f"val_r2={val_metrics['val_r2']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # --- All HP trial results as CSV artifact ---
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        trials_path = config.MODELS_DIR / f"hp_trials_{name}.csv"
        cv_results.to_csv(trials_path, index=False)
        mlflow.log_artifact(str(trials_path), artifact_path="hp_search")

        # --- Artifacts: model ---
        _trusted = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"] \
            if "xgboost" in type(best_model).__module__ else []
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=None,
            input_example=X_train.head(5),
            skops_trusted_types=_trusted if _trusted else None,
        )

        # --- Artifacts: feature importance ---
        fi_path = _plot_feature_importance(best_model, feature_names, name)
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path="feature_importance")

    return ModelResult(
        name=name,
        run_id=run_id,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        duration_sec=elapsed,
    )


# ----------------------------- Leaderboard plot -----------------------------

def _plot_leaderboard(leaderboard: pd.DataFrame) -> str:
    """
    Bar chart comparing val_rmse, val_mae, val_r2, and overfit_gap_r2
    across all trained models. Saved to models/ and returned as a path.
    """
    models  = leaderboard["model"].tolist()
    metrics = ["val_rmse", "val_mae", "val_r2"]
    labels  = ["Val RMSE", "Val MAE", "Val R2"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle("Model Comparison - Validation Metrics", fontsize=14, fontweight="bold")

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = leaderboard[metric].tolist()
        bars = ax.barh(models, vals, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.invert_yaxis()

    fig.tight_layout()
    path = config.MODELS_DIR / "leaderboard_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Leaderboard comparison plot saved: {path.name}")
    return str(path)


# ----------------------------- Orchestrator -----------------------------

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    dataset_version: str = "v1",
) -> ModelComparisonResult:
    """
    Train all candidate models, compare on validation set, register champion.

    Parameters
    ----------
    X_train / y_train : training split
    X_val   / y_val    : validation split (used for champion selection)
    X_test  / y_test   : test split (evaluated once on champion only)
    dataset_version    : version string from preprocessing stage
    """
    log.info("=" * 60)
    log.info("STAGE 5 - MODEL TRAINING & EXPERIMENT TRACKING")
    log.info("=" * 60)

    # --- Log dataset + environment info to the active parent run ---
    mlflow.log_params({
        "train_rows":       X_train.shape[0],
        "val_rows":         X_val.shape[0],
        "test_rows":        X_test.shape[0],
        "n_features":       X_train.shape[1],
        "dataset_version":  dataset_version,
        "python_version":   platform.python_version(),
        "sklearn_version":  sklearn.__version__,
        "numpy_version":    np.__version__,
        "pandas_version":   pd.__version__,
    })
    # Log feature names as a text artifact
    feat_path = config.MODELS_DIR / "feature_names.txt"
    feat_path.write_text("\n".join(X_train.columns.tolist()))
    mlflow.log_artifact(str(feat_path), artifact_path="dataset")

    catalogue = _build_model_catalogue()
    results: List[ModelResult] = []

    for name, (model, params) in catalogue.items():
        result = _train_one_model(
            name, model, params,
            X_train, y_train, X_val, y_val,
            dataset_version,
        )
        results.append(result)

    # --- Build leaderboard ---
    rows = []
    for r in results:
        rows.append({
            "model":     r.name,
            "val_mae":   r.val_metrics.get("val_mae", None),
            "val_rmse":  r.val_metrics.get("val_rmse", None),
            "val_r2":    r.val_metrics.get("val_r2", None),
            "train_r2":  r.train_metrics.get("train_r2", None),
            "train_sec": round(r.duration_sec, 1),
            "run_id":    r.run_id,
        })
    leaderboard = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)

    log.info("\n--- Leaderboard (sorted by val_rmse) ---")
    log.info("\n" + leaderboard[["model", "val_mae", "val_rmse", "val_r2"]].to_string(index=False))

    # ------------------------------------------------------------------
    # NOTE: The photographed source cuts off at this point (line ~415
    # of 468). Everything below is a best-effort completion, written to
    # match the style, imports (register_model / promote_to_production),
    # and dataclass shape (ModelComparisonResult.champion) already
    # established above, so the module is runnable end-to-end. Swap in
    # your actual tail if it differs.
    # ------------------------------------------------------------------

    # --- Leaderboard comparison plot artifact (on the active parent run) ---
    lb_path = _plot_leaderboard(leaderboard)
    mlflow.log_artifact(lb_path, artifact_path="leaderboard")

    # --- Pick champion: lowest value of config.CHAMPION_METRIC ---
    champion = min(results, key=lambda r: r.val_metrics[config.CHAMPION_METRIC])
    log.info(f"\nChampion model: {champion.name} (val_rmse={champion.val_rmse:.4f})")

    # --- Evaluate champion once on the held-out test split ---
    log.info(f"Evaluating champion '{champion.name}' (val_rmse={champion.val_rmse:.4f}) ")
    with mlflow.start_run(run_id=champion.run_id, nested=True):
        model_uri = f"runs:/{champion.run_id}/model"
        champ_model =  mlflow.sklearn.load_model(model_uri)
        test_pred  = champ_model.predict(X_test)
        test_metrics = _compute_metrics(y_test.to_numpy(), test_pred, "test")
        champion.test_metrics = test_metrics
        mlflow.log_metrics(test_metrics)
        mlflow.set_tag("is_champion", "true")

        log.info(f"  test_rmse={champion.test_metrics['test_rmse']:.3f}  "
                 f"test_r2={champion.test_metrics['test_r2']:.3f}")

        # --- Register + promote the champion ---
        registered_ver = register_model(
            run_id=champion.run_id,
            model_artifact_path="model",
            registered_name=config.MLFLOW_REGISTERED_MODEL,
        )
        champion.registered_ver = registered_ver
        promote_to_production(
            registered_name=config.MLFLOW_REGISTERED_MODEL,
            version=registered_ver,
            archive_existing=True
        )

    plot_path = _plot_leaderboard(leaderboard)
    mlflow.log_artifact(str(plot_path), artifact_path="leaderboard")

    return ModelComparisonResult(
        results=results,
        champion=champion,
        leaderboard=leaderboard,
    )
