# ETA Prediction Pipeline — Architecture Document

**Project:** NYC Taxi Trip Duration / ETA Prediction (Mini Project, PG AI & ML)
**Scope:** ingestion → validation → EDA → preprocessing/versioning → model training & tuning → experiment tracking & registry → serving (API + UI)

---

## 1. Problem & Approach

Predict trip ETA (`eta_minutes`) for an NYC taxi ride from pickup/dropoff location, time, passenger count, vendor, and weather. The pipeline is built as a sequence of independently runnable, independently testable stages rather than one monolithic script, each stage writing versioned artifacts the next stage (or the serving layer) consumes — so any stage can be re-run, inspected, or replaced without re-running the whole pipeline.

## 2. Pipeline Architecture

```
data/raw/{NYC.csv,train.csv}
        │
        ▼
┌─────────────────┐   validate schema, GPS bounds, timestamps,
│  1. Ingestion    │   ETA/passenger sanity, duplicates
│  ingestion.py    │   → clean_df, quarantine_df, ValidationReport
└─────────────────┘   (persisted as timestamped parquet + JSON)
        │
        ▼
┌─────────────────┐   target distribution, correlations, weather/
│  2. EDA          │   time-of-day patterns → 9 plots + summary_stats.json
│  eda.py          │   (optional: --skip-eda)
└─────────────────┘
        │
        ▼
┌─────────────────┐   split(stratified) → build_features (per split,
│  3. Preprocessing│   independently) → encode → impute → scale →
│  preprocessing.py│   version (data/processed/versioned/vN/)
└─────────────────┘
        │
        ▼
┌─────────────────┐   4 candidate models × RandomizedSearchCV →
│  4. Training     │   leaderboard → champion (val_rmse) → test eval →
│  model_training.py│  MLflow Model Registry (Production stage)
└─────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  5. Serving                   │
│  FastAPI (serve.py + api.py)  │◄── Streamlit UI (streamlit_app.py)
│  loads Production model +     │    thin HTTP client, no model
│  matching scaler/feature list │    loading of its own
└──────────────────────────────┘
```

Every stage is orchestrated by [`main.py`](main.py) inside one parent MLflow run, with each pipeline stage and each candidate model logged as a nested child run — so a single `python main.py` invocation produces one browsable experiment tree in the MLflow UI, not four unrelated runs.

## 3. Component Reference

| Stage | File | Responsibility |
|---|---|---|
| Ingestion | [`eta_pipeline/ingestion.py`](eta_pipeline/ingestion.py) | Load Kaggle NYC Taxi CSV, rename to internal schema, merge daily weather, validate row quality, split clean/quarantine |
| EDA | [`eta_pipeline/eda.py`](eta_pipeline/eda.py) | Descriptive stats + 9 diagnostic plots on the clean data |
| Preprocessing | [`eta_pipeline/preprocessing.py`](eta_pipeline/preprocessing.py) | Stratified split, per-split feature engineering, encode/impute/scale, dataset versioning |
| Feature engineering | [`eta_pipeline/feature_engineering.py`](eta_pipeline/feature_engineering.py) | Time cyclicals, haversine/bearing, grid cells, weather flags |
| Training | [`eta_pipeline/model_training.py`](eta_pipeline/model_training.py) | Model catalogue, hyperparameter search, leaderboard, champion selection, registry promotion |
| Tracking | [`eta_pipeline/mlflow_tracking.py`](eta_pipeline/mlflow_tracking.py) | MLflow setup, run helpers, registry register/promote |
| Serving — API | [`eta_pipeline/serving/api.py`](eta_pipeline/serving/api.py), [`predictor.py`](eta_pipeline/serving/predictor.py) | FastAPI app; loads Production model + preprocessing artifacts once, serves `/predict` |
| Serving — UI | [`streamlit_app.py`](streamlit_app.py) | Form-based client that calls the API over HTTP |
| Config | [`eta_pipeline/config.py`](eta_pipeline/config.py) | All thresholds, paths, and hyperparameter-search settings in one place |

---

## 4. Data Pipeline Design Decisions

| Decision | Value | Reasoning |
|---|---|---|
| GPS bounding box | lat `[40.50, 40.92]`, lon `[-74.26, -73.68]` | Tight NYC box — wide enough to cover all 5 boroughs + JFK/LGA, tight enough to reject clearly-erroneous pings (ocean, other states) that a looser box would let through |
| ETA bounds | `MIN_ETA_MINUTES=1`, `MAX_ETA_MINUTES=120` | Sub-minute trips are almost always meter/GPS glitches, not real rides; >2 hours is an outlier for a single continuous taxi trip, not a legitimate long fare |
| Passenger cap | `MAX_PASSENGER=6` | NYC TLC's legal passenger limit for a standard taxi |
| Max distance | `MAX_DISTANCE_KM=120` | Deliberately permissive — keeps legitimate long airport/out-of-city runs instead of truncating to an in-borough-only radius |
| Train/val/test split | 70/10/20 (`TEST_SIZE=0.20`, `VAL_SIZE=0.10` of remainder) | Val is spent freely comparing all 4 models (that's what it's for); test is touched exactly once, on the champion only, so it stays an honest, unbiased estimate of production performance |
| Split stratification | `pd.qcut(eta_minutes, q=5)` quantile bins | Trip duration is right-skewed (many short trips, a long tail of airport/outer-borough rides); stratifying on target quantiles keeps that skew consistent across all three splits instead of risking an unlucky split that over/under-represents long trips in val or test |
| Feature engineering timing | Applied **after** the split, independently per split | Any transform *fit* on data (scaler, encoder, bin edges) must never see val/test rows during fitting — row-wise features (haversine, cyclicals) are split-order-independent, but doing it uniformly after the split makes the no-leakage rule mechanical rather than case-by-case |
| Scaler | `StandardScaler` (`SCALER_TYPE="standard"`) | The feature set mixes near-normal continuous values (lat/lon deltas, haversine distance) with cyclical sin/cos terms already bounded to [-1, 1]; z-scoring is the safer general default across all 4 compared model families (matters most for `LinearRegression`'s coefficients) — `RobustScaler` would only be preferred if heavy outlier contamination remained *after* the validation-stage GPS/ETA bounds, which already strip most of it |
| Categorical encoding | `OrdinalEncoder`, configurable to one-hot via `--encoder` | Only `store_and_fwd_flag` is genuinely categorical text, and it's already binarized to 0/1 by `encode_store_flag()` before the encoder runs — in practice `categorical_features` ends up empty (confirmed in `metadata.json`). Kept configurable for when a true multi-category field (e.g. a real taxi-zone ID) is added |
| Dataset versioning | `data/processed/versioned/vN/` with `metadata.json` + `scaler.pkl` + parquet splits | Every `preprocess_and_version()` run gets its own immutable version folder; the serving layer picks up the *latest* version's scaler/feature-list, so training and serving are always looking at matching preprocessing state |

## 5. Model Training & Hyperparameter Reasoning

Four models are trained and compared every run — a **deliberately diverse catalogue** (plain linear, a single tree, and two different boosting implementations) rather than four variants of the same idea, so the leaderboard tells you something about which *kind* of model the problem rewards, not just which random seed got lucky.

Each tuned model runs `RandomizedSearchCV` with `n_iter=HP_SEARCH_ITER` × `cv=HP_CV_FOLDS` fits, scored on negative RMSE, refit on the full train split, then evaluated once on the held-out validation split.

### 5.1 Linear Regression — baseline floor

**Why included:** the simplest possible model. If a tuned ensemble can't beat it by a wide margin, the real gains in this problem come from feature engineering, not model sophistication — this is the sanity-check baseline everything else has to clear.

| Hyperparameter | Search space | Reasoning |
|---|---|---|
| `fit_intercept` | `[True, False]` | The only tuning knob a plain OLS model has (no regularization, no depth). |

> **Known issue** — because this space has only *one key* (`fit_intercept`), `_train_one_model()`'s `n_iter = HP_SEARCH_ITER if len(search_space) > 1 else 1` collapses to `n_iter=1`, meaning `RandomizedSearchCV` samples **one** random value out of `{True, False}` instead of trying both. In the latest run it happened to land on `fit_intercept=False`, producing a badly negative val R² (**-1.04**) — worse than predicting the mean — because the engineered feature set isn't centered. This makes the "baseline" comparison currently misleading rather than genuinely a floor. Fix: use `n_iter=min(HP_SEARCH_ITER, len(all combinations))` or just always try every value for discrete grids this small. Not fixed as part of this document — flagged for a follow-up.

### 5.2 Decision Tree — fast single-tree baseline

**Why included:** replaces what was originally a `RandomForestRegressor`. A Random Forest's 100–500 bagged trees per fit made it consistently the slowest model in the comparison; a single tree fits in a fraction of the time while still representing "non-linear, no boosting" in the leaderboard, so the ensembles below have something to prove they're worth their extra training cost against.

| Hyperparameter | Search space | Reasoning |
|---|---|---|
| `max_depth` | `[None, 5, 10, 15, 20, 30]` | Spans "unbounded" through shallow (likely underfit) to deep (likely overfit); mixing `None` with explicit caps lets the search directly compare "let it grow" against depth-based stopping. |
| `min_samples_split` | `randint(2, 20)` | Node-level pruning knob — wide range because a single tree has no bagging to average away variance, so this and `min_samples_leaf` do more regularization work than they would inside a forest. |
| `min_samples_leaf` | `randint(1, 20)` | Same reasoning as above, at the leaf level. |
| `max_features` | `["sqrt", "log2", None, 0.5, 0.8]` | Normally a bagging-ensemble decorrelation knob; here it doubles as injected randomness/regularization for a single tree, since restricting candidate split features per node also reduces overfitting to noisy columns. |
| `ccp_alpha` | `uniform(0.0, 0.01)` | Cost-complexity pruning — added specifically because one unbounded tree overfits far more aggressively than 100+ averaged trees would, so post-hoc pruning strength needed its own dedicated knob that a forest wouldn't need. |

**Latest run:** `max_depth=20, min_samples_leaf=17, min_samples_split=11, max_features=0.5, ccp_alpha≈0.0061` → val RMSE **5.56 min**, val R² **0.725**, trained in **28.8 s**.

### 5.3 Gradient Boosting (`HistGradientBoostingRegressor`)

**Why included:** represents *boosting* (sequential residual-fitting) as distinct from the tree/bagging family above. Originally `GradientBoostingRegressor` (sklearn's classic exact-split implementation); swapped to the histogram-binned variant — the same core algorithmic idea as LightGBM/XGBoost's `hist` mode — because exact-split boosting on tens of thousands of rows was the single slowest model in the original comparison.

| Hyperparameter | Search space | Reasoning |
|---|---|---|
| `max_iter` | `randint(100, 400)` | Analogous to `n_estimators`. Capped lower than XGBoost's range because `early_stopping` (below) already prunes wasted rounds — the ceiling only needs to cover cases where boosting keeps genuinely helping. |
| `learning_rate` | `uniform(0.01, 0.19)` | Standard boosting range, shared with XGBoost's space so the two boosters aren't confounded by different search spaces: too high oscillates/overshoots, too low needs more rounds than the budget allows. |
| `max_depth` | `[None, 3, 5, 8, 12]` | `None` lets `HistGradientBoostingRegressor` regularize purely through `max_leaf_nodes` (sklearn's documented preferred axis for this model), with explicit shallow→deep values still covering the classic boosting range. |
| `max_leaf_nodes` | `randint(15, 63)` | HistGB's primary complexity control; the sklearn default is 31, and this range brackets it on both sides. |
| `min_samples_leaf` | `randint(10, 50)` | Deliberately higher floor than the Decision Tree's `(1, 20)` — boosting fits residuals repeatedly, so small leaves let early rounds fit noise that later rounds then compound. |
| `l2_regularization` | `uniform(0, 1.0)` | HistGB's ridge-style penalty on leaf output values — the main defense against overfitting the residual signal round-over-round, since boosting (unlike bagging) has no built-in variance averaging. |

**Fixed (not searched):** `early_stopping=True, n_iter_no_change=15, validation_fraction=0.1` — each trial internally holds out 10% of its training fold and stops once validation loss hasn't improved for 15 rounds. This both auto-regularizes and is a large part of why the HistGB swap sped up training beyond just the histogram split-finding itself.

**Latest run:** `max_iter=379, learning_rate≈0.062, max_depth=12, max_leaf_nodes=40, min_samples_leaf=34, l2_regularization≈0.39` → val RMSE **4.73 min**, val R² **0.801**, trained in **125.5 s**.

### 5.4 XGBoost

**Why included:** the strongest, most widely-used boosting implementation for tabular data — kept as the fourth point of comparison specifically to see whether the extra dependency earns its keep over sklearn's own `HistGradientBoostingRegressor` on this dataset.

| Hyperparameter | Search space | Reasoning |
|---|---|---|
| `n_estimators` | `randint(200, 600)` | Wider ceiling than HistGB's `max_iter` because this search has no `early_stopping`/`eval_set` wired through `RandomizedSearchCV` here, so the range itself has to cover the useful boosting length. |
| `learning_rate` | `uniform(0.01, 0.19)` | Identical range to HistGB, deliberately, for a fair side-by-side. |
| `max_depth` | `randint(3, 8)` | Classic XGBoost range — below 3 underfits, above 8 overfits quickly on ~70k rows × 34 features. |
| `subsample` | `uniform(0.6, 0.4)` → `[0.6, 1.0]` | Row subsampling per round (stochastic gradient boosting) — reduces variance and speeds up each round. |
| `colsample_bytree` | `uniform(0.5, 0.5)` → `[0.5, 1.0]` | Column subsampling per tree — XGBoost's other main variance-reduction knob, independent of `subsample`. |
| `reg_alpha` (L1) | `uniform(0, 1.0)` | Searched alongside `reg_lambda` for combined L1+L2 leaf-weight regularization. |
| `reg_lambda` (L2) | `uniform(0.5, 2.0)` | Floor of 0.5 rather than 0 — XGBoost already defaults to `lambda=1`, so the search stays centered near a sane default instead of allowing "no regularization at all." |

**Fixed (not searched):** `tree_method="hist"` (histogram split-finding, same speed rationale as HistGB) and `n_jobs=1`. The latter is deliberate, not an oversight — the estimator used to be `n_jobs=-1` *inside* a `RandomizedSearchCV(n_jobs=-1)`, which oversubscribed the CPU (each of the search's parallel workers also spawning its own full thread pool). Fixed by parallelizing only at the search level.

**Latest run:** `n_estimators=509, learning_rate≈0.050, max_depth=6, subsample≈0.78, colsample_bytree≈0.71, reg_alpha≈0.031, reg_lambda≈2.18` → val RMSE **4.71 min**, val R² **0.803**, trained in **107.5 s** — the current champion by validation RMSE.

### 5.5 Search-budget decisions (apply to all tuned models)

| Setting | Value | Reasoning |
|---|---|---|
| `HP_SEARCH_ITER` | 12 | 12×3=36 fits per tuned model. This is a mini-project running on laptop-scale compute, not a production tuning job — 36 trials over each model's 4–7 dimensional space comfortably beats untuned defaults without turning a single `python main.py` run into a multi-hour job. (Reduced from an original 20 specifically to cut training time.) |
| `HP_CV_FOLDS` | 3 | With ~70k training rows, 3-fold still gives each fold ~46k rows — plenty stable for these model sizes — while keeping total fit count down. 5-fold would cost 67% more fits for a stability gain that isn't needed at this data size. |
| `RANDOM_SEED` | 42 | Fixed across every split, model constructor, and search, so re-running `main.py` on unchanged data reproduces the exact same leaderboard. |
| Scoring metric | `neg_root_mean_squared_error` | RMSE penalizes large errors more than MAE — appropriate here since a wildly wrong ETA (e.g. predicting 10 min for a 60 min airport run) is worse for a rider-facing product than being off by a minute on many short trips. |

### 5.6 Latest Leaderboard (dataset v5, 69,839 train / 9,977 val / 19,955 test rows, 34 features)

| Model | Val RMSE (min) | Val MAE (min) | Val R² | Train time |
|---|---|---|---|---|
| **XGBoost** (champion) | **4.71** | 3.05 | 0.803 | 107.5 s |
| Gradient Boosting (HistGB) | 4.73 | 3.05 | 0.801 | 125.5 s |
| Decision Tree | 5.56 | 3.70 | 0.725 | 28.8 s |
| Linear Regression | 15.13 | 13.92 | -1.04 | 7.9 s |

> **Known issue** — the champion is currently selected by `min(results, key=lambda r: r.val_metrics["val_mae"])` in `train_all_models()`, i.e. **lowest val MAE**, while `config.CHAMPION_METRIC = "val_rmse"` documents the intent as RMSE-based selection. On the latest run both metrics agree on XGBoost as champion, but the two are not guaranteed to agree in general — flagged as an inconsistency to fix (either change the `min()` key to `val_rmse`, or update `CHAMPION_METRIC` to reflect what the code actually does).

---

## 6. Experiment Tracking & Model Registry

- **Backend:** MLflow with a SQLite store (`sqlite:///mlruns/mlflow.db`) rather than a file store — SQLite is what enables the **Model Registry** (stage transitions, `Production`/`Archived`) on a local/Windows setup without standing up a full MLflow tracking server.
- **Run structure:** one parent run per `python main.py` invocation (tagged `pipeline.stage=orchestration`), with each of the 4 candidate models logged as a **nested child run** — keeps one pipeline execution browsable as a single tree in the MLflow UI instead of scattering unrelated runs.
- **Per-model artifacts logged:** best hyperparameters, CV/train/val metrics, all HP trial results (`hp_trials_<model>.csv`), the fitted model (`mlflow.sklearn.log_model`), and a feature-importance plot where the model exposes one (`HistGradientBoostingRegressor` doesn't, so it's skipped gracefully there).
- **Champion promotion:** the champion's run is re-opened, evaluated once on the test split, registered under `eta-best-model`, and promoted to the `Production` stage (archiving any prior Production version) — this is the version the serving layer loads.

## 7. Serving Architecture

**Design: two independently-runnable processes, talking over plain HTTP.**

```
┌────────────────────┐         HTTP POST /predict          ┌─────────────────────┐
│  Streamlit UI       │ ───────────────────────────────────▶│  FastAPI service     │
│  streamlit_app.py   │◀─────────────────────────────────── │  serve.py + api.py   │
│  (forms, map,       │         JSON {eta_minutes, ...}      │  + predictor.py       │
│   health check)     │                                       │  Production model +  │
└────────────────────┘                                       │  matching scaler      │
                                                               └─────────────────────┘
```

- **Why split into two processes** rather than one Streamlit app that loads the model directly: separates the *model-serving* concern (needs sklearn/xgboost/MLflow loaded, owns the inference latency) from the *UI* concern (just HTTP + forms). Either can be restarted independently, and the API alone is directly reusable by any other client (a mobile app, a batch job, `curl`) without dragging a Streamlit process along.
- **`Predictor` loads once, not per-request** (`get_predictor()` is a thread-safe lazy singleton) — deserializing the model and reloading the scaler on every request would dominate inference latency for a single-row prediction.
- **Train/serve feature parity by construction:** `predictor.py` calls the exact same `eta_pipeline.feature_engineering.build_features()` used at training time, and loads the `scaler.pkl` + `metadata.json` from the **same versioned dataset directory** that produced the currently-registered model — rather than re-implementing feature logic in the serving layer, which is the most common source of train/serve skew.
- **`mlflow.set_tracking_uri()` fix:** the serving process previously never pointed MLflow at the project's SQLite store, so registry lookups silently hit MLflow's default store and reported "model not found" even when one was registered — every process that talks to the registry has to independently set this; there's no ambient shared state across processes.

---

## 8. Known Limitations / Follow-ups

1. **Linear Regression's `n_iter=1` collapse** (§5.1) — the baseline model doesn't actually search both `fit_intercept` values, producing a misleadingly bad baseline on some runs.
2. **Champion selection metric mismatch** (§5.6) — code picks by `val_mae`, config declares `val_rmse`.
3. **Repo size growth** — `mlruns/` and `data/processed/versioned/` are now git-tracked (per request, for audit/reproducibility); every future `python main.py` run adds a new run + dataset version to the next commit, and being binary content, this only grows the repo, never shrinks it without a history rewrite.
4. **Weather auto-lookup not implemented at serving time** — `TripInput` defaults missing weather fields to 0 rather than looking up the actual date's weather, since no `weather_data_nyc.csv` is present in this environment.

---

## 9. Reproducing This Document's Numbers

```bash
python main.py                 # full pipeline; add --skip-eda for a faster run
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
python serve.py                 # API on :8000
streamlit run streamlit_app.py  # UI on :8501
```
