# TEST REPORT

Date: 2026-03-01
Project: Breast-Cancer-Classification

## 1. System Overview

- Training entrypoint: `train_automl.py`
  - Loads dataset (`sklearn_breast_cancer` or CSV), splits, scales, trains FLAML models, evaluates, saves `models_bundle.pkl`.
  - Evidence:
    - Data load path: `train_automl.py` (`load_data`)
    - Model training loop: `train_automl.py` (`model_types`, `train_flaml_model`)
    - Bundle save: `train_automl.py` (`joblib.dump(bundle, "models_bundle.pkl")`)
- Inference entrypoint: `app.py`
  - Loads `models_bundle.pkl`, aligns/normalizes input features, computes model predictions, consensus diagnosis, metrics, EDA, SHAP explainability.
  - Evidence:
    - Bundle load/validation: `app.py` (`load_bundle`)
    - Inference probability adapter: `app.py` (`get_positive_proba`)
    - Feature alignment/scaling: `app.py` (`reindex(..., fill_value=0)` + `scaler.transform`)
- Runtime/deployment:
  - Local: `streamlit run app.py`
  - Container: `Dockerfile`, `docker-compose.yml` service `classification-lab`

## 2. Issues Found

### Logic/ML correctness
- Incorrect positive-class probability handling for `decision_function` models (class-0 semantics mismatch).
  - Evidence: fixed function path in `app.py` (`get_positive_proba`, decision-function branch now maps to class 0).

### Robustness & error handling
- Bundle corruption/malformed bundle could fail without clear guard.
  - Evidence: `app.py` `load_bundle` now validates required keys and handles generic load exceptions.
- CSV upload invalid payload handling (non-CSV bytes/parse errors) needed explicit user-safe failure path.
  - Evidence: `app.py` `load_dataframe_from_upload` now catches `EmptyDataError`, `ParserError`, `UnicodeDecodeError` and raises controlled `ValueError`.
- Empty/invalid feature input path needed explicit safeguards.
  - Evidence: `app.py` checks for empty features and scaling errors before inference.

### Configuration/dependency/deployment
- Requirements had redundant/unused dependencies and less strict pin.
  - Evidence: `requirements.txt` cleaned to `flaml[automl]`, removed unused `openpyxl`/`fpdf`, pinned `numpy==2.3.0`, retained test deps.
- Docker reliability/security mismatches.
  - Evidence:
    - `Dockerfile` now uses `python:3.13-slim` and ensures model generation if bundle missing.
    - `docker-compose.yml` renamed service to `classification-lab` and removed insecure flags (`--server.enableCORS=false`, `--server.enableXsrfProtection=false`).

## 3. Tests Created

Test suite added under `tests/`:

- `tests/conftest.py` (shared fixtures)
- `tests/test_train_automl.py` (training unit/integration)
- `tests/test_app_utils.py` (app utility logic tests)
- `tests/test_ml_pipeline.py` (model/bundle/SHAP/pipeline tests)
- `tests/test_edge_cases.py` (empty/wrong schema/missing-corrupt model/nulls/threshold boundaries)

Execution evidence:
- Command: `./venv/Scripts/python.exe -m pytest tests/ -q`
- Result (latest): **99 passed, 0 failed**

## 4. Stress Results

Executed stress scenarios (system + ML + data + UI):

### Stress matrix summary
- Hard failures: **0**
- Status counts: **PASS=10**, **PASS_EXPECTED_NEGATIVE=2**
- Expected negative-path validations:
  - Missing model file -> `FileNotFoundError` (expected)
  - Corrupt model file -> load exception (expected)

### Performance/stability observations
- Large CSV batch: processed **119,490 rows** (PASS)
- Batch processing (all models): **69,987 rows across 5 models** in **0.868s** (PASS)
- Repeated inference: **500 loops**, avg **20.705 ms** per loop (PASS)
- Large dataset path: **219,634 rows** in **0.197s** (PASS)
- UI rapid interactions (Streamlit):
  - 300 requests, 300 OK, 0 failures
  - p50: 3.947 ms, p95: 27.408 ms, p99: 28.249 ms

## 5. Fixes Applied

### `app.py`
- Added robust bundle loading and key validation (`load_bundle`).
- Normalized `class_labels` keys to `int` and validated keys `0/1`.
- Corrected `get_positive_proba` class-0 mapping for:
  - `predict_proba` using `model.classes_` when available
  - `decision_function` binary/multiclass handling
- Added safe CSV parser error handling in `load_dataframe_from_upload`.
- Added feature/schema safeguards:
  - warns on missing/extra columns
  - guards empty inputs
  - catches scaler transform failures
- Replaced invalid Streamlit width usage with `use_container_width=True`.
- Hardened EDA correlation calls with `numeric_only=True`.
- Stabilized SHAP rendering using current matplotlib figure (`plt.gcf()`).

### `train_automl.py`
- Added robust non-file CSV load exception wrapping.
- Moved FLAML logs to `logs/` directory.
- Corrected training output label from “Best accuracy” to “Best ROC-AUC”.
- Added `zero_division=0` to precision/recall/F1 metric calls.
- Restored warning visibility (`warnings.filterwarnings("default")`).
- Added hard stop if no model trains before bundle save.

### Config/deps
- `requirements.txt` cleaned and pinned (`numpy==2.3.0`), removed dead deps, kept test tooling.
- `Dockerfile` updated for stable base image and startup model generation guard.
- `docker-compose.yml` aligned service naming and safer Streamlit command.
- `.gitignore` / `.dockerignore` updated for log/cache artifacts.

## 6. Cleanup Done

Removed generated/dead artifacts from repo root:
- `flaml_extra_tree.log`
- `flaml_lgbm.log`
- `flaml_lrl1.log`
- `flaml_rf.log`
- `flaml_xgboost.log`
- `__pycache__/`
- `.pytest_cache/`
- `.pytest_full_output.txt`

Added ignore rules for future generated artifacts:
- `.gitignore`: `*.log`, `logs/`
- `.dockerignore`: `*.log`, `logs`, `*.pyc`, `**/__pycache__/`

## 7. Final Stability

Final validation loop status:
- Tests: **PASS** (99/99)
- Stress matrix: **PASS** (0 hard failures)
- UI rapid interaction stress: **PASS** (0 request failures)

Conclusion (evidence-based):
- No test regressions detected after fixes.
- No unexpected crashes detected in exercised system/ML/data/UI paths.
- Negative-path behaviors (missing/corrupt model) fail correctly and predictably.
