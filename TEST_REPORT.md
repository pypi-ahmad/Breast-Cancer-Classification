# Test report

Date: 2026-09-22
Project: Breast-Cancer-Classification

## System overview

- Training entrypoint: `train_automl.py`
  - Loads `sklearn_breast_cancer` or a CSV, splits and scales the data, trains FLAML models, can add the leading LazyPredict classifiers, evaluates them, and saves `models_bundle.pkl`.
  - Source locations:
    - Data load path: `train_automl.py` (`load_data`)
    - Model training loop: `train_automl.py` (`model_types`, `train_flaml_model`)
    - Bundle save: `train_automl.py` (`joblib.dump(bundle, "models_bundle.pkl")`)
- Inference entrypoint: `app.py`
  - Loads `models_bundle.pkl`, aligns and normalizes input features, then calculates predictions, consensus diagnosis, metrics, EDA, and SHAP output.
  - Source locations:
    - Bundle load/validation: `app.py` (`load_bundle`)
    - Inference probability adapter: `app.py` (`get_positive_proba`)
    - Feature alignment/scaling: `app.py` (`reindex(..., fill_value=0)` + `scaler.transform`)
- Runtime/deployment:
  - Local: `uv run streamlit run app.py`
  - Container: `Dockerfile`, `docker-compose.yml` service `classification-lab`

## Issues found

### Logic and ML correctness

- `decision_function` models previously handled the positive-class probability incorrectly for class `0`.
  - Evidence: `app.py` now maps the decision-function path in `get_positive_proba` to class `0`.

### Error handling

- A corrupt or malformed bundle could fail without a clear guard.
  - Evidence: `load_bundle` validates required keys and handles generic load exceptions.
- Invalid CSV payloads needed a controlled failure path.
  - Evidence: `load_dataframe_from_upload` catches `EmptyDataError`, `ParserError`, and `UnicodeDecodeError`, then raises `ValueError`.
- Empty or invalid feature input needed explicit checks.
  - Evidence: `app.py` checks for empty features and scaling errors before inference.

### Configuration, dependencies, and deployment

- The old requirements list contained redundant dependencies and a less strict pin.
  - Evidence: `pyproject.toml` defines runtime and test dependencies, pins `numpy==2.3.0`, and removes `openpyxl` and `fpdf`.
- Docker configuration had reliability and security mismatches.
  - Evidence:
    - `Dockerfile` now uses `python:3.13-slim` and ensures model generation if bundle missing.
    - `docker-compose.yml` renamed service to `classification-lab` and removed insecure flags (`--server.enableCORS=false`, `--server.enableXsrfProtection=false`).

## Tests

Test suite added under `tests/`:

- `tests/conftest.py` (shared fixtures)
- `tests/test_train_automl.py` (training unit/integration)
- `tests/test_app_utils.py` (app utility logic tests)
- `tests/test_ml_pipeline.py` (model/bundle/SHAP/pipeline tests)
- `tests/test_edge_cases.py` (empty/wrong schema/missing-corrupt model/nulls/threshold boundaries)
- `tests/test_foundation_models.py` (adapter, serialization, probability, and license-gate contracts)

Execution:

- Command: `uv run pytest tests/ -q`
- Result (latest): **106 passed, 0 failed**
- Streamlit smoke: `/_stcore/health` returned `ok` on port 8596.

### Foundation-model run

- `uv sync --extra foundation`: passed with the pinned AutoGluon, TabFM, PyTorch, and checkpoint dependencies.
- Mitra v2 classifier: passed training and test-set inference on CPU. AutoGluon reported 0.978 validation accuracy and a 306.67-second fit.
- TabFM classifier checkpoint: the pinned 6.56 GB PyTorch classification checkpoint downloaded successfully.
- TabFM inference: blocked during local weight restore with Windows error 1455: `The paging file is too small for this operation to complete.` The earlier JAX path was also rejected because Orbax could not allocate a 1.50 GB memory region. No TabFM metric or completed foundation bundle is claimed.

## Stress results

Stress scenarios covered the system, ML path, data path, and UI.

### Stress matrix

- Hard failures: **0**
- Status counts: **PASS=10**, **PASS_EXPECTED_NEGATIVE=2**
- Expected negative-path validations:
  - Missing model file -> `FileNotFoundError` (expected)
  - Corrupt model file -> load exception (expected)

### Performance and stability

- Large CSV batch: processed **119,490 rows** (PASS)
- Batch processing (all models): **69,987 rows across 5 models** in **0.868s** (PASS)
- Repeated inference: **500 loops**, avg **20.705 ms** per loop (PASS)
- Large dataset path: **219,634 rows** in **0.197s** (PASS)
- UI rapid interactions (Streamlit):
  - 300 requests, 300 OK, 0 failures
  - p50: 3.947 ms, p95: 27.408 ms, p99: 28.249 ms

## Fixes

### `app.py`

- Added bundle loading and key validation in `load_bundle`.
- Normalized `class_labels` keys to `int` and validated keys `0/1`.
- Corrected `get_positive_proba` class-0 mapping for:
  - `predict_proba` using `model.classes_` when available
  - `decision_function` binary/multiclass handling
- Added controlled CSV parser errors in `load_dataframe_from_upload`.
- Added feature/schema safeguards:
  - warns on missing/extra columns
  - guards empty inputs
  - catches scaler transform failures
- Replaced invalid Streamlit width usage with `use_container_width=True`.
- Hardened EDA correlation calls with `numeric_only=True`.
- Rendered SHAP output through the current matplotlib figure (`plt.gcf()`).

### `train_automl.py`

- Added clear exception wrapping for CSV load failures.
- Moved FLAML logs to `logs/` directory.
- Corrected training output label from “Best accuracy” to “Best ROC-AUC”.
- Added `zero_division=0` to precision/recall/F1 metric calls.
- Restored warning visibility with `warnings.filterwarnings("default")`.
- Added hard stop if no model trains before bundle save.
- Added pinned Mitra and TabFM classifier adapters, explicit TabFM license acceptance, model provenance, and runtime release between foundation models.

### Configuration and dependencies

- `pyproject.toml` and `uv.lock` now define and lock runtime and test dependencies, including `numpy==2.3.0`.
- `Dockerfile` updated for stable base image and startup model generation guard.
- `docker-compose.yml` aligned service naming and safer Streamlit command.
- `.gitignore` / `.dockerignore` updated for log/cache artifacts.

## Cleanup

Removed generated artifacts from the repository root:

- `flaml_extra_tree.log`
- `flaml_lgbm.log`
- `flaml_lrl1.log`
- `flaml_rf.log`
- `flaml_xgboost.log`
- `__pycache__/`
- `.pytest_cache/`
- `.pytest_full_output.txt`

Added ignore rules for generated artifacts:

- `.gitignore`: `*.log`, `logs/`
- `.dockerignore`: `*.log`, `logs`, `*.pyc`, `**/__pycache__/`

## Final status

Validation status:

- Tests: **PASS** (106/106)
- Mitra end-to-end: **PASS**
- TabFM end-to-end: **BLOCKED by Windows paging-file capacity during weight restore**
- Stress matrix: **PASS** (0 hard failures)
- UI rapid interaction stress: **PASS** (0 request failures)

The tests found no regressions. The classical and Mitra paths passed. TabFM remains unavailable on this machine until the Windows paging file has more capacity. Missing and corrupt model paths fail predictably.
