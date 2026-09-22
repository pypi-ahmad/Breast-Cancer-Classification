# Development and operations guide

Use this guide when working on the project from Windows PowerShell. uv manages
the Python environment and commands.

## Prerequisites

- Windows 11 or another environment supported by the locked dependencies
- Python 3.13, installed or selected by uv
- uv
- Docker Desktop only if you want to run the classical application in a container
- Enough local disk space for optional foundation checkpoints

## Set up the project

Install the base application and development dependencies:

```powershell
uv sync
```

Install the optional foundation stack only when you need it:

```powershell
uv sync --extra foundation
```

Use `--frozen` in automation to require exact agreement with `uv.lock`:

```powershell
uv sync --frozen
```

## Train and run locally

Create a classical model bundle:

```powershell
uv run python train_automl.py
```

Start the dashboard:

```powershell
uv run streamlit run app.py
```

Streamlit uses port 8501 unless you supply another port. Its health endpoint is `/_stcore/health`.

## Use a CSV dataset

Edit the configuration constants near the top of `train_automl.py`:

```python
DATA_SOURCE = "data/my_dataset.csv"
TARGET_COLUMN = "target"
APP_TITLE = "My Binary Classifier"
CLASS_LABELS = {0: "Positive", 1: "Negative"}
```

The target must be binary and compatible with class values `0` and `1`. The Streamlit app expects the same binary contract. Input columns are converted to the feature order stored in the bundle.

CSV files are ignored by Git to reduce the chance of committing private data.
Add an exception only for a dataset that is safe to version.

## Run checks

Run the complete test suite:

```powershell
uv run pytest tests/ -q
```

Some bundle integration tests skip when `models_bundle.pkl` is absent. Train a
bundle first when you need those checks.

Run focused foundation adapter tests without loading real checkpoints:

```powershell
uv run pytest tests/test_foundation_models.py -q
```

Check the edited Python modules with Ruff without adding Ruff to the project environment:

```powershell
uvx ruff check --ignore BLE001 train_automl.py foundation_models.py tests/test_foundation_models.py
```

Validate the Compose file:

```powershell
docker compose config --quiet
```

## Run with Docker

Build the image and train the classical bundle if needed:

```powershell
docker compose build
docker compose run --rm classification-lab uv run --no-sync python train_automl.py
docker compose up
```

The Compose service mounts the repository at `/app`, so it can read a locally
generated `models_bundle.pkl`. The container excludes foundation dependencies
and checkpoints.

## Generated files

| Path | Created by | Versioned |
|---|---|---:|
| `models_bundle.pkl` | Training pipeline | No |
| `logs/` | FLAML searches | No |
| `model_artifacts/mitra-v2/` | AutoGluon Mitra training | No |
| Hugging Face user cache | Checkpoint downloads | No |

Deleting `models_bundle.pkl` forces a fresh classical training run. Mitra
artifacts and Hugging Face checkpoints can be reused until their pinned revision
changes.

## Troubleshooting

### The app reports that the bundle is missing

Run:

```powershell
uv run python train_automl.py
```

Then start Streamlit from the repository root so its relative bundle path resolves correctly.

### The app rejects the bundle

The file must contain `models`, `scaler`, and `feature_names`. Retrain it if an
incompatible revision created it. Do not repair an untrusted pickle manually.

### Uploaded columns do not match

Use the exact names listed in `bundle["feature_names"]`. The app drops extra columns and fills missing columns with zero, but those fallbacks can reduce prediction quality.

### Tests warn about estimator versions

Regenerate `models_bundle.pkl` with the current locked environment. Scikit-learn and XGBoost warn when a persisted estimator was created by another version.

### Foundation training fails

Use the [foundation-model runbook](foundation-models.md) for license,
checkpoint, memory, and Windows paging-file details.

## Security notes

- Treat Joblib bundles as executable artifacts and load only files you created or trust.
- Keep patient and organizational data outside Git.
- The app has no authentication or authorization layer. Bind it only to an appropriate local or controlled network environment.
- Review [SECURITY.md](../SECURITY.md) before reporting a vulnerability.
