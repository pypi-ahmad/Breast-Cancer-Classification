# ADR 0001: Run optional foundation models locally

- Status: Accepted
- Date: 2026-09-22

## Context

The project needs to compare its classical binary classifiers with Mitra v2 and
TabFM while keeping the default application small.

The two foundation models have different operational constraints:

- Mitra is distributed through AutoGluon and can use CUDA or CPU.
- TabFM is a 1.6B-parameter in-context model. Its weights are restricted to noncommercial, nonproduction research.
- Both checkpoints are much larger than the repository and default Docker image.
- The dashboard expects small, Joblib-compatible classifier objects with `predict`, probability support, class order, and parameter reporting.

## Decision

Foundation models are optional local dependencies installed with
`uv sync --extra foundation`.

The training command selects them explicitly with `--foundation-models`. TabFM also requires `--accept-tabfm-noncommercial-license` before any training or download begins.

Both models are wrapped in lazy, serializable adapters:

- checkpoint files stay in the Hugging Face user cache;
- AutoGluon output stays in ignored `model_artifacts/`;
- loaded runtimes are excluded from Joblib serialization;
- runtimes are released between evaluation steps;
- provenance and license state are stored in bundle metadata;
- Tree SHAP is disabled for these adapters.

TabFM uses the pinned PyTorch checkpoint on CPU for Windows. The JAX/Orbax path
aborted during checkpoint restore while requesting a 1.50 GB memory region.
PyTorch reports an operating-system error that the application can handle.

The Docker image remains on the base dependency set. It does not download, train, or serve foundation models.

## Consequences

### Benefits

- The classical setup remains small and reproducible.
- Model revisions and license state travel with the bundle metadata.
- No checkpoint or generated AutoGluon artifact is committed.
- The Streamlit app can treat classical and foundation classifiers through one interface.
- Requested foundation failures stop the run instead of silently producing an incomplete bundle.

### Costs and limits

- A foundation bundle is not self-contained. It depends on the local checkpoint cache and, for Mitra, the AutoGluon artifact directory.
- First use requires substantial download, disk, memory, and inference time.
- TabFM cannot currently restore on the verified Windows host because its paging-file capacity is too small.
- TabFM weights cannot be used for commercial or production work under the recorded license.
- Foundation models do not support the dashboard’s Tree SHAP view.

## Alternatives considered

### Put foundation dependencies in the base environment

Rejected because every user and Docker build would carry the dependency and
image-size cost, even for classical models.

### Commit checkpoints or generated artifacts

Rejected because the files are large, cacheable, and governed by upstream
licenses. Git is not the model registry.

### Integrate regressor checkpoints

Rejected because the configured target is binary classification.

### Fine-tune Mitra

Rejected for this integration. The selected mode is inference-only with `fine_tune=False`, which keeps the first implementation bounded.

### Use JAX TabFM on Windows

Rejected for the current host after two checkpoint restores failed at the same
native allocation. The PyTorch path remains, though the host still needs more
paging-file capacity to finish restoration.

## Revisit when

- the host can complete TabFM restoration and inference;
- deployment requirements expand beyond local research;
- TabFM’s weight license changes;
- Docker needs an explicit foundation-model image; or
- the bundle must become portable across machines without external artifact paths.
