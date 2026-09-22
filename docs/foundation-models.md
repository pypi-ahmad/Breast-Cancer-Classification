# Foundation-model runbook

This runbook covers the optional Mitra v2 and TabFM classifiers. The classical
FLAML and LazyPredict workflow runs without them.

## Model matrix

| Model | Pinned source | Runtime | License in this project | Current local result |
|---|---|---|---|---|
| Mitra v2 classifier | `autogluon/mitra-classifier-2` at `edada0d20759c58ada8c8605c25f22f6e98ea5f0` | AutoGluon; CUDA when available, otherwise CPU | Apache-2.0 | Passed training and inference on CPU |
| TabFM classifier | `google/tabfm-1.0.0-pytorch` at `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83` | PyTorch CPU | `tabfm-non-commercial-v1.0` weights | Checkpoint downloaded; restore blocked by Windows paging-file capacity |

Only classifier checkpoints are integrated. Regressor checkpoints do not match
this repository's binary target, so the runbook does not download them.

## Install

```powershell
uv sync --extra foundation
```

The extra installs AutoGluon 1.6.3 with Mitra support, TabFM from the pinned
upstream Git commit, and `truststore`. `truststore` lets Python use the Windows
certificate store for Hugging Face downloads.

## Train one model

Mitra only:

```powershell
uv run python train_automl.py --foundation-models mitra
```

TabFM only:

```powershell
uv run python train_automl.py --foundation-models tabfm --accept-tabfm-noncommercial-license
```

## Train both models

```powershell
uv run python train_automl.py --foundation-models mitra tabfm --accept-tabfm-noncommercial-license
```

The TabFM acceptance flag is mandatory. Without it, the command exits before
classical training or checkpoint download.

## What the pipeline does

### Mitra v2

1. Downloads the pinned Hugging Face snapshot.
2. Removes only the generated `model_artifacts/mitra-v2` directory from a previous attempt.
3. Builds a named DataFrame from scaled training features.
4. Fits AutoGluon with `fine_tune=False` and without a weighted ensemble.
5. Uses CUDA when the installed PyTorch runtime reports it as available.
6. Retries on CPU only when the CUDA attempt raises an out-of-memory error.
7. Stores a lazy adapter in the bundle rather than embedding the loaded runtime.

AutoGluon receives `ag.max_memory_usage_ratio=1.25`. This lets the verified
77M-parameter model run when AutoGluon's conservative estimate sits slightly
above the default safety threshold. It does not make an undersized machine safe.

### TabFM

1. Downloads only the pinned PyTorch classification checkpoint.
2. Stores the scaled training context and checkpoint path in the adapter.
3. Loads the model lazily on CPU.
4. Uses four estimators, a maximum of 100 context rows, batch size 1, and random seed 42.
5. Releases the runtime after evaluation and removes it during bundle serialization.

TabFM remains research-only. Its weights are not licensed for commercial or
production work.

## Verified results

The latest run is recorded in [TEST_REPORT.md](../TEST_REPORT.md).

- Mitra completed on CPU. AutoGluon reported 0.978 validation accuracy and a 306.67-second fit.
- The 6.56 GB TabFM PyTorch classification checkpoint downloaded at the pinned revision.
- TabFM restore failed with Windows error 1455: `The paging file is too small for this operation to complete.` No TabFM accuracy or completed combined bundle is claimed.

These results come from one local run. They are not a benchmark guarantee.

## Troubleshooting

### TabFM exits because license acceptance is missing

Add the explicit flag only after confirming the run is noncommercial and nonproduction:

```powershell
--accept-tabfm-noncommercial-license
```

### Hugging Face reports a certificate verification error

Keep `truststore` installed through the `foundation` extra.
`train_foundation_models()` injects the Windows trust store before importing the
Hugging Face download client.

Do not disable TLS verification.

### AutoGluon skips Mitra because memory is low

Close unrelated memory-heavy applications and retry. AutoGluon estimated the
verified run at roughly 7.1 GB. Do not terminate unrelated Python processes or
raise the configured memory ratio without measuring available memory.

### Mitra runs slowly

CPU execution is slow because the model attends over its training context for
each prediction. In the verified run, fitting and validation took several
minutes. CUDA is used only when the installed PyTorch build and driver expose it.

### TabFM reports Windows error 1455

The checkpoint is already cached, so another run does not need to download it again. The operating system rejected the weight mapping because paging-file capacity was insufficient.

Increase Windows virtual-memory capacity outside the repository, or move the run
to a machine with enough memory and commit headroom. This project does not alter
system paging settings automatically.

After changing the host configuration, rerun:

```powershell
uv run python train_automl.py --foundation-models tabfm --accept-tabfm-noncommercial-license
```

The TabFM path is incomplete until the command evaluates the model and writes
`models_bundle.pkl`.

### The app cannot find Mitra artifacts

Run Streamlit from the repository root. `MitraV2ClassifierAdapter` stores the relative path `model_artifacts/mitra-v2`.

If the directory was removed, retrain Mitra before loading a bundle that references it.

## Storage and cleanup

- AutoGluon output lives in ignored `model_artifacts/`.
- Hugging Face checkpoints live in the user cache and are not copied into this repository.
- Docker excludes `model_artifacts/` and does not install the foundation extra.
- The serialized adapter drops loaded model objects, but the TabFM training context remains in the bundle because it is required for in-context inference.
