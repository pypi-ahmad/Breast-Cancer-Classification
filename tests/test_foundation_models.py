"""Fast contract tests for optional foundation-model adapters."""

import pickle

import numpy as np
import pandas as pd
import pytest

from foundation_models import MitraV2ClassifierAdapter, TabFMClassifierAdapter
from train_automl import parse_args, validate_foundation_args


class FakeMitraPredictor:
    def predict(self, frame):
        assert list(frame.columns) == ["a", "b"]
        return pd.Series([0, 1])

    def predict_proba(self, frame):
        return pd.DataFrame({1: [0.2, 0.8], 0: [0.8, 0.2]})


class FakeTabFMClassifier:
    def predict(self, values):
        return np.array([0, 1])

    def predict_proba(self, values):
        return np.array([[0.7, 0.3], [0.1, 0.9]])


def test_mitra_adapter_preserves_class_order_and_feature_names():
    adapter = MitraV2ClassifierAdapter("unused", ["a", "b"])
    adapter._predictor = FakeMitraPredictor()
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert adapter.predict(values).tolist() == [0, 1]
    assert np.allclose(adapter.predict_proba(values), [[0.8, 0.2], [0.2, 0.8]])


def test_tabfm_adapter_probability_contract():
    adapter = TabFMClassifierAdapter("unused", np.ones((2, 2)), np.array([0, 1]))
    adapter._classifier = FakeTabFMClassifier()

    probabilities = adapter.predict_proba(np.ones((2, 2)))
    assert adapter.predict(np.ones((2, 2))).tolist() == [0, 1]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert adapter.research_only is True


def test_adapters_release_loaded_runtime():
    mitra = MitraV2ClassifierAdapter("unused", ["a"])
    tabfm = TabFMClassifierAdapter("unused", np.ones((2, 1)), np.array([0, 1]))
    mitra._predictor = object()
    tabfm._classifier = object()

    mitra.release_runtime()
    tabfm.release_runtime()

    assert mitra._predictor is None
    assert tabfm._classifier is None


@pytest.mark.parametrize(
    "adapter",
    [
        MitraV2ClassifierAdapter("unused", ["a"]),
        TabFMClassifierAdapter("unused", np.ones((2, 1)), np.array([0, 1])),
    ],
)
def test_serialization_drops_loaded_runtime(adapter):
    if isinstance(adapter, MitraV2ClassifierAdapter):
        adapter._predictor = object()
        restored = pickle.loads(pickle.dumps(adapter))
        assert restored._predictor is None
    else:
        adapter._classifier = object()
        restored = pickle.loads(pickle.dumps(adapter))
        assert restored._classifier is None


def test_tabfm_requires_explicit_license_acceptance():
    args = parse_args(["--foundation-models", "tabfm"])
    with pytest.raises(SystemExit, match="noncommercial"):
        validate_foundation_args(args)


def test_tabfm_license_acceptance_passes():
    args = parse_args(
        ["--foundation-models", "tabfm", "--accept-tabfm-noncommercial-license"]
    )
    validate_foundation_args(args)
