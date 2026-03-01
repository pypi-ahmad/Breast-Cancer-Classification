"""
Edge case tests — boundary conditions, invalid data, missing files, corrupted models.

Tests cover:
- Empty input arrays
- Wrong-shaped input
- Missing models_bundle.pkl
- Corrupted/tampered bundle
- NaN / Inf in features
- Single-feature / single-sample
- All-same predictions
- Extreme threshold values
"""
import io
import os
import sys
import tempfile
import pickle
import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_automl

# Re-use the extracted get_positive_proba from test_app_utils
from tests.test_app_utils import get_positive_proba


# ===========================================================================
# EDGE CASES: Empty / Zero Input
# ===========================================================================

class TestEmptyInput:
    """Edge cases with empty or zero-length data."""

    def test_predict_empty_array(self, bundle_models, bundle_feature_names):
        """predict() on a (0, n_features) array should raise or return empty (model-dependent)."""
        X_empty = np.empty((0, len(bundle_feature_names)))
        for name, model in bundle_models.items():
            try:
                preds = model.predict(X_empty)
                # If it doesn't raise, result should be empty
                assert len(preds) == 0, f"Non-empty prediction for '{name}' on empty input"
            except ValueError:
                pass  # sklearn 1.8+ rejects 0-sample input for some estimators

    def test_predict_proba_empty_array(self, bundle_models, bundle_feature_names):
        """predict_proba on empty input should raise or return empty (model-dependent)."""
        X_empty = np.empty((0, len(bundle_feature_names)))
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_empty)
                    assert proba.shape[0] == 0, f"Non-empty proba for '{name}'"
                except ValueError:
                    pass  # sklearn 1.8+ rejects 0-sample input for some estimators

    def test_scaler_transform_empty(self, bundle_scaler, bundle_feature_names):
        """Scaler.transform on empty array should raise ValueError (sklearn 1.8+ rejects 0 samples)."""
        X_empty = np.empty((0, len(bundle_feature_names)))
        with pytest.raises(ValueError, match="0 sample"):
            bundle_scaler.transform(X_empty)

    def test_evaluate_models_empty_test_set(self):
        """evaluate_models with empty test array should work or fail gracefully."""
        from sklearn.dummy import DummyClassifier
        X_train = np.random.randn(10, 5)
        y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

        X_empty = np.empty((0, 5))
        y_empty = pd.Series(dtype=int)

        # This will raise because accuracy_score with empty data raises ValueError
        with pytest.raises((ValueError, ZeroDivisionError)):
            train_automl.evaluate_models({"Dummy": dummy}, X_empty, y_empty)


# ===========================================================================
# EDGE CASES: Wrong-Shaped Input
# ===========================================================================

class TestWrongShape:
    """Models must reject data with wrong number of features."""

    def test_too_few_features(self, bundle_models, bundle_feature_names):
        """Input with fewer features than expected should raise."""
        X_wrong = np.random.randn(5, len(bundle_feature_names) - 5)
        for name, model in bundle_models.items():
            with pytest.raises((ValueError, Exception)):
                model.predict(X_wrong)

    def test_too_many_features(self, bundle_models, bundle_feature_names):
        """Input with more features than expected should raise."""
        X_wrong = np.random.randn(5, len(bundle_feature_names) + 5)
        for name, model in bundle_models.items():
            with pytest.raises((ValueError, Exception)):
                model.predict(X_wrong)

    def test_1d_input_rejected(self, bundle_models, bundle_feature_names):
        """1D array should be rejected (not a matrix)."""
        X_1d = np.random.randn(len(bundle_feature_names))
        for name, model in bundle_models.items():
            with pytest.raises((ValueError, Exception)):
                model.predict(X_1d)

    def test_scaler_wrong_features(self, bundle_scaler):
        """Scaler should reject wrong number of features."""
        X_wrong = np.random.randn(5, 10)  # not 30
        with pytest.raises(ValueError):
            bundle_scaler.transform(X_wrong)


# ===========================================================================
# EDGE CASES: Missing / Corrupted Bundle
# ===========================================================================

class TestMissingCorruptedBundle:
    """Tests for missing or malformed models_bundle.pkl."""

    def test_load_nonexistent_bundle(self):
        """joblib.load on non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            joblib.load("nonexistent_bundle.pkl")

    def test_load_empty_file_as_bundle(self):
        """Loading an empty file should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
            f.write(b"")
            tmp_path = f.name
        try:
            with pytest.raises(Exception):
                joblib.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_non_pickle_file(self):
        """Loading a non-pickle file should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="w") as f:
            f.write("This is not a pickle file")
            tmp_path = f.name
        try:
            with pytest.raises(Exception):
                joblib.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_bundle_missing_models_key(self):
        """Bundle without 'models' key should fail on access."""
        bundle = {"scaler": StandardScaler(), "feature_names": ["a"]}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        joblib.dump(bundle, tmp_path)
        try:
            loaded = joblib.load(tmp_path)
            with pytest.raises(KeyError):
                _ = loaded["models"]
        finally:
            os.unlink(tmp_path)

    def test_bundle_missing_scaler_key(self):
        """Bundle without 'scaler' key should fail on access."""
        bundle = {"models": {}, "feature_names": ["a"]}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        joblib.dump(bundle, tmp_path)
        try:
            loaded = joblib.load(tmp_path)
            with pytest.raises(KeyError):
                _ = loaded["scaler"]
        finally:
            os.unlink(tmp_path)

    def test_bundle_missing_feature_names_key(self):
        """Bundle without 'feature_names' key should fail on access."""
        bundle = {"models": {}, "scaler": StandardScaler()}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        joblib.dump(bundle, tmp_path)
        try:
            loaded = joblib.load(tmp_path)
            with pytest.raises(KeyError):
                _ = loaded["feature_names"]
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# EDGE CASES: NaN / Inf in Input Data
# ===========================================================================

class TestNaNInfInput:
    """Edge cases with NaN and Inf values in feature data."""

    def test_scaler_transform_with_nan(self, bundle_scaler, bundle_feature_names):
        """Scaler.transform with NaN should produce NaN output (not crash)."""
        X_nan = np.full((3, len(bundle_feature_names)), np.nan)
        result = bundle_scaler.transform(X_nan)
        assert np.all(np.isnan(result))

    def test_scaler_transform_with_inf(self, bundle_scaler, bundle_feature_names):
        """Scaler.transform with Inf should raise ValueError (sklearn rejects infinite values)."""
        X_inf = np.full((3, len(bundle_feature_names)), np.inf)
        with pytest.raises(ValueError, match="infinity"):
            bundle_scaler.transform(X_inf)

    def test_predict_with_nan_features(self, bundle_models, bundle_feature_names):
        """Models may produce NaN predictions or raise — we verify no silent garbage.
        At minimum, the model should not crash with an unhandled exception."""
        X_nan = np.full((2, len(bundle_feature_names)), np.nan)
        for name, model in bundle_models.items():
            try:
                preds = model.predict(X_nan)
                # If it doesn't crash, output length should match
                assert len(preds) == 2
            except (ValueError, TypeError):
                # Some models may rightfully refuse NaN input
                pass

    def test_get_positive_proba_with_nan(self, bundle_models, bundle_feature_names):
        """get_positive_proba should not raise unhandled exception on NaN input."""
        X_nan = np.full((2, len(bundle_feature_names)), np.nan)
        for name, model in bundle_models.items():
            try:
                result = get_positive_proba(model, X_nan)
                assert len(result) == 2
            except (ValueError, TypeError):
                pass


# ===========================================================================
# EDGE CASES: Single Sample / All Same Data
# ===========================================================================

class TestBoundaryData:
    """Tests with boundary-condition data."""

    def test_all_zeros_input(self, bundle_models, bundle_scaler, bundle_feature_names):
        """All-zero feature vector (before scaling) should produce valid prediction."""
        X_zeros = np.zeros((1, len(bundle_feature_names)))
        X_scaled = bundle_scaler.transform(X_zeros)
        for name, model in bundle_models.items():
            pred = model.predict(X_scaled)
            assert pred[0] in (0, 1), f"Invalid prediction from '{name}' on all-zeros"

    def test_all_same_features(self, bundle_models, bundle_scaler, bundle_feature_names):
        """Constant feature vector should still produce valid prediction."""
        X_const = np.full((1, len(bundle_feature_names)), 42.0)
        X_scaled = bundle_scaler.transform(X_const)
        for name, model in bundle_models.items():
            pred = model.predict(X_scaled)
            assert pred[0] in (0, 1)

    def test_very_large_values(self, bundle_models, bundle_scaler, bundle_feature_names):
        """Extreme values should not crash models."""
        X_large = np.full((1, len(bundle_feature_names)), 1e10)
        X_scaled = bundle_scaler.transform(X_large)
        for name, model in bundle_models.items():
            pred = model.predict(X_scaled)
            assert len(pred) == 1

    def test_very_small_values(self, bundle_models, bundle_scaler, bundle_feature_names):
        """Very small (near-zero) values should not crash models."""
        X_small = np.full((1, len(bundle_feature_names)), 1e-15)
        X_scaled = bundle_scaler.transform(X_small)
        for name, model in bundle_models.items():
            pred = model.predict(X_scaled)
            assert len(pred) == 1


# ===========================================================================
# EDGE CASES: Threshold Boundaries (mirrors app.py L140-147)
# ===========================================================================

class TestThresholdBoundaries:
    """Test extreme threshold values for the consensus logic."""

    def test_threshold_zero_all_positive(self, bundle_models, inference_data):
        """With threshold=0.0, all samples should be classified as Positive (Malignant)."""
        X_scaled, _, _ = inference_data
        threshold = 0.0
        POS_CLASS = "Malignant"

        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 0]
                pred_label = np.where(proba >= threshold, 0, 1)
                # threshold=0.0 means any proba >= 0 → all predict 0
                assert np.all(pred_label == 0), \
                    f"threshold=0 should make all predictions Positive for '{name}'"

    def test_threshold_one_all_negative(self, bundle_models, inference_data):
        """With threshold=1.0, only perfect P(0)=1.0 samples are Positive."""
        X_scaled, _, _ = inference_data
        threshold = 1.0
        NEG_CLASS = "Benign"

        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 0]
                pred_label = np.where(proba >= threshold, 0, 1)
                # Most probabilities < 1.0, so most should be class 1
                # At minimum, at least some should be Negative
                neg_count = (pred_label == 1).sum()
                # Almost all should be negative at threshold 1.0
                assert neg_count > len(X_scaled) * 0.5, \
                    f"threshold=1.0 should produce mostly Negative for '{name}'"

    def test_consensus_threshold_zero(self, bundle_models, inference_data):
        """Consensus threshold=0.0 → everything is consensus Positive."""
        X_scaled, _, _ = inference_data
        POS_CLASS = "Malignant"
        NEG_CLASS = "Benign"
        mal_threshold = 0.5
        consensus_threshold = 0.0

        model_preds = pd.DataFrame(index=range(len(X_scaled)))
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 0]
            else:
                proba = np.zeros(len(X_scaled))
            pred_label = np.where(proba >= mal_threshold, 0, 1)
            model_preds[name] = np.where(pred_label == 0, POS_CLASS, NEG_CLASS)

        pos_votes = (model_preds == POS_CLASS).sum(axis=1)
        consensus_score = pos_votes / len(bundle_models)
        diagnosis = np.where(consensus_score >= consensus_threshold, POS_CLASS, NEG_CLASS)
        # consensus_threshold=0.0 → score >= 0.0 always true → all Positive
        assert np.all(diagnosis == POS_CLASS)


# ===========================================================================
# EDGE CASES: CSV upload with mismatched columns
# ===========================================================================

class TestCSVMismatch:
    """Tests for uploaded CSV with wrong/missing columns."""

    def test_upload_csv_completely_wrong_columns(self, bundle_scaler, bundle_feature_names):
        """CSV with no matching columns → all zeros after reindex."""
        csv_data = "foo,bar,baz\n1,2,3\n4,5,6\n"
        df = pd.read_csv(io.StringIO(csv_data))
        df_aligned = df.reindex(columns=bundle_feature_names, fill_value=0)
        assert (df_aligned.values == 0).all()
        # Scaler should still work
        X_scaled = bundle_scaler.transform(df_aligned)
        assert X_scaled.shape == (2, len(bundle_feature_names))

    def test_upload_csv_partial_columns(self, bundle_scaler, bundle_feature_names):
        """CSV with some matching columns → partial data, rest zeros."""
        # Use first 3 feature names
        cols = bundle_feature_names[:3]
        csv_data = ",".join(cols) + "\n" + ",".join(["1.0"] * len(cols)) + "\n"
        df = pd.read_csv(io.StringIO(csv_data))
        df_aligned = df.reindex(columns=bundle_feature_names, fill_value=0)
        # First 3 columns should be 1.0, rest 0
        assert all(df_aligned.iloc[0, :3] == 1.0)
        assert all(df_aligned.iloc[0, 3:] == 0.0)

    def test_upload_csv_with_target(self, bundle_feature_names):
        """CSV with 'target' column should be separable."""
        cols = bundle_feature_names[:2] + ["target"]
        csv_data = ",".join(cols) + "\n1.0,2.0,0\n3.0,4.0,1\n"
        df = pd.read_csv(io.StringIO(csv_data))
        assert "target" in df.columns
        labels = df["target"].astype(int)
        df_features = df.drop(columns=["target"])
        assert "target" not in df_features.columns
        assert len(labels) == 2
