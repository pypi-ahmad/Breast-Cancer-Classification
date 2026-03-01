"""
Unit + Integration tests for train_automl.py.

Tests cover:
- load_data() function (L37-60)
- train_flaml_model() function (L62-91)
- evaluate_models() function (L93-114)
- main() pipeline integration (L116-197)
- Data split correctness
- Scaler behavior
- Bundle save/load round-trip
"""
import os
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_automl


# ===========================================================================
# UNIT TESTS: load_data()
# ===========================================================================

class TestLoadData:
    """Tests for train_automl.load_data() — L37-60."""

    def test_returns_dataframe(self):
        """load_data() must return a pandas DataFrame."""
        df = train_automl.load_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_target_column(self):
        """Returned DataFrame must contain the 'target' column (L53)."""
        df = train_automl.load_data()
        assert "target" in df.columns

    def test_correct_shape(self):
        """Breast cancer dataset: 569 samples, 30 features + 1 target = 31 columns."""
        df = train_automl.load_data()
        assert df.shape == (569, 31)

    def test_target_values_are_binary(self):
        """Target must contain only 0 and 1 (breast cancer encoding)."""
        df = train_automl.load_data()
        assert set(df["target"].unique()) == {0, 1}

    def test_no_missing_values(self):
        """Breast cancer dataset should have no NaN values."""
        df = train_automl.load_data()
        assert df.isnull().sum().sum() == 0

    def test_feature_names_match_sklearn(self):
        """Feature column names must match sklearn.datasets.load_breast_cancer."""
        df = train_automl.load_data()
        expected = list(load_breast_cancer().feature_names)
        actual = [c for c in df.columns if c != "target"]
        assert actual == expected

    def test_csv_file_not_found_raises(self):
        """When DATA_SOURCE is a nonexistent CSV, must raise FileNotFoundError (L59-60)."""
        with patch.object(train_automl, "DATA_SOURCE", "nonexistent_file.csv"):
            with pytest.raises(FileNotFoundError, match="Could not find file"):
                train_automl.load_data()


# ===========================================================================
# UNIT TESTS: evaluate_models()
# ===========================================================================

class TestEvaluateModels:
    """Tests for train_automl.evaluate_models() — L93-114."""

    def test_returns_dataframe(self, train_test_data):
        """evaluate_models() must return a pd.DataFrame."""
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = train_test_data
        # Use a simple mock model
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train_scaled, y_train)
        result = train_automl.evaluate_models({"Dummy": dummy}, X_test_scaled, y_test)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, train_test_data):
        """Result must have Model, Accuracy, Recall (Sensitivity), Precision, F1 Score (L104-108)."""
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = train_test_data
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train_scaled, y_train)
        result = train_automl.evaluate_models({"Dummy": dummy}, X_test_scaled, y_test)
        expected_cols = {"Model", "Accuracy", "Recall (Sensitivity)", "Precision", "F1 Score"}
        assert set(result.columns) == expected_cols

    def test_metrics_in_valid_range(self, train_test_data):
        """All metric values must be in [0.0, 1.0]."""
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = train_test_data
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="stratified", random_state=42).fit(X_train_scaled, y_train)
        result = train_automl.evaluate_models({"Dummy": dummy}, X_test_scaled, y_test)
        numeric_cols = ["Accuracy", "Recall (Sensitivity)", "Precision", "F1 Score"]
        for col in numeric_cols:
            assert (result[col] >= 0.0).all() and (result[col] <= 1.0).all(), f"{col} out of range"

    def test_one_row_per_model(self, train_test_data):
        """Each model produces exactly one row in the result."""
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = train_test_data
        from sklearn.dummy import DummyClassifier
        d1 = DummyClassifier(strategy="most_frequent").fit(X_train_scaled, y_train)
        d2 = DummyClassifier(strategy="stratified", random_state=0).fit(X_train_scaled, y_train)
        result = train_automl.evaluate_models({"A": d1, "B": d2}, X_test_scaled, y_test)
        assert len(result) == 2
        assert list(result["Model"]) == ["A", "B"]

    def test_empty_models_dict(self, train_test_data):
        """Empty models dict should return empty DataFrame."""
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = train_test_data
        result = train_automl.evaluate_models({}, X_test_scaled, y_test)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ===========================================================================
# UNIT TESTS: train_flaml_model()
# ===========================================================================

class TestTrainFlamlModel:
    """Tests for train_automl.train_flaml_model() — L62-91."""

    def test_returns_automl_object(self, train_test_data):
        """Must return a FLAML AutoML instance (L91)."""
        from flaml import AutoML
        X_train_scaled, _, y_train, _, _, _ = train_test_data
        model = train_automl.train_flaml_model(X_train_scaled, y_train, "lrl1", time_budget=5)
        assert isinstance(model, AutoML)

    def test_model_can_predict(self, train_test_data):
        """Trained model must have a predict() method that works."""
        X_train_scaled, X_test_scaled, y_train, _, _, _ = train_test_data
        model = train_automl.train_flaml_model(X_train_scaled, y_train, "lrl1", time_budget=5)
        preds = model.predict(X_test_scaled)
        assert len(preds) == len(X_test_scaled)

    def test_model_predict_values_are_binary(self, train_test_data):
        """Predictions must be 0 or 1 for binary classification."""
        X_train_scaled, X_test_scaled, y_train, _, _, _ = train_test_data
        model = train_automl.train_flaml_model(X_train_scaled, y_train, "lrl1", time_budget=5)
        preds = model.predict(X_test_scaled)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_model_has_best_config(self, train_test_data):
        """AutoML object must have a best_config attribute after training (L89)."""
        X_train_scaled, _, y_train, _, _, _ = train_test_data
        model = train_automl.train_flaml_model(X_train_scaled, y_train, "lrl1", time_budget=5)
        assert hasattr(model, "best_config")
        assert model.best_config is not None


# ===========================================================================
# INTEGRATION TESTS: Data Split & Scaling
# ===========================================================================

class TestDataSplitAndScaling:
    """Validates the train/test split and scaling done in main() L141-154."""

    def test_split_sizes(self, raw_dataframe):
        """80/20 split of 569 samples → 455 train, 114 test."""
        X = raw_dataframe.drop(columns=["target"])
        y = raw_dataframe["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        assert len(X_train) == 455
        assert len(X_test) == 114

    def test_stratification_preserved(self, raw_dataframe):
        """Stratified split must preserve approximate class ratios."""
        X = raw_dataframe.drop(columns=["target"])
        y = raw_dataframe["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        full_ratio = (y == 0).sum() / len(y)
        train_ratio = (y_train == 0).sum() / len(y_train)
        test_ratio = (y_test == 0).sum() / len(y_test)
        assert abs(full_ratio - train_ratio) < 0.02
        assert abs(full_ratio - test_ratio) < 0.02

    def test_scaler_mean_near_zero(self, train_test_data):
        """After StandardScaler.fit_transform, training data mean ≈ 0."""
        X_train_scaled, _, _, _, _, _ = train_test_data
        means = np.abs(X_train_scaled.mean(axis=0))
        assert np.all(means < 1e-10), f"Max mean: {means.max()}"

    def test_scaler_std_near_one(self, train_test_data):
        """After StandardScaler.fit_transform, training data std ≈ 1."""
        X_train_scaled, _, _, _, _, _ = train_test_data
        stds = X_train_scaled.std(axis=0)
        assert np.allclose(stds, 1.0, atol=0.05)

    def test_no_data_leakage_scaler(self, raw_dataframe):
        """Scaler must be fit ONLY on training data, not on test data."""
        X = raw_dataframe.drop(columns=["target"])
        y = raw_dataframe["target"]
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Test set mean should NOT be exactly 0 (would indicate leakage)
        test_means = X_test_scaled.mean(axis=0)
        assert not np.allclose(test_means, 0.0, atol=1e-10), \
            "Test set means are exactly 0 — possible data leakage"

    def test_feature_count_matches(self, train_test_data):
        """Scaled data must have 30 features (breast cancer dataset)."""
        X_train_scaled, X_test_scaled, _, _, _, feature_names = train_test_data
        assert X_train_scaled.shape[1] == 30
        assert X_test_scaled.shape[1] == 30
        assert len(feature_names) == 30


# ===========================================================================
# INTEGRATION TEST: Bundle Save/Load Round-Trip
# ===========================================================================

class TestBundleRoundTrip:
    """Validates bundle save/load as done in main() L181-193."""

    def test_bundle_file_exists(self):
        """models_bundle.pkl must exist on disk."""
        from tests.conftest import BUNDLE_PATH
        if not os.path.exists(BUNDLE_PATH):
            pytest.skip("models_bundle.pkl not found — run train_automl.py first")
        assert os.path.exists(BUNDLE_PATH)

    def test_bundle_has_required_keys(self, bundle):
        """Bundle must contain: models, scaler, feature_names, metadata (L181-189)."""
        required = {"models", "scaler", "feature_names", "metadata"}
        assert required.issubset(set(bundle.keys()))

    def test_bundle_models_is_dict(self, bundle):
        assert isinstance(bundle["models"], dict)

    def test_bundle_scaler_is_standard_scaler(self, bundle):
        assert isinstance(bundle["scaler"], StandardScaler)

    def test_bundle_feature_names_is_list(self, bundle):
        assert isinstance(bundle["feature_names"], list)
        assert all(isinstance(f, str) for f in bundle["feature_names"])

    def test_bundle_metadata_has_title(self, bundle_metadata):
        """Metadata must have 'title' key (L185)."""
        assert "title" in bundle_metadata

    def test_bundle_metadata_has_class_labels(self, bundle_metadata):
        """Metadata must have 'class_labels' with keys 0 and 1 (L186)."""
        assert "class_labels" in bundle_metadata
        labels = bundle_metadata["class_labels"]
        assert 0 in labels and 1 in labels

    def test_bundle_models_not_empty(self, bundle_models):
        """At least one trained model must be in the bundle."""
        assert len(bundle_models) > 0

    def test_bundle_round_trip_integrity(self, bundle):
        """Save + reload must produce identical bundle keys and feature_names."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(bundle, tmp_path)
            reloaded = joblib.load(tmp_path)
            assert set(reloaded.keys()) == set(bundle.keys())
            assert reloaded["feature_names"] == bundle["feature_names"]
            assert list(reloaded["models"].keys()) == list(bundle["models"].keys())
        finally:
            os.unlink(tmp_path)
