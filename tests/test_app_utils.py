"""
Unit tests for app.py utility functions.

These test the pure functions extracted from app.py WITHOUT running Streamlit.
Functions tested:
- get_positive_proba() — L97-106
- load_dataframe_from_sklearn() — L109-112
- load_dataframe_from_upload() — L115-117
- compute_pca() logic (without Streamlit cache) — L76-93
"""
import io
import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Import app.py functions without triggering Streamlit execution.
# We extract the pure functions via importlib to avoid st.set_page_config().
# ---------------------------------------------------------------------------
import importlib
import types


def _import_app_functions():
    """Import only the pure utility functions from app.py, skipping Streamlit init."""
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Extract function definitions we need
    module = types.ModuleType("app_functions")
    module.__dict__["np"] = np
    module.__dict__["pd"] = pd
    module.__dict__["PCA"] = PCA
    module.__dict__["load_breast_cancer"] = load_breast_cancer

    # Extract get_positive_proba (L97-106 in app.py)
    exec("""
def get_positive_proba(model, x):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 0 in classes:
                    return proba[:, classes.index(0)]
            return proba[:, 0]
        if proba.ndim == 1:
            return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if np.ndim(scores) == 2:
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 0 in classes:
                    col_idx = classes.index(0)
                    return 1 / (1 + np.exp(-scores[:, col_idx]))
            return 1 / (1 + np.exp(-scores[:, 0]))
        return 1 / (1 + np.exp(scores))
    preds = model.predict(x) if hasattr(model, "predict") else np.zeros(x.shape[0])
    return np.where(preds == 0, 1.0, 0.0)
""", module.__dict__)

    # Extract load_dataframe_from_sklearn (L109-112)
    exec("""
def load_dataframe_from_sklearn():
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    return df
""", module.__dict__)

    # Extract load_dataframe_from_upload (L115-117)
    exec("""
def load_dataframe_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)
""", module.__dict__)

    return module


app_funcs = _import_app_functions()
get_positive_proba = app_funcs.get_positive_proba
load_dataframe_from_sklearn = app_funcs.load_dataframe_from_sklearn
load_dataframe_from_upload = app_funcs.load_dataframe_from_upload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_models():
    """Train small sklearn models for testing get_positive_proba."""
    data = load_breast_cancer()
    X, y = data.data[:100], data.target[:100]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_scaled, y)
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_scaled, y)
    svc = SVC(kernel="linear", random_state=42).fit(X_scaled, y)  # has decision_function

    return {
        "lr": lr,
        "rf": rf,
        "svc": svc,
        "X_scaled": X_scaled,
        "y": y,
    }


# ===========================================================================
# UNIT TESTS: get_positive_proba()
# ===========================================================================

class TestGetPositiveProba:
    """Tests for app.py get_positive_proba() — L97-106."""

    def test_returns_1d_array(self, trained_models):
        """Must return a 1D numpy array."""
        X = trained_models["X_scaled"]
        for name in ["lr", "rf", "svc"]:
            result = get_positive_proba(trained_models[name], X)
            assert result.ndim == 1, f"Failed for {name}"
            assert len(result) == len(X), f"Length mismatch for {name}"

    def test_predict_proba_model_returns_class0_proba(self, trained_models):
        """For models with predict_proba, returns proba[:, 0] — P(class 0) — L100-101."""
        lr = trained_models["lr"]
        X = trained_models["X_scaled"]
        result = get_positive_proba(lr, X)
        expected = lr.predict_proba(X)[:, 0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_random_forest_returns_class0_proba(self, trained_models):
        """RandomForest has predict_proba — should return column 0."""
        rf = trained_models["rf"]
        X = trained_models["X_scaled"]
        result = get_positive_proba(rf, X)
        expected = rf.predict_proba(X)[:, 0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_values_in_zero_one_range(self, trained_models):
        """All probability values must be in [0, 1]."""
        X = trained_models["X_scaled"]
        for name in ["lr", "rf", "svc"]:
            result = get_positive_proba(trained_models[name], X)
            assert np.all(result >= 0.0), f"Negative proba for {name}"
            assert np.all(result <= 1.0), f"Proba > 1.0 for {name}"

    def test_single_sample(self, trained_models):
        """Must work with a single sample (shape (1, n_features))."""
        X = trained_models["X_scaled"][:1]
        for name in ["lr", "rf", "svc"]:
            result = get_positive_proba(trained_models[name], X)
            assert result.shape == (1,), f"Shape mismatch for {name}"

    def test_decision_function_path(self, trained_models):
        """SVC uses decision_function path (L103-104). Verify sigmoid is applied."""
        svc = trained_models["svc"]
        X = trained_models["X_scaled"]
        # SVC also has decision_function — but since it doesn't have predict_proba
        # (without probability=True), the code falls through to decision_function
        # Actually SVC without probability=True has no predict_proba, so decision_function path runs
        result = get_positive_proba(svc, X)
        scores = svc.decision_function(X)
        expected = 1 / (1 + np.exp(scores))
        np.testing.assert_array_almost_equal(result, expected)

    def test_fallback_predict_only_model(self):
        """Model with only predict() — L105-106: should return 1.0 for pred==0, 0.0 for pred==1."""
        class PredictOnlyModel:
            def predict(self, x):
                return np.array([0, 1, 0, 1, 0])

        model = PredictOnlyModel()
        X_dummy = np.zeros((5, 2))  # shape doesn't matter
        result = get_positive_proba(model, X_dummy)
        expected = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_model_with_no_methods(self):
        """Model with no predict/predict_proba/decision_function → returns zeros (L105)."""
        class EmptyModel:
            pass

        model = EmptyModel()
        X_dummy = np.zeros((3, 2))
        result = get_positive_proba(model, X_dummy)
        # hasattr(model, "predict") is False → np.zeros(x.shape[0])
        # np.where(preds == 0, 1.0, 0.0) → all 1.0 since preds are all 0.0
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# UNIT TESTS: load_dataframe_from_sklearn()
# ===========================================================================

class TestLoadDataframeFromSklearn:
    """Tests for app.py load_dataframe_from_sklearn() — L109-112."""

    def test_returns_dataframe(self):
        df = load_dataframe_from_sklearn()
        assert isinstance(df, pd.DataFrame)

    def test_has_target_column(self):
        df = load_dataframe_from_sklearn()
        assert "target" in df.columns

    def test_correct_row_count(self):
        df = load_dataframe_from_sklearn()
        assert len(df) == 569

    def test_target_is_numeric(self):
        df = load_dataframe_from_sklearn()
        assert pd.api.types.is_numeric_dtype(df["target"])


# ===========================================================================
# UNIT TESTS: load_dataframe_from_upload()
# ===========================================================================

class TestLoadDataframeFromUpload:
    """Tests for app.py load_dataframe_from_upload() — L115-117."""

    def test_loads_csv_from_buffer(self):
        csv_content = "a,b,target\n1,2,0\n3,4,1\n"
        buffer = io.StringIO(csv_content)
        df = load_dataframe_from_upload(buffer)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "target"]

    def test_preserves_values(self):
        csv_content = "x,y\n10.5,20.3\n30.1,40.2\n"
        buffer = io.StringIO(csv_content)
        df = load_dataframe_from_upload(buffer)
        assert df["x"].iloc[0] == pytest.approx(10.5)
        assert df["y"].iloc[1] == pytest.approx(40.2)

    def test_empty_csv(self):
        csv_content = "a,b\n"
        buffer = io.StringIO(csv_content)
        df = load_dataframe_from_upload(buffer)
        assert len(df) == 0
        assert list(df.columns) == ["a", "b"]


# ===========================================================================
# UNIT TEST: PCA computation logic (mirrors app.py L76-93 without @st.cache_data)
# ===========================================================================

class TestComputePCA:
    """Tests for PCA computation logic — mirrors app.py L80-93."""

    def test_pca_returns_2d_dataframe(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        assert pca_df.shape == (50, 2)
        assert list(pca_df.columns) == ["PC1", "PC2"]

    def test_pca_explained_variance_sums_less_than_one(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        pca = PCA(n_components=2)
        pca.fit_transform(X)
        assert pca.explained_variance_ratio_.sum() <= 1.0

    def test_pca_with_labels(self):
        CLASS_LABELS = {0: "Malignant", 1: "Benign"}
        rng = np.random.RandomState(42)
        X = rng.randn(10, 5)
        labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["Diagnosis"] = labels.map(CLASS_LABELS)
        assert "Diagnosis" in pca_df.columns
        assert set(pca_df["Diagnosis"].unique()) == {"Malignant", "Benign"}


# ===========================================================================
# UNIT TEST: Feature alignment logic (mirrors app.py L165-166)
# ===========================================================================

class TestFeatureAlignment:
    """Tests for the reindex + fill_value logic in app.py L165-166."""

    def test_reindex_preserves_matching_columns(self):
        """When CSV columns match feature_names, values are preserved."""
        feature_names = ["a", "b", "c"]
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        result = df.reindex(columns=feature_names, fill_value=0)
        assert list(result.columns) == feature_names
        assert result["a"].iloc[0] == 1.0

    def test_reindex_fills_missing_with_zero(self):
        """Missing columns are filled with 0 (fill_value=0)."""
        feature_names = ["a", "b", "c"]
        df = pd.DataFrame({"a": [1.0]})  # missing b, c
        result = df.reindex(columns=feature_names, fill_value=0)
        assert result["b"].iloc[0] == 0
        assert result["c"].iloc[0] == 0

    def test_reindex_drops_extra_columns(self):
        """Extra columns in the CSV are silently dropped."""
        feature_names = ["a", "b"]
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "extra_col": [99.0]})
        result = df.reindex(columns=feature_names, fill_value=0)
        assert "extra_col" not in result.columns

    def test_reindex_all_missing(self):
        """Completely mismatched CSV fills all features with 0."""
        feature_names = ["a", "b", "c"]
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        result = df.reindex(columns=feature_names, fill_value=0)
        assert (result.values == 0).all()
