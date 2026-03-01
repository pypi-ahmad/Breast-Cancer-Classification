"""
Shared fixtures for the Breast Cancer Classification test suite.

All fixtures are derived from actual code in train_automl.py and app.py.
No assumptions — only code-verified behavior.
"""
import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT_DIR, "models_bundle.pkl")


# ---------------------------------------------------------------------------
# Raw data fixtures (mirrors train_automl.py load_data, L48-54)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def raw_dataframe():
    """Load the breast cancer dataset exactly as train_automl.py load_data() does."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


@pytest.fixture(scope="session")
def feature_target_split(raw_dataframe):
    """Split into X, y exactly as train_automl.py main() L131-133."""
    X = raw_dataframe.drop(columns=["target"])
    y = raw_dataframe["target"]
    return X, y


@pytest.fixture(scope="session")
def train_test_data(feature_target_split):
    """Split + scale as in train_automl.py main() L141-154."""
    X, y = feature_target_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns.tolist()
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# Bundle fixture (loads the saved models_bundle.pkl if it exists)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def bundle():
    """Load the production model bundle from disk."""
    if not os.path.exists(BUNDLE_PATH):
        pytest.skip("models_bundle.pkl not found — run train_automl.py first")
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="session")
def bundle_models(bundle):
    return bundle["models"]


@pytest.fixture(scope="session")
def bundle_scaler(bundle):
    return bundle["scaler"]


@pytest.fixture(scope="session")
def bundle_feature_names(bundle):
    return bundle["feature_names"]


@pytest.fixture(scope="session")
def bundle_metadata(bundle):
    return bundle.get("metadata", {})


# ---------------------------------------------------------------------------
# Scaled inference data (uses bundle scaler, mirrors app.py L165-168)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def inference_data(bundle_scaler, bundle_feature_names):
    """Prepare scaled data exactly as app.py does for the sklearn sample."""
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    labels = df["target"].astype(int)
    df_features = df.drop(columns=["target"])
    df_features = df_features.reindex(columns=bundle_feature_names, fill_value=0)
    X_scaled = bundle_scaler.transform(df_features)
    return X_scaled, labels, df_features


# ---------------------------------------------------------------------------
# Tiny synthetic data for fast unit tests
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_scaled_data(bundle_feature_names):
    """5 synthetic samples, correct number of features, scaled-like values."""
    rng = np.random.RandomState(0)
    n_features = len(bundle_feature_names)
    X = rng.randn(5, n_features)
    return X
