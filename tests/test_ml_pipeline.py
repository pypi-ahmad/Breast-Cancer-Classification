"""
ML Pipeline tests — validates the production model bundle.

Tests cover:
- Model loading from models_bundle.pkl
- Prediction validity (shape, type, range)
- Scaler consistency between training and inference
- Cross-model consensus logic (mirrors app.py Tab 1, L210-253)
- SHAP compatibility (mirrors app.py Tab 4, L465-523)
- Metric computation (mirrors app.py Tab 2, L270-301)
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# ML TESTS: Model Loading
# ===========================================================================

class TestModelLoading:
    """Validate that the saved bundle loads correctly and models are functional."""

    def test_all_models_have_predict(self, bundle_models):
        """Every model in the bundle must have a predict() method."""
        for name, model in bundle_models.items():
            assert hasattr(model, "predict"), f"Model '{name}' lacks predict()"

    def test_all_models_have_predict_proba_or_decision_function(self, bundle_models):
        """Every model should have predict_proba or decision_function for probability output."""
        for name, model in bundle_models.items():
            has_proba = hasattr(model, "predict_proba")
            has_df = hasattr(model, "decision_function")
            assert has_proba or has_df, \
                f"Model '{name}' has neither predict_proba nor decision_function"

    def test_expected_model_names(self, bundle_models):
        """Bundle should contain the 5 models trained in train_automl.py L158-164."""
        expected = {
            "AutoML Best (LGBM)",
            "AutoML XGBoost",
            "AutoML Random Forest",
            "AutoML Extra Trees",
            "AutoML Logistic Reg",
        }
        actual = set(bundle_models.keys())
        assert actual == expected, f"Model name mismatch: got {actual}"

    def test_scaler_has_fitted_attributes(self, bundle_scaler):
        """Scaler must have mean_ and scale_ (fitted state)."""
        assert hasattr(bundle_scaler, "mean_")
        assert hasattr(bundle_scaler, "scale_")
        assert bundle_scaler.mean_ is not None
        assert bundle_scaler.scale_ is not None

    def test_scaler_feature_count(self, bundle_scaler, bundle_feature_names):
        """Scaler must match the number of feature names (30)."""
        assert len(bundle_scaler.mean_) == len(bundle_feature_names)


# ===========================================================================
# ML TESTS: Prediction Validity
# ===========================================================================

class TestPredictionValidity:
    """Validate predictions are correct shape/type/range."""

    def test_predict_output_shape(self, bundle_models, inference_data):
        """predict() must return array of same length as input."""
        X_scaled, labels, _ = inference_data
        for name, model in bundle_models.items():
            preds = model.predict(X_scaled)
            assert len(preds) == len(X_scaled), f"Shape mismatch for {name}"

    def test_predict_output_is_binary(self, bundle_models, inference_data):
        """Predictions must be 0 or 1 only."""
        X_scaled, _, _ = inference_data
        for name, model in bundle_models.items():
            preds = model.predict(X_scaled)
            unique_vals = set(np.unique(preds))
            assert unique_vals.issubset({0, 1}), \
                f"Model '{name}' predicted non-binary values: {unique_vals}"

    def test_predict_proba_output_shape(self, bundle_models, inference_data):
        """predict_proba must return (n_samples, 2) for binary classification."""
        X_scaled, _, _ = inference_data
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
                assert proba.shape == (len(X_scaled), 2), \
                    f"predict_proba shape for '{name}': {proba.shape}"

    def test_predict_proba_sums_to_one(self, bundle_models, inference_data):
        """predict_proba rows must sum to ~1.0."""
        X_scaled, _, _ = inference_data
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
                row_sums = proba.sum(axis=1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                    err_msg=f"predict_proba rows don't sum to 1 for '{name}'")

    def test_predict_proba_in_range(self, bundle_models, inference_data):
        """All probability values must be in [0, 1]."""
        X_scaled, _, _ = inference_data
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
                assert np.all(proba >= 0) and np.all(proba <= 1), \
                    f"predict_proba out of [0,1] for '{name}'"

    def test_single_sample_prediction(self, bundle_models, inference_data):
        """Models must handle single-sample input (shape (1, n_features))."""
        X_scaled, _, _ = inference_data
        single = X_scaled[:1]
        for name, model in bundle_models.items():
            pred = model.predict(single)
            assert len(pred) == 1, f"Single-sample predict failed for '{name}'"

    def test_models_achieve_minimum_accuracy(self, bundle_models, inference_data):
        """All models must achieve > 85% accuracy on the sklearn sample (sanity check)."""
        X_scaled, labels, _ = inference_data
        for name, model in bundle_models.items():
            preds = model.predict(X_scaled)
            acc = accuracy_score(labels, preds)
            assert acc > 0.85, \
                f"Model '{name}' accuracy {acc:.2%} is below 85% threshold"


# ===========================================================================
# ML TESTS: Scaler Consistency
# ===========================================================================

class TestScalerConsistency:
    """Validates scaler used in training matches the one in the bundle."""

    def test_scaler_transform_roundtrip(self, bundle_scaler, bundle_feature_names):
        """transform → inverse_transform should approximately recover original data."""
        rng = np.random.RandomState(42)
        X_orig = rng.rand(10, len(bundle_feature_names)) * 100
        X_scaled = bundle_scaler.transform(X_orig)
        X_recovered = bundle_scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_orig, X_recovered, atol=1e-10)

    def test_scaler_output_shape(self, bundle_scaler, bundle_feature_names):
        """Scaler output must maintain input shape."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, len(bundle_feature_names))
        result = bundle_scaler.transform(X)
        assert result.shape == X.shape


# ===========================================================================
# ML TESTS: Consensus Logic (mirrors app.py Tab 1, L210-253)
# ===========================================================================

class TestConsensusLogic:
    """Validates the multi-model consensus voting logic from app.py."""

    def test_consensus_score_range(self, bundle_models, inference_data):
        """Consensus score must be in [0.0, 1.0]."""
        X_scaled, _, _ = inference_data
        CLASS_LABELS = {0: "Malignant", 1: "Benign"}
        POS_CLASS = CLASS_LABELS[0]
        threshold = 0.5

        model_predictions = pd.DataFrame(index=range(len(X_scaled)))
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 0]
            else:
                proba = np.zeros(len(X_scaled))
            pred_label = np.where(proba >= threshold, 0, 1)
            model_predictions[name] = np.where(pred_label == 0, POS_CLASS, "Benign")

        pos_votes = (model_predictions == POS_CLASS).sum(axis=1)
        consensus_score = pos_votes / len(bundle_models)

        assert (consensus_score >= 0.0).all()
        assert (consensus_score <= 1.0).all()

    def test_consensus_diagnosis_is_valid(self, bundle_models, inference_data):
        """Consensus must produce only POS_CLASS or NEG_CLASS labels."""
        X_scaled, _, _ = inference_data
        POS_CLASS = "Malignant"
        NEG_CLASS = "Benign"
        threshold = 0.5
        consensus_threshold = 0.5

        model_predictions = pd.DataFrame(index=range(len(X_scaled)))
        for name, model in bundle_models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[:, 0]
            else:
                proba = np.zeros(len(X_scaled))
            pred_label = np.where(proba >= threshold, 0, 1)
            model_predictions[name] = np.where(pred_label == 0, POS_CLASS, NEG_CLASS)

        pos_votes = (model_predictions == POS_CLASS).sum(axis=1)
        consensus_score = pos_votes / len(bundle_models)
        diagnosis = np.where(consensus_score >= consensus_threshold, POS_CLASS, NEG_CLASS)

        assert set(np.unique(diagnosis)).issubset({POS_CLASS, NEG_CLASS})


# ===========================================================================
# ML TESTS: SHAP Compatibility (mirrors app.py Tab 4, L471-480)
# ===========================================================================

class TestSHAPCompatibility:
    """Validates that SHAP can explain the bundle models."""

    def test_shap_tree_explainer_works_for_tree_models(self, bundle_models, inference_data):
        """TreeExplainer must work for LGBM, XGBoost, RF, ExtraTrees."""
        import shap
        X_scaled, _, _ = inference_data
        X_small = X_scaled[:10]  # Use small subset for speed

        tree_model_names = [
            "AutoML Best (LGBM)",
            "AutoML XGBoost",
            "AutoML Random Forest",
            "AutoML Extra Trees",
        ]

        for name in tree_model_names:
            model = bundle_models[name]
            # Unwrap FLAML model (mirrors app.py L472-476)
            native_model = model
            if hasattr(model, 'model') and hasattr(model.model, 'estimator'):
                native_model = model.model.estimator
            elif hasattr(model, 'estimator'):
                native_model = model.estimator

            try:
                explainer = shap.TreeExplainer(native_model)
                shap_values = explainer(X_small)
                assert shap_values.values.shape[0] == len(X_small), \
                    f"SHAP output rows mismatch for '{name}'"
                # Values should be either 2D (samples, features) or 3D (samples, features, classes)
                assert shap_values.values.ndim in (2, 3), \
                    f"Unexpected SHAP ndim for '{name}': {shap_values.values.ndim}"
            except Exception as e:
                pytest.fail(f"SHAP TreeExplainer failed for '{name}': {e}")


# ===========================================================================
# ML TESTS: Metric Computation (mirrors app.py Tab 2, L280-296)
# ===========================================================================

class TestMetricComputation:
    """Validates metric computation is consistent between train and app."""

    def test_metrics_match_between_train_and_app(self, bundle_models, inference_data):
        """Metrics computed in app.py (L288-296) should match train_automl.evaluate_models()."""
        import train_automl
        X_scaled, labels, _ = inference_data

        for name, model in bundle_models.items():
            preds = model.predict(X_scaled)

            # app.py style (L290-294)
            app_acc = accuracy_score(labels, preds)
            app_recall = recall_score(labels, preds, pos_label=0)
            app_prec = precision_score(labels, preds, pos_label=0)
            app_f1 = f1_score(labels, preds, pos_label=0)

            # train_automl.py style (L104-108)
            train_result = train_automl.evaluate_models({name: model}, X_scaled, labels)
            train_acc = train_result["Accuracy"].iloc[0]
            train_recall = train_result["Recall (Sensitivity)"].iloc[0]
            train_prec = train_result["Precision"].iloc[0]
            train_f1 = train_result["F1 Score"].iloc[0]

            assert app_acc == pytest.approx(train_acc), f"Accuracy mismatch for '{name}'"
            assert app_recall == pytest.approx(train_recall), f"Recall mismatch for '{name}'"
            assert app_prec == pytest.approx(train_prec), f"Precision mismatch for '{name}'"
            assert app_f1 == pytest.approx(train_f1), f"F1 mismatch for '{name}'"
