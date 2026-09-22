"""Lazy, serializable adapters for optional tabular foundation models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class MitraV2ClassifierAdapter:
    """Expose an AutoGluon Mitra predictor through the dashboard model API."""

    classes_ = np.array([0, 1])
    supports_tree_shap = False
    research_only = False

    def __init__(self, predictor_path: str, feature_names: list[str]) -> None:
        self.predictor_path = predictor_path
        self.feature_names = feature_names
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from autogluon.tabular import TabularPredictor

            self._predictor = TabularPredictor.load(self.predictor_path)
        return self._predictor

    def _frame(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, columns=self.feature_names)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._load().predict(self._frame(values)), dtype=int)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probabilities = self._load().predict_proba(self._frame(values))
        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.reindex(columns=self.classes_).to_numpy()
        return np.asarray(probabilities, dtype=float)

    def get_params(self, deep: bool = True) -> dict:
        return {"model": "autogluon/mitra-classifier-2", "fine_tune": False}

    def release_runtime(self) -> None:
        self._predictor = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_predictor"] = None
        return state


class TabFMClassifierAdapter:
    """Load TabFM only when inference is requested and keep it out of bundles."""

    classes_ = np.array([0, 1])
    supports_tree_shap = False
    research_only = True

    def __init__(
        self,
        checkpoint_path: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 4,
        max_num_rows: int = 100,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self._classifier = None

    def _load(self):
        if self._classifier is None:
            from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

            model = tabfm_v1_0_0_pytorch.load(
                model_type="classification",
                checkpoint_path=Path(self.checkpoint_path),
                device="cpu",
            )
            self._classifier = TabFMClassifier(
                model,
                n_estimators=self.n_estimators,
                max_num_rows=self.max_num_rows,
                batch_size=1,
                random_state=42,
            )
            self._classifier.fit(self.x_train, self.y_train)
        return self._classifier

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._load().predict(values), dtype=int)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._load().predict_proba(values), dtype=float)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "model": "google/tabfm-1.0.0-pytorch",
            "n_estimators": self.n_estimators,
            "max_num_rows": self.max_num_rows,
            "device": "cpu",
        }

    def release_runtime(self) -> None:
        self._classifier = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_classifier"] = None
        return state
