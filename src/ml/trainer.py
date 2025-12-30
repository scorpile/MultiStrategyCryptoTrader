"""Training helpers for the ML components (classification/regression)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainingResult:
    """Container for a fitted model and its evaluation metrics."""

    model: Any
    metrics: Dict[str, float]
    report: str


class ModelTrainer:
    """Handles preprocessing, fitting, evaluation, and artifact creation."""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        hyperparams: Dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        """Store the selected model family and hyperparameters."""
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        self.random_state = random_state

    def train(self, features: Any, labels: Any, *, test_size: float = 0.2) -> TrainingResult:
        """Fit the selected model using the provided dataset."""
        X_train, X_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=test_size,
            shuffle=False,
        )

        model = self._build_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = np.zeros_like(y_val, dtype=float)

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "roc_auc": float(roc_auc_score(y_val, y_prob)) if len(set(y_val)) > 1 else 0.0,
        }
        report = classification_report(y_val, y_pred, zero_division=0)

        return TrainingResult(model=model, metrics=metrics, report=report)

    def cross_validate(self, features: Any, labels: Any, folds: int = 3) -> Dict[str, float]:
        """Return basic validation metrics to track model performance."""
        # Simple rolling-window CV: split dataset into `folds` chronological chunks.
        fold_metrics: Dict[str, float] = {"accuracy": 0.0}
        chunk_size = max(len(features) // folds, 1)
        accuracies = []
        for fold in range(folds):
            start = fold * chunk_size
            end = start + chunk_size
            X_train = features[:start]
            y_train = labels[:start]
            X_val = features[start:end]
            y_val = labels[start:end]
            if len(X_val) == 0 or len(X_train) == 0:
                continue
            model = self._build_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracies.append(float(accuracy_score(y_val, y_pred)))
        if accuracies:
            fold_metrics["accuracy"] = float(np.mean(accuracies))
        return fold_metrics

    def _build_model(self) -> Any:
        """Instantiate the requested model family."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=self.hyperparams.get("max_iter", 1000),
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported model type: {self.model_type}")
