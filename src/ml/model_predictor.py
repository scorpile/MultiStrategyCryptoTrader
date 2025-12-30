"""Prediction helpers for LightGBM/XGBoost style models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .model_manager import ModelManager


class TrendModelPredictor:
    """Loads persisted boosters and produces probability forecasts."""

    def __init__(self, model_manager: ModelManager, *, model_name: Optional[str] = None) -> None:
        self.model_manager = model_manager
        self.model_name = model_name
        self.runtime_model: Any | None = None
        self.metadata: Dict[str, Any] = {}

    def load(self, model_name: Optional[str] = None) -> None:
        target_name = model_name or self.model_name
        if target_name is None:
            available = self.model_manager.list_available_models()
            if not available:
                # Create a dummy model if none exist
                self._create_dummy_model()
                return
            target_name = available[-1]
        model, metadata = self.model_manager.load_model(target_name)
        self.runtime_model = model
        self.metadata = metadata
        self.model_name = target_name

    def _create_dummy_model(self) -> None:
        """Create a simple dummy model that predicts 0.5 (neutral)."""
        import lightgbm as lgb
        # Dummy dataset
        dummy_feature_names = [
            "close_return_1",
            "close_return_5",
            "rolling_mean_5",
            "rolling_mean_20",
            "rolling_std_20",
            "volume_zscore_20",
            "high_low_range",
            "close_position_20",
            "vwap_deviation",
            "momentum_3",
            "momentum_5",
            "volume_ratio_5",
        ]
        dummy_features = pd.DataFrame([[0.0] * len(dummy_feature_names)], columns=dummy_feature_names)
        dummy_labels = pd.Series([0.5])
        dummy_dataset = lgb.Dataset(dummy_features, label=dummy_labels)
        
        # Train a minimal model
        params = {'objective': 'regression', 'verbosity': -1}
        dummy_booster = lgb.train(params, dummy_dataset, num_boost_round=1)
        
        metadata = {
            "name": "dummy_model",
            "params": params,
            "metrics": {"accuracy": 0.5, "roc_auc": 0.5},
            "num_features": len(dummy_feature_names),
            "feature_names": dummy_feature_names,
            "is_dummy": True,
        }
        self.model_manager.save_model(dummy_booster, metadata)
        self.runtime_model = dummy_booster
        self.metadata = metadata
        self.model_name = "dummy_model"

    def _expected_feature_names(self) -> list[str]:
        names = self.metadata.get("feature_names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
        if self.runtime_model is not None:
            try:
                model_names = self.runtime_model.feature_name()
                if model_names:
                    return [str(n) for n in model_names]
            except Exception:
                pass
        return []

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        expected = self._expected_feature_names()
        if expected:
            aligned = features.reindex(columns=expected, fill_value=0.0)
            return aligned
        # Fallback for legacy models without metadata.
        legacy = [
            "close_return_1",
            "close_return_5",
            "rolling_mean_5",
            "rolling_mean_20",
            "rolling_std_20",
            "volume_zscore_20",
            "high_low_range",
            "close_position_20",
        ]
        available = [f for f in legacy if f in features.columns]
        return features[available]

    def predict_probabilities(self, features: pd.DataFrame) -> pd.Series:
        if self.runtime_model is None:
            self.load(self.model_name)
        if features.empty:
            return pd.Series(dtype=float)
        features_subset = self._align_features(features)
        preds = self.runtime_model.predict(features_subset)
        return pd.Series(preds, index=features.index)

    def build_signal_context(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Return a dict consumed by StrategyManager / UI with ML confidence."""
        probabilities = self.predict_probabilities(features)
        if probabilities.empty:
            return {}
        latest_prob = float(probabilities.iloc[-1])
        confidence = abs(latest_prob - 0.5) * 2
        return {
            "probability_up": latest_prob,
            "confidence": confidence,
            "model_name": self.model_name,
            "timestamp": features.index[-1].isoformat() if hasattr(features.index, "tz") else str(features.index[-1]),
        }
