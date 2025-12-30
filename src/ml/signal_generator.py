"""Signal generation helpers combining ML probabilities with technical context."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class SignalGenerator:
    """Produces buy/sell/hold decisions with confidence scores."""

    def __init__(
        self,
        model_manager: Any,
        *,
        decision_threshold: float = 0.55,
    ) -> None:
        """Store dependencies so trained models can be loaded lazily."""
        self.model_manager = model_manager
        self.decision_threshold = decision_threshold
        self.runtime_model: Any | None = None
        self.runtime_metadata: Dict[str, Any] = {}
        self.last_signal_summary: Dict[str, Any] | None = None

    def load_runtime_context(self, model_name: Optional[str] = None) -> None:
        """Load the latest model, thresholds, and metadata before inference."""
        if model_name is None:
            available = self.model_manager.list_available_models()
            if not available:
                raise RuntimeError("No trained models found. Train and save a model first.")
            model_name = available[-1]
        model, metadata = self.model_manager.load_model(model_name)
        self.runtime_model = model
        self.runtime_metadata = metadata

    def generate_signals(
        self,
        features: pd.DataFrame,
        technical_decisions: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return structured signal dictionaries with confidence scores."""
        if self.runtime_model is None:
            raise RuntimeError("Runtime model not loaded. Call load_runtime_context() first.")
        if features.empty:
            return []
        probabilities = self._predict_probabilities(features)
        signals: List[Dict[str, Any]] = []
        for idx, prob in enumerate(probabilities):
            tech = technical_decisions[idx] if idx < len(technical_decisions) else {"decision": "hold"}
            decision, reasons = self._merge_decisions(prob, tech.get("decision", "hold"))
            signal = {
                "index": idx,
                "probability_up": prob,
                "technical_decision": tech.get("decision", "hold"),
                "final_decision": decision,
                "confidence": abs(prob - 0.5) * 2,
                "reasons": reasons + tech.get("reasons", []),
            }
            signals.append(signal)
        if signals:
            self.last_signal_summary = signals[-1]
        return signals

    def explain_last_signal(self) -> Dict[str, Any]:
        """Provide context (feature importances, thresholds) for observability."""
        return self.last_signal_summary or {}

    def _predict_probabilities(self, features: pd.DataFrame) -> List[float]:
        """Use the runtime model to compute upward probability."""
        model = self.runtime_model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(features)
            probs = 1 / (1 + np.exp(-decision))
        else:
            raise RuntimeError("Loaded model does not expose predict_proba or decision_function.")
        return probs.tolist()

    def _merge_decisions(self, ml_prob: float, tech_decision: str) -> tuple[str, List[str]]:
        """Blend ML probabilities with technical decisions."""
        reasons: List[str] = []
        if ml_prob >= self.decision_threshold and tech_decision == "buy":
            reasons.append(
                f"ML probability {ml_prob:.2f} >= threshold {self.decision_threshold:.2f} and technical says buy."
            )
            return "buy", reasons
        if ml_prob <= (1 - self.decision_threshold) and tech_decision == "sell":
            reasons.append(
                f"ML probability {ml_prob:.2f} <= short threshold {1 - self.decision_threshold:.2f} and technical says sell."
            )
            return "sell", reasons
        reasons.append("Signals diverged; defaulting to hold.")
        return "hold", reasons
