"""High-level auto-optimizer orchestrating OpenAI recommendations."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.evaluation.openai_optimizer import OpenAIOptimizer
from src.strategy.strategy_manager import StrategyManager


class AIAutoOptimizer:
    """Wraps OpenAI optimizer and applies safe config tweaks + persistence."""

    def __init__(
        self,
        *,
        state_manager: Any,
        strategy_manager: StrategyManager,
        openai_config: Dict[str, Any],
    ) -> None:
        self.state_manager = state_manager
        self.strategy_manager = strategy_manager
        self.client = OpenAIOptimizer(
            state_manager,
            model_name=openai_config.get("model", "gpt-4.1-mini"),
            max_tokens=openai_config.get("max_tokens", 1500),
        )

    def run(self, daily_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full loop: build payload, request, interpret, apply."""
        payload = self.client.build_payload(daily_summary)
        response = self.client.request_recommendations(payload)
        parsed = self.client.interpret_response(response)
        suggested_changes = parsed.get("suggested_changes") or self._legacy_to_suggestions(parsed)
        applied_changes = self._apply_suggestions(suggested_changes or {})
        return {
            "suggested_changes": suggested_changes,
            "applied_changes": applied_changes,
            "raw_response": parsed,
        }

    def _apply_suggestions(self, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        if not suggestions:
            return {}
        active_strategy = self.strategy_manager.get_active_strategy_name()
        self.strategy_manager.update_strategy_config(active_strategy, suggestions)
        # Mirror changes to system parameters for audibility.
        params = self.state_manager.load_parameters()
        params.setdefault("strategy_adjustments", {})[active_strategy] = suggestions
        self.state_manager.save_parameters(params)
        return suggestions

    @staticmethod
    def _legacy_to_suggestions(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert older OpenAIOptimizer format to the new schema."""
        legacy = payload.get("recommendations", [])
        changes: Dict[str, Any] = {}
        for item in legacy:
            parameter = item.get("parameter")
            value = item.get("value")
            action = item.get("action")
            if parameter is None or value is None:
                continue
            if action == "increase":
                changes[parameter] = changes.get(parameter, 0) + value
            elif action == "decrease":
                changes[parameter] = changes.get(parameter, 0) - value
            else:
                changes[parameter] = value
        return changes
