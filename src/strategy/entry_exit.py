"""Legacy compatibility wrapper exposing the prior EntryExit API."""

from __future__ import annotations

from typing import Dict, List

from .moderate import ModerateStrategy


class EntryExitDecider(ModerateStrategy):
    """For backward compatibility: behaves like the new ModerateStrategy."""

    def __init__(self, risk_manager, config: Dict | None = None) -> None:
        super().__init__(config=config, risk_manager=risk_manager)

    def build_orders(self, indicators: Dict) -> List[Dict]:
        """Translate Modern signal format into the original list-based output."""
        signal = self.generate_signal(indicators)
        decision = {
            "decision": signal.get("decision", "hold"),
            "price": signal.get("price"),
            "suggested_size": signal.get("suggested_size", 0.0),
            "reasons": signal.get("reasons", []),
            "metadata": signal.get("metadata", {}),
        }
        return [decision]
