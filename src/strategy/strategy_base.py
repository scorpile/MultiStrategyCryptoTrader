"""Base abstractions shared by every hot-swappable trading strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseStrategy(ABC):
    """Common helpers so concrete strategies stay tiny and declarative."""

    description: str = ""
    BASE_DEFAULTS: Dict[str, float] = {
        "min_position_size": 0.05,
        "max_position_size": 0.5,
        "base_position_multiplier": 10.0,
        "stop_loss_pct": 0.05,  # 5% stop loss
        "take_profit_pct": 0.10,  # 10% take profit
    }

    def __init__(self, *, config: Optional[Dict[str, Any]] = None, risk_manager: Optional[Any] = None) -> None:
        """Compose runtime configuration combining defaults + overrides."""
        defaults: Dict[str, Any] = dict(self.BASE_DEFAULTS)
        defaults.update(self.config_schema())
        if config:
            defaults.update(config)
        self.config = defaults
        self.risk_manager = risk_manager

    @abstractmethod
    def name(self) -> str:
        """Return the canonical strategy identifier."""

    def config_schema(self) -> Dict[str, Any]:
        """Return default configuration values for UI rendering."""
        return {}

    @abstractmethod
    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a normalized signal dict consumed by the scheduler."""

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def update_config(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        """Hot-update runtime config (used by OpenAI optimizer and UI)."""
        if overrides:
            self.config.update(overrides)

    def _build_signal(
        self,
        *,
        decision: str,
        price: float,
        confidence: float,
        suggested_size: float,
        reasons: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a scheduler-friendly payload with consistent keys."""
        return {
            "strategy": self.name(),
            "decision": decision,
            "price": float(price),
            "confidence": max(0.0, min(1.0, confidence)),
            "suggested_size": round(max(0.0, suggested_size), 6),
            "reasons": reasons or [],
            "metadata": metadata or {},
        }

    def _volatility_position_size(
        self,
        *,
        atr_value: float,
        price: float,
        aggressiveness: float = 1.0,
    ) -> float:
        """Derive a position size anchored to ATR % of price."""
        if price <= 0:
            return self.config["min_position_size"]
        atr_pct = atr_value / price if atr_value and price else 0.01
        base = atr_pct * self.config["base_position_multiplier"] * aggressiveness
        bounded = max(self.config["min_position_size"], min(self.config["max_position_size"], base))
        return round(bounded, 4)

    @staticmethod
    def _latest(series: Optional[list[float]], default: float = 0.0) -> float:
        """Return the last element of a numeric sequence."""
        if not series:
            return default
        return float(series[-1])
