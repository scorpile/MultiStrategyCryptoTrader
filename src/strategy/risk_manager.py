"""Risk management helpers for the simulated strategy.

This module must never place real exchange orders; it only computes limits and sizing
for the simulation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PositionSizingResult:
    quantity: float
    risk_amount_usdt: float
    stop_distance_usdt: float
    capped_by_max_notional: bool


class RiskManager:
    """Evaluates trade proposals against portfolio- and trade-level constraints."""

    def __init__(
        self,
        max_position_size_usdt: float,
        max_daily_loss_usdt: float,
        *,
        default_risk_pct_per_trade: float = 0.01,
        min_stop_distance_pct: float = 0.001,
    ) -> None:
        """Store baseline guardrails pulled from configuration/state manager."""
        self.max_position_size_usdt = max_position_size_usdt
        self.max_daily_loss_usdt = max_daily_loss_usdt
        self.default_risk_pct_per_trade = float(default_risk_pct_per_trade)
        self.min_stop_distance_pct = float(min_stop_distance_pct)

    def size_position_for_risk(
        self,
        *,
        equity_usdt: float,
        entry_price: float,
        stop_loss_price: float,
        risk_pct_per_trade: Optional[float] = None,
        max_position_size_usdt: Optional[float] = None,
        max_quantity: Optional[float] = None,
    ) -> PositionSizingResult:
        """Compute quantity (base units) such that loss at stop ≈ risk% of equity.

        Returns a `PositionSizingResult` that can be attached to order metadata.
        """
        equity_usdt = float(equity_usdt)
        entry_price = float(entry_price)
        stop_loss_price = float(stop_loss_price)
        risk_pct = self.default_risk_pct_per_trade if risk_pct_per_trade is None else float(risk_pct_per_trade)
        notional_cap = self.max_position_size_usdt if max_position_size_usdt is None else float(max_position_size_usdt)

        if equity_usdt <= 0 or entry_price <= 0:
            return PositionSizingResult(quantity=0.0, risk_amount_usdt=0.0, stop_distance_usdt=0.0, capped_by_max_notional=False)

        stop_distance = abs(entry_price - stop_loss_price)
        min_stop_distance = max(entry_price * self.min_stop_distance_pct, 1e-9)
        stop_distance = max(stop_distance, min_stop_distance)

        risk_amount = max(0.0, equity_usdt * max(0.0, risk_pct))
        quantity = risk_amount / stop_distance if stop_distance > 0 else 0.0

        cap_qty = notional_cap / entry_price if notional_cap > 0 else quantity
        capped = False
        if cap_qty > 0 and quantity > cap_qty:
            quantity = cap_qty
            capped = True

        if max_quantity is not None and max_quantity > 0 and quantity > float(max_quantity):
            quantity = float(max_quantity)

        return PositionSizingResult(
            quantity=round(max(0.0, quantity), 6),
            risk_amount_usdt=round(risk_amount, 6),
            stop_distance_usdt=round(stop_distance, 6),
            capped_by_max_notional=capped,
        )

    def apply_stop_take_rules(self, trade_state: Dict[str, float]) -> Dict[str, float]:
        """Return updated trade state after enforcing stop-loss/take-profit rules."""
        raise NotImplementedError("Stop/take logic will be added in integration.")

    def assess_drawdown_limits(self, current_drawdown: float) -> bool:
        """Signal whether global drawdown rules require pausing trading."""
        raise NotImplementedError("Drawdown assessment has not been implemented.")

    @staticmethod
    def compute_stop_loss_price(*, entry_price: float, stop_loss_pct: float) -> float:
        entry_price = float(entry_price)
        stop_loss_pct = float(stop_loss_pct)
        if entry_price <= 0:
            return 0.0
        return max(0.0, entry_price * (1.0 - max(0.0, stop_loss_pct)))

    @staticmethod
    def compute_take_profit_price(*, entry_price: float, take_profit_pct: float) -> float:
        entry_price = float(entry_price)
        take_profit_pct = float(take_profit_pct)
        if entry_price <= 0:
            return 0.0
        return max(0.0, entry_price * (1.0 + max(0.0, take_profit_pct)))

    def can_open_trade(self, proposed_trade: Dict[str, Any]) -> bool:
        """Return True when the proposed trade respects configured hard limits.

        This is a conservative guardrail. Strategy logic decides *when* to trade.
        """
        try:
            entry_price = float(proposed_trade.get("entry_price") or proposed_trade.get("price") or 0.0)
            quantity = float(proposed_trade.get("quantity") or 0.0)
            if entry_price <= 0 or quantity <= 0:
                return False
            notional = entry_price * quantity
            return notional <= float(self.max_position_size_usdt)
        except Exception:
            return False
