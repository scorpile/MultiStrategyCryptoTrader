"""Live trading engine abstraction (Binance Spot only).

This module provides a small adapter so the scheduler can route orders to either
paper trading or live execution while keeping the same high-level flow.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.exchange.binance_spot_executor import BinanceSpotExecutor
from src.exchange.errors import OrderRejected
from src.exchange.paper_trading import SimulatedOrder
from src.exchange.symbol_rules import SymbolRules


@dataclass(frozen=True)
class LiveLimits:
    allowed_symbol: str = "SOLUSDT"
    max_notional_usdt_per_trade: float = 10.0
    max_daily_loss_usdt: float = 10.0
    max_orders_per_day: int = 20
    min_order_quantity: float = 0.001
    require_env_var: str = "BINANCE_LIVE_ARMED"
    require_env_value: str = "1"


class LiveTradingEngine:
    """Executes orders on Binance Spot, with local SQLite ledger for PnL/SL/TP metadata."""

    def __init__(
        self,
        *,
        state_manager: Any,
        executor: BinanceSpotExecutor,
        limits: LiveLimits,
    ) -> None:
        self.state_manager = state_manager
        self.executor = executor
        self.limits = limits
        self.min_order_quantity = float(limits.min_order_quantity)
        self._account_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self.symbol_rules: Dict[str, SymbolRules] = {}

        try:
            self.state_manager.ensure_execution_tables()
        except Exception:
            pass

    def set_symbol_rules(self, symbol: str, rules: Optional[SymbolRules]) -> None:
        sym = str(symbol or "").upper()
        if not sym:
            return
        if rules is None:
            self.symbol_rules.pop(sym, None)
            return
        self.symbol_rules[sym] = rules

    @staticmethod
    def _apply_lot_step(quantity: float, step_size: float) -> float:
        qty = float(quantity or 0.0)
        step = float(step_size or 0.0)
        if qty <= 0 or step <= 0:
            return qty
        try:
            steps = math.floor((qty + 1e-12) / step)
            return float(steps * step)
        except Exception:
            return qty

    def _is_armed(self) -> bool:
        key = str(self.limits.require_env_var or "").strip()
        expected = str(self.limits.require_env_value or "").strip()
        if not key:
            return False
        return str(os.getenv(key, "")).strip() == expected

    def get_portfolio_snapshot(self, *, current_price: float = 0.0) -> Dict[str, Any]:
        acct = self._get_account_cached()
        usdt_free = float(self._find_free_balance(acct, "USDT"))
        sol_free = float(self._find_free_balance(acct, "SOL"))
        pos = self.state_manager.get_execution_position(symbol=self.limits.allowed_symbol, mode="live") or {}
        qty = float(pos.get("quantity", 0.0) or 0.0)
        avg_price = float(pos.get("avg_price", 0.0) or 0.0)
        stop_loss = pos.get("stop_loss")
        take_profit = pos.get("take_profit")

        open_positions = []
        if qty > 0 and abs(qty) >= float(self.min_order_quantity):
            open_positions.append(
                {
                    "symbol": self.limits.allowed_symbol,
                    "quantity": qty,
                    "avg_price": avg_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            )

        try:
            realized_pnl = float(self.state_manager.get_execution_realized_pnl_total(mode="live") or 0.0)
        except Exception:
            realized_pnl = 0.0

        total_value = float(usdt_free) + (float(sol_free) * float(current_price or 0.0))
        return {
            "cash_balance": usdt_free,
            "realized_pnl": realized_pnl,
            "open_positions": open_positions,
            "balances": {"USDT": usdt_free, "SOL": sol_free},
            "total_value": total_value,
        }

    def get_open_positions(self) -> list[Dict[str, Any]]:
        pos = self.state_manager.get_execution_position(symbol=self.limits.allowed_symbol, mode="live") or {}
        qty = float(pos.get("quantity", 0.0) or 0.0)
        if qty <= 0 or abs(qty) < float(self.min_order_quantity):
            return []
        return [
            {
                "symbol": self.limits.allowed_symbol,
                "quantity": qty,
                "avg_price": float(pos.get("avg_price", 0.0) or 0.0),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
            }
        ]

    def submit_order(self, order: SimulatedOrder) -> Dict[str, Any]:
        if not self._is_armed():
            raise OrderRejected(
                "not_armed",
                {"env_var": self.limits.require_env_var, "env_value": self.limits.require_env_value},
            )

        symbol = str(order.symbol or "").upper()
        if symbol != str(self.limits.allowed_symbol).upper():
            raise OrderRejected("symbol_not_allowed", {"symbol": symbol, "allowed": self.limits.allowed_symbol})

        side = str(order.side or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Order side must be BUY or SELL.")

        rules = self.symbol_rules.get(symbol)
        min_qty = float(self.min_order_quantity)
        step_size = 0.0
        min_notional = 0.0
        if rules is not None:
            try:
                min_qty = max(min_qty, float(rules.min_qty))
            except Exception:
                pass
            try:
                step_size = float(rules.step_size)
            except Exception:
                step_size = 0.0
            try:
                min_notional = float(rules.min_notional)
            except Exception:
                min_notional = 0.0

        today = datetime.utcnow().date().isoformat()
        if side == "BUY":
            count_today = int(self.state_manager.count_execution_orders_for_day(today, mode="live") or 0)
            if self.limits.max_orders_per_day > 0 and count_today >= int(self.limits.max_orders_per_day):
                raise OrderRejected("max_orders_per_day", {"max": int(self.limits.max_orders_per_day), "count": count_today})
            pnl_today = float(self.state_manager.get_execution_realized_pnl_for_day(today, mode="live") or 0.0)
            if self.limits.max_daily_loss_usdt > 0 and pnl_today <= -float(self.limits.max_daily_loss_usdt):
                raise OrderRejected("max_daily_loss", {"pnl_today": pnl_today, "max_daily_loss": float(self.limits.max_daily_loss_usdt)})

        requested_qty = float(order.quantity or 0.0)
        if requested_qty <= 0:
            raise ValueError("Order quantity must be > 0.")

        if side == "BUY":
            notional = float(order.price or 0.0) * requested_qty
            notional = min(float(notional), float(self.limits.max_notional_usdt_per_trade))
            acct = self._get_account_cached(force=True)
            usdt_free = float(self._find_free_balance(acct, "USDT"))
            notional = min(notional, usdt_free)
            if min_notional > 0 and notional < min_notional:
                raise OrderRejected("min_notional", {"notional": notional, "min_notional": min_notional, "symbol": symbol})
            if notional <= 0:
                raise OrderRejected("insufficient_cash", {"cash": usdt_free, "need": float(min_notional or 0.0)})
            response = self.executor.create_market_order(
                symbol=symbol,
                side="BUY",
                quote_order_qty=notional,
                client_order_id=self._make_client_id(order),
            )
        else:
            acct = self._get_account_cached(force=True)
            sol_free = float(self._find_free_balance(acct, "SOL"))
            sell_qty = min(requested_qty, sol_free)
            sell_qty = self._apply_lot_step(sell_qty, step_size)
            sell_qty = self._clamp_qty(sell_qty)
            if min_qty > 0 and sell_qty < min_qty:
                raise OrderRejected("no_position", {"available_qty": sol_free, "requested_qty": requested_qty, "min_qty": min_qty})
            if min_notional > 0 and (sell_qty * float(order.price or 0.0)) < min_notional:
                raise OrderRejected("min_notional", {"notional": sell_qty * float(order.price or 0.0), "min_notional": min_notional, "symbol": symbol})
            response = self.executor.create_market_order(
                symbol=symbol,
                side="SELL",
                quantity=sell_qty,
                client_order_id=self._make_client_id(order),
            )

        trade_record = self._record_execution(order=order, response=response)
        self._account_cache = (0.0, None)
        return trade_record

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_free_balance(account_payload: Dict[str, Any], asset: str) -> float:
        target = str(asset or "").upper()
        for bal in account_payload.get("balances", []) or []:
            if str(bal.get("asset", "")).upper() == target:
                try:
                    return float(bal.get("free", 0.0) or 0.0)
                except Exception:
                    return 0.0
        return 0.0

    def _get_account_cached(self, *, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        cached_at, payload = self._account_cache
        if not force and payload is not None and (now - float(cached_at)) < 5.0:
            return payload
        payload = self.executor.get_account()
        self._account_cache = (now, payload)
        return payload

    def _clamp_qty(self, qty: float) -> float:
        q = round(float(qty), 6)
        if abs(q) < float(self.min_order_quantity):
            return 0.0
        return q

    @staticmethod
    def _make_client_id(order: SimulatedOrder) -> str:
        ts = str(order.timestamp or "").replace(":", "").replace("-", "").replace(".", "")
        return f"codex_{ts}"[:36]

    def _record_execution(self, *, order: SimulatedOrder, response: Dict[str, Any]) -> Dict[str, Any]:
        fills = response.get("fills") or []
        total_qty = 0.0
        total_notional = 0.0
        fee_usdt = 0.0

        for fill in fills:
            try:
                qty = float(fill.get("qty", 0.0) or 0.0)
                price = float(fill.get("price", 0.0) or 0.0)
            except Exception:
                continue
            if qty <= 0 or price <= 0:
                continue
            total_qty += qty
            total_notional += qty * price
            try:
                commission = float(fill.get("commission", 0.0) or 0.0)
                asset = str(fill.get("commissionAsset", "")).upper()
                if commission > 0:
                    if asset == "USDT":
                        fee_usdt += commission
                    elif asset == "SOL":
                        fee_usdt += commission * price
            except Exception:
                pass

        avg_price = (total_notional / total_qty) if total_qty > 0 else float(order.price or 0.0)
        side = str(order.side or "").upper()
        symbol = str(order.symbol or "").upper()
        meta = order.metadata or {}

        pnl = 0.0
        if side == "BUY":
            self.state_manager.upsert_execution_position(
                symbol=symbol,
                mode="live",
                quantity_delta=total_qty,
                execution_price=avg_price,
                cost_basis_delta_usdt=float(total_notional) + float(fee_usdt),
                stop_loss=meta.get("stop_loss"),
                take_profit=meta.get("take_profit"),
            )
        else:
            pnl = float(
                self.state_manager.reduce_execution_position_and_realize_pnl(
                    symbol=symbol,
                    mode="live",
                    sell_quantity=total_qty,
                    sell_price=avg_price,
                    sell_fee_usdt=float(fee_usdt),
                )
                or 0.0
            )

        record = {
            "trade_time": order.timestamp,
            "symbol": symbol,
            "side": side,
            "quantity": float(total_qty),
            "price": float(avg_price),
            "fee": float(fee_usdt),
            "pnl": float(pnl),
            "order_id": response.get("orderId"),
            "client_order_id": response.get("clientOrderId"),
            "metadata": {
                **(meta or {}),
                "raw": {
                    "orderId": response.get("orderId"),
                    "transactTime": response.get("transactTime"),
                    "status": response.get("status"),
                    "executedQty": response.get("executedQty"),
                    "cummulativeQuoteQty": response.get("cummulativeQuoteQty"),
                },
            },
        }

        self.state_manager.save_execution_trade(record, mode="live")
        return record
