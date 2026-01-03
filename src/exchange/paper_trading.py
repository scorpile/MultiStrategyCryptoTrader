"""Paper trading utilities for simulating orders, positions, and PnL."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.exchange.errors import OrderRejected
from src.exchange.symbol_rules import SymbolRules


@dataclass
class SimulatedOrder:
    """Represents a single simulated order/transaction."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    fee: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "symbol": self.symbol.upper(),
            "side": self.side.upper(),
            "quantity": float(self.quantity),
            "price": float(self.price),
            "trade_time": self.timestamp,
            "fee": float(self.fee),
            "metadata": self.metadata or {},
        }


class PaperTradingEngine:
    """Executes buy/sell signals against an in-memory or persisted ledger."""

    def __init__(
        self,
        *,
        state_manager: Optional[Any] = None,
        initial_cash: float = 10_000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0005,
        spread_bps: float = 5.0,
        dynamic_slippage_enabled: bool = True,
        slippage_atr_multiplier: float = 0.5,
        partial_fills_enabled: bool = True,
        partial_fill_probability: float = 0.3,
        partial_fill_min_ratio: float = 0.3,
        partial_fill_max_ratio: float = 1.0,
        min_order_quantity: float = 0.001,  # Min order size for SOL
    ) -> None:
        """Store collaborators and fee assumptions; never touches real balances."""
        self.state_manager = state_manager
        self.initial_cash = float(initial_cash)
        self.cash_balance = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)
        self.spread_bps = float(spread_bps)
        self.dynamic_slippage_enabled = bool(dynamic_slippage_enabled)
        self.slippage_atr_multiplier = float(slippage_atr_multiplier)
        self.partial_fills_enabled = bool(partial_fills_enabled)
        self.partial_fill_probability = float(partial_fill_probability)
        self.partial_fill_min_ratio = float(partial_fill_min_ratio)
        self.partial_fill_max_ratio = float(partial_fill_max_ratio)
        self.min_order_quantity = float(min_order_quantity)
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {"qty", "avg_price", "cost_basis", "stop_loss", "take_profit"}
        self.realized_pnl: float = 0.0
        self.trade_history: List[Dict[str, Any]] = []
        self.symbol_rules: Dict[str, SymbolRules] = {}
        self._last_seen_trade_count: int = 0

    def set_symbol_rules(self, symbol: str, rules: Optional[SymbolRules]) -> None:
        sym = str(symbol or "").upper()
        if not sym:
            return
        if rules is None:
            self.symbol_rules.pop(sym, None)
            return
        self.symbol_rules[sym] = rules

    def _clamp_dust_quantity(self, quantity: float, *, symbol: Optional[str] = None) -> float:
        """Round and drop tiny residuals so we don't generate sub-minimum orders.

        When exchange symbol rules are available, dust is clamped using the symbol's
        minQty; otherwise falls back to `min_order_quantity`.
        """
        qty = round(float(quantity), 6)
        min_qty = float(self.min_order_quantity)
        if symbol:
            rules = self.symbol_rules.get(str(symbol).upper())
            if rules is not None:
                try:
                    candidate = float(rules.min_qty)
                    if candidate > 0:
                        min_qty = candidate
                except Exception:
                    pass
        if min_qty > 0 and abs(qty) < min_qty:
            return 0.0
        return qty

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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def submit_order(self, order: SimulatedOrder) -> Dict[str, Any]:
        """Process an order, update positions, and optionally persist it."""
        requested_qty_raw = float(order.quantity)
        requested_qty = float(order.quantity)
        symbol = str(order.symbol or "").upper()
        rules = self.symbol_rules.get(symbol)
        min_qty = float(self.min_order_quantity)
        step_size = 0.0
        min_notional = 0.0
        if rules is not None:
            try:
                candidate = float(rules.min_qty)
                if candidate > 0:
                    min_qty = candidate
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

        requested_qty = self._apply_lot_step(requested_qty, step_size)
        if requested_qty < min_qty:
            raise OrderRejected(
                "min_qty",
                {"requested_qty": requested_qty, "requested_qty_raw": requested_qty_raw, "min_qty": min_qty, "symbol": symbol},
            )
        metadata = order.metadata or {}

        fill_qty, fill_meta = self._apply_partial_fill(requested_qty, metadata)
        exec_price, exec_meta = self._apply_execution_model(order.price, order.side, metadata)
        fill_qty = self._apply_lot_step(fill_qty, step_size)
        if fill_qty < min_qty:
            raise OrderRejected(
                "min_qty",
                {"filled_qty": fill_qty, "min_qty": min_qty, "symbol": symbol, "note": "partial_fill_or_step"},
            )
        notional = float(exec_price) * float(fill_qty)
        if min_notional > 0 and notional < min_notional:
            raise OrderRejected(
                "min_notional",
                {"notional": notional, "min_notional": min_notional, "symbol": symbol, "price": exec_price, "qty": fill_qty},
            )
        fee = exec_price * fill_qty * self.fee_rate
        order.fee = fee

        if order.side.upper() == "BUY":
            trade_pnl = self._handle_buy(fill_qty, exec_price, order.symbol, metadata=metadata)
        elif order.side.upper() == "SELL":
            trade_pnl = self._handle_sell(fill_qty, exec_price, order.symbol)
        else:
            raise ValueError(f"Unsupported side: {order.side}")

        record = order.to_dict()
        record["price"] = exec_price
        record["fee"] = fee
        record["pnl"] = trade_pnl
        record["trade_pnl"] = trade_pnl
        record["cumulative_pnl"] = self.realized_pnl
        record["quantity"] = float(fill_qty)
        side = str(record.get("side", "")).upper()
        meta_side = None
        try:
            meta_side = (metadata or {}).get("position_side")
        except Exception:
            meta_side = None
        record["position_side"] = str(meta_side or ("long" if side == "BUY" else "short")).lower()
        record["metadata"] = {
            **(metadata or {}),
            "execution": exec_meta,
            "fill": fill_meta,
            "requested_quantity": requested_qty,
            "requested_quantity_raw": requested_qty_raw,
            "filled_quantity": float(fill_qty),
            "symbol_rules": (
                {"min_qty": float(min_qty), "step_size": float(step_size), "min_notional": float(min_notional)}
                if (min_qty or step_size or min_notional)
                else {}
            ),
        }
        self.trade_history.append(record)

        if self.state_manager is not None:
            self._ensure_bootstrap_initial_cash()
            self.state_manager.save_simulated_trade(record)
            try:
                self._last_seen_trade_count = int(self.state_manager.count_total_trades())
            except Exception:
                self._last_seen_trade_count = int(self._last_seen_trade_count or 0) + 1

        return record

    def get_open_positions(self) -> List[Dict[str, float]]:
        """Return current open positions with average price and quantity."""
        positions: List[Dict[str, Any]] = []
        for symbol, data in self.positions.items():
            qty = self._clamp_dust_quantity(float(data.get("qty", 0.0) or 0.0), symbol=symbol)
            if qty != 0.0:
                data["qty"] = qty
                direction = "long" if qty > 0 else "short"
                positions.append(
                    {
                        "symbol": symbol,
                        "quantity": float(qty),
                        "avg_price": float(data.get("avg_price", 0.0)),
                        "stop_loss": data.get("stop_loss"),
                        "take_profit": data.get("take_profit"),
                        "direction": direction,
                    }
                )
        return positions

    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """Return cash balance, open positions, realized PnL, and trade count."""
        self._maybe_rehydrate_if_ledger_changed()
        trades_executed = len(self.trade_history)
        if self.state_manager is not None:
            try:
                trades_executed = int(self.state_manager.count_total_trades())
            except Exception:
                trades_executed = len(self.trade_history)
        return {
            "cash_balance": self.cash_balance,
            "open_positions": self.get_open_positions(),
            "realized_pnl": self.realized_pnl,
            "trades_executed": trades_executed,
        }

    def _maybe_rehydrate_if_ledger_changed(self) -> None:
        """Keep in-memory cash/positions consistent with the persisted ledger.

        This prevents dashboard mismatches when the process starts without having
        successfully rehydrated (or when another process writes to the ledger).
        """
        if self.state_manager is None:
            return
        try:
            total = int(self.state_manager.count_total_trades())
        except Exception:
            return
        if total <= 0:
            self._last_seen_trade_count = 0
            return
        if self._last_seen_trade_count == total:
            return
        try:
            self.rehydrate_from_ledger()
            self._last_seen_trade_count = total
        except Exception:
            # Never raise from a snapshot call; the trading loop must keep running.
            return

    def rehydrate_from_ledger(self) -> None:
        """Rebuild cash/positions/realized PnL by replaying the persisted trade ledger.

        This makes restarts fully traceable: the simulated portfolio is derived from the
        same `simulated_trades` records used for reporting.
        """
        if self.state_manager is None:
            return
        self._ensure_bootstrap_initial_cash()
        try:
            params = self.state_manager.load_parameters()
            bootstrap_cash = float(params.get("paper_initial_cash", self.initial_cash))
        except Exception:
            bootstrap_cash = float(self.initial_cash)

        self.cash_balance = float(bootstrap_cash)
        self.positions = {}
        self.realized_pnl = 0.0
        self.trade_history = []

        try:
            trades = self.state_manager.get_all_trades()
        except Exception:
            return

        for trade in trades:
            try:
                symbol = str(trade.get("symbol", "")).upper()
                side = str(trade.get("side", "")).upper()
                qty_record = float(trade.get("quantity", 0.0) or 0.0)
                price = float(trade.get("price", 0.0) or 0.0)
                fee = float(trade.get("fee", 0.0) or 0.0)
                metadata_raw = trade.get("metadata") or {}
                if isinstance(metadata_raw, dict):
                    metadata = metadata_raw
                elif isinstance(metadata_raw, str):
                    try:
                        parsed = json.loads(metadata_raw)
                        metadata = parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        metadata = {}
                else:
                    metadata = {}
            except Exception:
                continue

            if qty_record <= 0 or price <= 0 or side not in {"BUY", "SELL"}:
                continue

            try:
                if side == "BUY":
                    cost = qty_record * price + fee
                    self.cash_balance -= cost
                    position = self.positions.setdefault(
                        symbol,
                        {"qty": 0.0, "avg_price": 0.0, "cost_basis": 0.0, "stop_loss": None, "take_profit": None},
                    )
                    prev_qty = float(position.get("qty", 0.0))
                    prev_avg = float(position.get("avg_price", 0.0))
                    prev_cost = float(position.get("cost_basis", 0.0) or 0.0)
                    new_qty = prev_qty + qty_record
                    if new_qty > 0:
                        position["avg_price"] = ((prev_avg * prev_qty) + (price * qty_record)) / new_qty
                        position["qty"] = self._clamp_dust_quantity(new_qty, symbol=symbol)
                        position["cost_basis"] = prev_cost + cost
                    stop_loss = metadata.get("stop_loss") or metadata.get("stop_loss_price")
                    take_profit = metadata.get("take_profit") or metadata.get("take_profit_price")
                    if stop_loss is not None:
                        position["stop_loss"] = float(stop_loss)
                    if take_profit is not None:
                        position["take_profit"] = float(take_profit)

                if side == "SELL":
                    position = self.positions.setdefault(
                        symbol,
                        {"qty": 0.0, "avg_price": 0.0, "cost_basis": 0.0, "stop_loss": None, "take_profit": None},
                    )
                    available = float(position.get("qty", 0.0))
                    cost_basis = float(position.get("cost_basis", 0.0) or 0.0)
                    # Never allow the replay to go short: if the ledger contains SELLs with no
                    # available inventory (e.g. after a reset, race, or DB corruption),
                    # skip them so we don't mint cash out of thin air.
                    if available <= 0.0:
                        continue
                    qty = qty_record
                    ratio = 1.0
                    if qty > available and available > 0:
                        ratio = available / qty
                        qty = available
                    if qty <= 0:
                        continue
                    proceeds = qty * price - (fee * ratio)
                    self.cash_balance += proceeds
                    entry_cost = (cost_basis * (qty / available)) if available > 0 else 0.0
                    pnl_net = proceeds - entry_cost
                    self.realized_pnl += pnl_net
                    position["qty"] = self._clamp_dust_quantity(max(0.0, available - qty), symbol=symbol)
                    position["cost_basis"] = max(0.0, cost_basis - entry_cost)
                    if float(position.get("qty", 0.0)) == 0.0:
                        position["avg_price"] = 0.0
                        position["cost_basis"] = 0.0
                        position["stop_loss"] = None
                        position["take_profit"] = None
            except Exception:
                continue

        # Keep a small in-memory history for UI/debug; DB is the source of truth.
        try:
            self.trade_history = list(reversed(self.state_manager.get_recent_trades(limit=500)))
        except Exception:
            self.trade_history = []

        # Mark ledger as seen so snapshot calls don't keep rehydrating.
        try:
            self._last_seen_trade_count = int(self.state_manager.count_total_trades())
        except Exception:
            self._last_seen_trade_count = len(trades)

    def _ensure_bootstrap_initial_cash(self) -> None:
        if self.state_manager is None:
            return
        try:
            params = self.state_manager.load_parameters()
            if "paper_initial_cash" in params:
                return
            self.state_manager.save_parameters({"paper_initial_cash": float(self.initial_cash)})
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _handle_buy(self, quantity: float, price: float, symbol: str, *, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Add to (or open) a long position and update cash balance."""
        position = self.positions.setdefault(
            symbol,
            {"qty": 0.0, "avg_price": 0.0, "cost_basis": 0.0, "stop_loss": None, "take_profit": None},
        )
        qty_now = float(position.get("qty", 0.0))
        cost = quantity * price + price * quantity * self.fee_rate

        # Closing/reducing a short position
        if qty_now < 0:
            available = abs(qty_now)
            close_qty = min(quantity, available)
            if close_qty <= 0:
                return 0.0
            buy_cost = close_qty * price + price * close_qty * self.fee_rate
            if buy_cost > self.cash_balance:
                raise OrderRejected("insufficient_cash", {"need": float(buy_cost), "cash": float(self.cash_balance)})
            self.cash_balance -= buy_cost
            entry_price = float(position.get("avg_price", 0.0) or 0.0)
            cost_basis = float(position.get("cost_basis", 0.0) or 0.0)
            entry_cost = entry_price * close_qty
            pnl_net = (entry_price - price) * close_qty  # short: profit if price falls
            self.realized_pnl += pnl_net
            new_qty = qty_now + close_qty  # qty_now is negative
            position["qty"] = self._clamp_dust_quantity(new_qty, symbol=symbol)
            position["cost_basis"] = max(0.0, cost_basis - entry_cost)
            if float(position.get("qty", 0.0)) == 0.0:
                position["avg_price"] = 0.0
                position["cost_basis"] = 0.0
                position["stop_loss"] = None
                position["take_profit"] = None
            return pnl_net

        # Opening/adding to a long position
        if cost > self.cash_balance:
            raise OrderRejected(
                "insufficient_cash",
                {"need": float(cost), "cash": float(self.cash_balance), "qty": float(quantity), "price": float(price), "symbol": str(symbol).upper()},
            )
        self.cash_balance -= cost
        new_qty = qty_now + quantity
        if new_qty <= 0:
            position["qty"] = 0.0
            position["avg_price"] = 0.0
            position["cost_basis"] = 0.0
            position["stop_loss"] = None
            position["take_profit"] = None
            return 0.0
        prev_qty = float(position.get("qty", 0.0))
        prev_avg = float(position.get("avg_price", 0.0))
        prev_cost = float(position.get("cost_basis", 0.0) or 0.0)
        position["avg_price"] = ((prev_avg * prev_qty) + (price * quantity)) / new_qty
        position["qty"] = self._clamp_dust_quantity(new_qty, symbol=symbol)
        position["cost_basis"] = prev_cost + cost

        if metadata:
            stop_loss = metadata.get("stop_loss") or metadata.get("stop_loss_price")
            take_profit = metadata.get("take_profit") or metadata.get("take_profit_price")
            position["stop_loss"] = float(stop_loss) if stop_loss is not None else position.get("stop_loss")
            position["take_profit"] = float(take_profit) if take_profit is not None else position.get("take_profit")
        return 0.0

    def _handle_sell(self, quantity: float, price: float, symbol: str) -> float:
        """Reduce an existing long position and compute realized PnL."""
        position = self.positions.setdefault(
            symbol,
            {"qty": 0.0, "avg_price": 0.0, "cost_basis": 0.0, "stop_loss": None, "take_profit": None},
        )
        available_qty = float(position.get("qty", 0.0))
        # Open/increase short if flat or already short
        if available_qty <= 0:
            proceeds = quantity * price - price * quantity * self.fee_rate
            self.cash_balance += proceeds
            prev_qty = available_qty
            prev_avg = float(position.get("avg_price", 0.0) or 0.0)
            prev_cost = float(position.get("cost_basis", 0.0) or 0.0)
            new_qty = prev_qty - quantity  # more negative
            new_abs = abs(new_qty)
            prev_abs = abs(prev_qty)
            avg_price = ((prev_avg * prev_abs) + (price * quantity)) / max(new_abs, 1e-12)
            position["avg_price"] = avg_price
            position["qty"] = self._clamp_dust_quantity(new_qty, symbol=symbol)
            position["cost_basis"] = prev_cost + proceeds  # store proceeds as cost basis for symmetry
            return 0.0

        # Reduce an existing long position
        if quantity > available_qty:
            raise OrderRejected(
                "no_position",
                {"requested_qty": float(quantity), "available_qty": float(available_qty), "symbol": str(symbol).upper()},
            )
        proceeds = quantity * price - price * quantity * self.fee_rate
        self.cash_balance += proceeds
        cost_basis = float(position.get("cost_basis", 0.0) or 0.0)
        alloc_cost = (cost_basis * (quantity / available_qty)) if available_qty > 0 else 0.0
        pnl_net = proceeds - alloc_cost
        self.realized_pnl += pnl_net
        new_qty = float(position.get("qty", 0.0)) - quantity
        position["qty"] = self._clamp_dust_quantity(new_qty, symbol=symbol)
        position["cost_basis"] = max(0.0, cost_basis - alloc_cost)
        if float(position.get("qty", 0.0)) == 0.0:
            position["avg_price"] = 0.0
            position["cost_basis"] = 0.0
            position["stop_loss"] = None
            position["take_profit"] = None
        return pnl_net

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply simple symmetric slippage to the execution price."""
        adjustment = price * self.slippage
        if side.upper() == "BUY":
            return price + adjustment
        if side.upper() == "SELL":
            return price - adjustment
        return price

    def _apply_execution_model(self, mid_price: float, side: str, metadata: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Return (execution_price, execution_metadata) applying spread + dynamic slippage."""
        mid = float(mid_price)
        spread_bps = self._resolve_spread_bps(metadata)
        spread = max(0.0, float(spread_bps)) / 10_000.0

        dyn_slip = self._resolve_dynamic_slippage(mid, metadata)
        base_slip = float(self.slippage)
        total_slip = max(0.0, base_slip + dyn_slip)

        half_spread = spread / 2.0
        if side.upper() == "BUY":
            exec_price = mid * (1.0 + half_spread + total_slip)
        else:
            exec_price = mid * (1.0 - half_spread - total_slip)

        return float(exec_price), {
            "mid_price": mid,
            "spread_bps": float(spread_bps),
            "base_slippage": float(base_slip),
            "dynamic_slippage": float(dyn_slip),
            "total_slippage": float(total_slip),
        }

    def _resolve_spread_bps(self, metadata: Dict[str, Any]) -> float:
        try:
            override = (metadata or {}).get("execution", {}).get("spread_bps")
            if override is not None:
                return float(override)
        except Exception:
            pass
        return float(self.spread_bps)

    def _resolve_dynamic_slippage(self, mid: float, metadata: Dict[str, Any]) -> float:
        if not self.dynamic_slippage_enabled:
            return 0.0
        mc = {}
        try:
            mc = (metadata or {}).get("market_context") or {}
        except Exception:
            mc = {}
        try:
            atr = mc.get("atr")
            atr_f = float(atr) if atr is not None else 0.0
        except Exception:
            atr_f = 0.0
        atr_pct = (atr_f / mid) if mid > 0 and atr_f > 0 else 0.0
        extra = atr_pct * float(self.slippage_atr_multiplier)
        return max(0.0, float(extra))

    def _apply_partial_fill(self, requested_qty: float, metadata: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        if not self.partial_fills_enabled:
            return float(requested_qty), {"enabled": False, "fill_ratio": 1.0}
        try:
            exec_meta = (metadata or {}).get("execution") or {}
            if bool(exec_meta.get("force_full_fill", False)):
                return float(requested_qty), {"enabled": True, "fill_ratio": 1.0, "forced_full_fill": True}
        except Exception:
            pass
        prob = max(0.0, min(1.0, float(self.partial_fill_probability)))
        if random.random() > prob:
            return float(requested_qty), {"enabled": True, "fill_ratio": 1.0}

        lo = max(0.0, min(1.0, float(self.partial_fill_min_ratio)))
        hi = max(lo, min(1.0, float(self.partial_fill_max_ratio)))
        ratio = random.uniform(lo, hi)
        filled = float(requested_qty) * float(ratio)
        # Round and enforce bounds; remainder is cancelled in this simple simulator.
        filled = round(filled, 6)
        remainder = max(0.0, float(requested_qty) - filled)
        return max(0.0, filled), {"enabled": True, "fill_ratio": round(ratio, 4), "remainder_quantity": round(remainder, 6)}
