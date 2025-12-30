"""Backtesting engine capable of running any registered strategy on OHLCV data."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.ml import feature_engineering
from src.exchange.symbol_rules import SymbolRules
from src.strategy.registry import STRATEGY_REGISTRY
from src.strategy.strategy_base import BaseStrategy
from src.strategy import technicals


@dataclass
class BacktestMetrics:
    balance: float
    pnl: float
    win_rate: float
    max_drawdown: float
    sharpe: float
    trades: int


@dataclass
class BacktestResult:
    strategy: str
    metrics: BacktestMetrics
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "metrics": self.metrics.__dict__,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }


class BacktestingEngine:
    """Runs historical simulations and produces metrics + exports."""

    def __init__(
        self,
        *,
        initial_cash: float = 10_000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0005,
        strategy_configs: Optional[Dict[str, Any]] = None,
        risk_manager: Optional[Any] = None,
        symbol_rules: Optional[SymbolRules] = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.strategy_configs = strategy_configs or {}
        self.risk_manager = risk_manager
        self.symbol_rules = symbol_rules
        self._indicator_cache: Dict[tuple, Dict[str, Any]] = {}
        self._indicator_cache_max = 32

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

    def run(
        self,
        strategy_name: str,
        candles: Sequence[Dict[str, Any]],
        *,
        strategy_config: Optional[Dict[str, Any]] = None,
        export_dir: Optional[Path | str] = None,
        ml_context_provider: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        collect_trades: bool = True,
        collect_equity_curve: bool = True,
    ) -> BacktestResult:
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{strategy_name}'")
        if export_dir is not None:
            collect_trades = True
            collect_equity_curve = True
        config = strategy_config or self.strategy_configs.get(strategy_name)
        strategy = STRATEGY_REGISTRY[strategy_name](
            config=config,
            risk_manager=self.risk_manager,
        )
        cash = self.initial_cash
        position_qty = 0.0
        entry_price = 0.0
        entry_cost = 0.0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []
        total_pnl = 0.0
        realized_trades = 0
        wins = 0
        peak_equity = float(self.initial_cash)
        max_drawdown = 0.0
        prev_equity: Optional[float] = None
        ret_count = 0
        ret_mean = 0.0
        ret_m2 = 0.0

        candles_list = candles if isinstance(candles, list) else list(candles)

        if not candles_list:
            metrics = BacktestMetrics(
                balance=float(self.initial_cash),
                pnl=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe=0.0,
                trades=0,
            )
            result = BacktestResult(strategy=strategy_name, metrics=metrics, equity_curve=equity_curve, trades=trades)
            if export_dir:
                self._export_results(result, Path(export_dir))
            return result

        def _int_from_cfg(key: str, default: int) -> int:
            try:
                return int(strategy.config.get(key, default) or default)
            except Exception:
                return int(default)

        def _float_from_cfg(key: str, default: float) -> float:
            try:
                return float(strategy.config.get(key, default) or default)
            except Exception:
                return float(default)

        rsi_period = max(1, _int_from_cfg("rsi_period", 14))
        ema_fast_period = max(1, _int_from_cfg("ema_fast_period", 20))
        ema_slow_period = max(1, _int_from_cfg("ema_slow_period", 50))
        volume_sma_period = max(1, _int_from_cfg("volume_sma_period", 20))
        bollinger_period = max(1, _int_from_cfg("bollinger_period", 20))
        bollinger_std = max(0.1, _float_from_cfg("bollinger_stddev", 2.0))
        atr_period = max(1, _int_from_cfg("atr_period", 14))

        cache_key = (
            id(candles_list),
            len(candles_list),
            rsi_period,
            ema_fast_period,
            ema_slow_period,
            volume_sma_period,
            bollinger_period,
            round(float(bollinger_std), 6),
            atr_period,
        )
        cached = self._indicator_cache.get(cache_key)
        if cached is None:
            # Pre-compute indicator series once per candle sequence/config (avoid O(n^2) rebuilds).
            closes_all = [row["close"] for row in candles_list]
            highs_all = [row["high"] for row in candles_list]
            lows_all = [row["low"] for row in candles_list]
            volumes_all = [row.get("volume", 0.0) for row in candles_list]
            rsi_all = technicals.compute_rsi(closes_all, period=rsi_period)
            macd_all = technicals.compute_macd(closes_all)
            boll_all = technicals.compute_bollinger_bands(closes_all, period=bollinger_period, num_std_dev=bollinger_std)
            atr_all = technicals.compute_atr(highs_all, lows_all, closes_all, period=atr_period)
            ema_fast_all = _ema_series(closes_all, ema_fast_period)
            ema_slow_all = _ema_series(closes_all, ema_slow_period)
            volume_sma_all = technicals.compute_volume_sma(volumes_all, period=volume_sma_period)

            cached = {
                "closes": closes_all,
                "highs": highs_all,
                "lows": lows_all,
                "volumes": volumes_all,
                "rsi": rsi_all,
                "macd": macd_all,
                "bollinger": boll_all,
                "atr": atr_all,
                "ema_fast": ema_fast_all,
                "ema_slow": ema_slow_all,
                "volume_sma": volume_sma_all,
            }
            if len(self._indicator_cache) >= int(self._indicator_cache_max):
                self._indicator_cache.clear()
            self._indicator_cache[cache_key] = cached
        else:
            closes_all = cached["closes"]
            highs_all = cached["highs"]
            lows_all = cached["lows"]
            volumes_all = cached["volumes"]
            rsi_all = cached["rsi"]
            macd_all = cached["macd"]
            boll_all = cached["bollinger"]
            atr_all = cached["atr"]
            ema_fast_all = cached["ema_fast"]
            ema_slow_all = cached["ema_slow"]
            volume_sma_all = cached["volume_sma"]

        ohlcv_prefix: List[Dict[str, Any]] = []
        closes_prefix: List[float] = []
        highs_prefix: List[float] = []
        lows_prefix: List[float] = []
        volumes_prefix: List[float] = []
        volume_sma_prefix: List[float] = []
        rsi_prefix: List[float] = []
        atr_prefix: List[float] = []
        ema_fast_prefix: List[float] = []
        ema_slow_prefix: List[float] = []
        macd_prefix: Dict[str, List[float]] = {"macd": [], "signal": [], "histogram": []}
        boll_prefix: Dict[str, List[float]] = {"middle": [], "upper": [], "lower": []}

        for idx, current_candle in enumerate(candles_list):
            ohlcv_prefix.append(current_candle)
            closes_prefix.append(float(closes_all[idx]))
            highs_prefix.append(float(highs_all[idx]))
            lows_prefix.append(float(lows_all[idx]))
            volumes_prefix.append(float(volumes_all[idx]))
            volume_sma_prefix.append(float(volume_sma_all[idx]) if idx < len(volume_sma_all) else 0.0)
            rsi_prefix.append(float(rsi_all[idx]) if idx < len(rsi_all) else 0.0)
            atr_prefix.append(float(atr_all[idx]) if idx < len(atr_all) else 0.0)
            ema_fast_prefix.append(float(ema_fast_all[idx]) if idx < len(ema_fast_all) else 0.0)
            ema_slow_prefix.append(float(ema_slow_all[idx]) if idx < len(ema_slow_all) else 0.0)
            macd_prefix["macd"].append(float(macd_all.get("macd", [0.0])[idx]) if macd_all.get("macd") else 0.0)
            macd_prefix["signal"].append(float(macd_all.get("signal", [0.0])[idx]) if macd_all.get("signal") else 0.0)
            macd_prefix["histogram"].append(float(macd_all.get("histogram", [0.0])[idx]) if macd_all.get("histogram") else 0.0)
            boll_prefix["middle"].append(float(boll_all.get("middle", [0.0])[idx]) if boll_all.get("middle") else 0.0)
            boll_prefix["upper"].append(float(boll_all.get("upper", [0.0])[idx]) if boll_all.get("upper") else 0.0)
            boll_prefix["lower"].append(float(boll_all.get("lower", [0.0])[idx]) if boll_all.get("lower") else 0.0)

            indicators = {
                "ohlcv": ohlcv_prefix,
                "closes": closes_prefix,
                "highs": highs_prefix,
                "lows": lows_prefix,
                "volumes": volumes_prefix,
                "volume_sma": volume_sma_prefix,
                "rsi": rsi_prefix,
                "macd": macd_prefix,
                "bollinger": boll_prefix,
                "atr": atr_prefix,
                "ema_fast": ema_fast_prefix,
                "ema_slow": ema_slow_prefix,
            }

            if len(closes_prefix) < 30:
                continue

            price = float(closes_prefix[-1])

            if position_qty > 0:
                low = float(current_candle.get("low", price))
                high = float(current_candle.get("high", price))
                triggered: Optional[str] = None
                exit_level: Optional[float] = None
                if stop_loss_price and low <= stop_loss_price:
                    triggered = "stop_loss"
                    exit_level = float(stop_loss_price)
                elif take_profit_price and high >= take_profit_price:
                    triggered = "take_profit"
                    exit_level = float(take_profit_price)
                if triggered and exit_level is not None:
                    exec_price = exit_level * (1 - self.slippage)
                    fee = exec_price * position_qty * self.fee_rate
                    proceeds = position_qty * exec_price - fee
                    pnl = proceeds - entry_cost
                    cash += proceeds
                    total_pnl += float(pnl)
                    realized_trades += 1
                    if float(pnl) > 0.0:
                        wins += 1
                    if collect_trades:
                        trades.append(
                            {
                                "time": current_candle["open_time"],
                                "action": "sell",
                                "reason": triggered,
                                "price": exec_price,
                                "quantity": position_qty,
                                "fee": fee,
                                "pnl": pnl,
                            }
                        )
                    position_qty = 0.0
                    entry_price = 0.0
                    entry_cost = 0.0
                    stop_loss_price = 0.0
                    take_profit_price = 0.0
            ml_ctx = ml_context_provider(indicators) if ml_context_provider else {}
            signal = strategy.generate_signal(indicators, context={"ml": ml_ctx, "position_qty": position_qty, "capital": cash})
            decision = signal.get("decision", "hold")
            quantity = float(signal.get("suggested_size", 0.0) or 0.0)
            signal_metadata = signal.get("metadata") or {}

            if decision == "buy" and position_qty <= 0:
                equity = cash + position_qty * price
                stop_loss_price, take_profit_price = self._resolve_stop_take_prices(
                    strategy=strategy,
                    indicators=indicators,
                    entry_price=price,
                    metadata=signal_metadata,
                )

                qty_units = quantity
                if self.risk_manager is not None and stop_loss_price > 0:
                    risk_pct = float(strategy.config.get("risk_pct_per_trade", signal_metadata.get("risk_pct_per_trade", 0.01)))
                    sized = self.risk_manager.size_position_for_risk(
                        equity_usdt=equity,
                        entry_price=price,
                        stop_loss_price=stop_loss_price,
                        risk_pct_per_trade=risk_pct,
                        max_quantity=qty_units if qty_units > 0 else None,
                    )
                    if sized.quantity > 0:
                        qty_units = min(qty_units, sized.quantity) if qty_units > 0 else sized.quantity

                max_affordable = cash / (price * (1 + self.fee_rate)) if price > 0 else 0.0
                qty_units = min(qty_units, max_affordable)
                qty_units = round(max(0.0, qty_units), 6)

                # Enforce exchange-like filters in backtest (LOT_SIZE + MIN_NOTIONAL).
                if self.symbol_rules is not None:
                    min_qty = float(self.symbol_rules.min_qty or 0.0)
                    step_size = float(self.symbol_rules.step_size or 0.0)
                    min_notional = float(self.symbol_rules.min_notional or 0.0)
                    qty_units = self._apply_lot_step(qty_units, step_size)
                    if min_qty > 0 and qty_units < min_qty:
                        continue

                if qty_units <= 0:
                    continue

                exec_price = price * (1 + self.slippage)
                fee = exec_price * qty_units * self.fee_rate
                cost = qty_units * exec_price + fee

                if self.symbol_rules is not None:
                    min_notional = float(self.symbol_rules.min_notional or 0.0)
                    if min_notional > 0 and (qty_units * exec_price) < min_notional:
                        continue

                cash -= cost
                position_qty = qty_units
                entry_price = exec_price
                entry_cost = cost
                if collect_trades:
                    trades.append(
                        {
                            "time": current_candle["open_time"],
                            "action": "buy",
                            "price": exec_price,
                            "quantity": qty_units,
                            "fee": fee,
                            "stop_loss": stop_loss_price,
                            "take_profit": take_profit_price,
                        }
                    )
            elif decision == "sell" and position_qty > 0:
                exec_price = price * (1 - self.slippage)
                fee = exec_price * position_qty * self.fee_rate
                proceeds = position_qty * exec_price - fee
                pnl = proceeds - entry_cost
                cash += proceeds
                total_pnl += float(pnl)
                realized_trades += 1
                if float(pnl) > 0.0:
                    wins += 1
                if collect_trades:
                    trades.append(
                        {
                            "time": current_candle["open_time"],
                            "action": "sell",
                            "price": exec_price,
                            "quantity": position_qty,
                            "fee": fee,
                            "pnl": pnl,
                        }
                    )
                position_qty = 0.0
                entry_price = 0.0
                entry_cost = 0.0
                stop_loss_price = 0.0
                take_profit_price = 0.0

            equity = cash + position_qty * price
            if equity > peak_equity:
                peak_equity = float(equity)
            drawdown = float(peak_equity) - float(equity)
            if drawdown > max_drawdown:
                max_drawdown = float(drawdown)
            if prev_equity is not None and float(prev_equity) != 0.0:
                ret = (float(equity) - float(prev_equity)) / float(prev_equity)
                ret_count += 1
                delta = ret - ret_mean
                ret_mean += delta / ret_count
                delta2 = ret - ret_mean
                ret_m2 += delta * delta2
            prev_equity = float(equity)
            if collect_equity_curve:
                equity_curve.append({"time": current_candle["open_time"], "equity": equity})

        balance = cash + position_qty * (candles_list[-1]["close"] if candles_list else 0.0)
        win_rate = float(wins) / float(realized_trades) if realized_trades else 0.0
        variance = (ret_m2 / ret_count) if ret_count else 0.0
        std_dev = variance**0.5 if variance > 0 else 0.0
        sharpe = (ret_mean / std_dev) if std_dev else 0.0
        metrics = BacktestMetrics(
            balance=float(balance),
            pnl=float(total_pnl),
            win_rate=float(win_rate),
            max_drawdown=float(max_drawdown),
            sharpe=float(sharpe),
            trades=int(realized_trades),
        )
        result = BacktestResult(strategy=strategy_name, metrics=metrics, equity_curve=equity_curve, trades=trades)
        if export_dir:
            self._export_results(result, Path(export_dir))
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_indicator_payload(self, candles: Sequence[Dict[str, Any]], *, compute_features: bool = False) -> Dict[str, Any]:
        closes = [row["close"] for row in candles]
        highs = [row["high"] for row in candles]
        lows = [row["low"] for row in candles]
        volumes = [row.get("volume", 0.0) for row in candles]
        payload = {
            "ohlcv": list(candles),
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "rsi": technicals.compute_rsi(closes),
            "macd": technicals.compute_macd(closes),
            "bollinger": technicals.compute_bollinger_bands(closes),
            "atr": technicals.compute_atr(highs, lows, closes),
        }
        if compute_features:
            payload["features_df"] = feature_engineering.generate_features(candles)
        payload["ema_fast"] = _ema_series(closes, 20)
        payload["ema_slow"] = _ema_series(closes, 50)
        return payload

    def _build_metrics(
        self,
        balance: float,
        pnl_values: Sequence[float],
        trades: Sequence[Dict[str, Any]],
        equity_curve: Sequence[Dict[str, Any]],
    ) -> BacktestMetrics:
        realized_trades = [t for t in trades if t.get("action") == "sell"]
        total_trades = len(realized_trades)
        wins = sum(1 for trade in realized_trades if float(trade.get("pnl", 0.0) or 0.0) > 0.0)
        win_rate = wins / total_trades if total_trades else 0.0
        max_drawdown = self._calculate_drawdown(equity_curve)
        sharpe = self._calculate_sharpe(equity_curve)
        return BacktestMetrics(
            balance=balance,
            pnl=sum(pnl_values),
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            trades=total_trades,
        )

    @staticmethod
    def _resolve_stop_take_prices(
        *,
        strategy: BaseStrategy,
        indicators: Dict[str, Any],
        entry_price: float,
        metadata: Dict[str, Any],
    ) -> tuple[float, float]:
        if entry_price <= 0:
            return (0.0, 0.0)
        stop_loss = metadata.get("stop_loss") or metadata.get("stop_loss_price")
        take_profit = metadata.get("take_profit") or metadata.get("take_profit_price")
        if stop_loss is not None and take_profit is not None:
            return (float(stop_loss), float(take_profit))

        atr_series = indicators.get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else entry_price * 0.008
        stop_atr_mult = strategy.config.get("stop_atr_multiplier")
        profit_target_mult = strategy.config.get("profit_target_multiplier")
        if stop_atr_mult is not None and profit_target_mult is not None and atr_val > 0:
            stop_dist = atr_val * float(stop_atr_mult)
            stop_loss_price = max(0.0, entry_price - stop_dist)
            take_profit_price = max(0.0, entry_price + stop_dist * float(profit_target_mult))
            return (stop_loss_price, take_profit_price)

        stop_loss_pct = float(strategy.config.get("stop_loss_pct", 0.05))
        take_profit_pct = float(strategy.config.get("take_profit_pct", 0.10))
        stop_loss_price = max(0.0, entry_price * (1.0 - max(0.0, stop_loss_pct)))
        take_profit_price = max(0.0, entry_price * (1.0 + max(0.0, take_profit_pct)))
        return (stop_loss_price, take_profit_price)

    def _export_results(self, result: BacktestResult, export_dir: Path) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        json_path = export_dir / f"{result.strategy}_backtest.json"
        csv_path = export_dir / f"{result.strategy}_trades.csv"
        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=result.trades[0].keys() if result.trades else ["time", "action", "price", "quantity", "fee", "pnl"])
            writer.writeheader()
            for trade in result.trades:
                writer.writerow(trade)

    @staticmethod
    def _calculate_drawdown(equity_curve: Sequence[Dict[str, Any]]) -> float:
        peak = 0.0
        max_drawdown = 0.0
        for point in equity_curve:
            equity = point["equity"]
            peak = max(peak, equity)
            drawdown = peak - equity
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    @staticmethod
    def _calculate_sharpe(equity_curve: Sequence[Dict[str, Any]]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        returns = []
        for prev, curr in zip(equity_curve[:-1], equity_curve[1:]):
            if prev["equity"] == 0:
                continue
            returns.append((curr["equity"] - prev["equity"]) / prev["equity"])
        if not returns:
            return 0.0
        avg_return = mean(returns)
        variance = mean([(r - avg_return) ** 2 for r in returns]) or 1e-9
        std_dev = variance ** 0.5
        return avg_return / std_dev if std_dev else 0.0


def _ema_series(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    multiplier = 2 / (period + 1)
    ema_values = [float(values[0])]
    for price in values[1:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values
