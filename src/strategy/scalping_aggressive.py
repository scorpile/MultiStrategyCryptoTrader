"""Ultra-aggressive scalping strategy with ML integration and auto-adjustment."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence

from .strategy_base import BaseStrategy


class ScalpingAggressiveStrategy(BaseStrategy):
    """Extreme scalping strategy: rapid entries/exits, high risk, ML-boosted signals, auto-balancing."""

    description = "Scalping with tight RSI/StochRSI, EMA micro-crosses, momentum spikes, ML-enhanced confidence, auto-position sizing."

    def name(self) -> str:
        return "scalping_aggressive"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "rsi_buy": 40,  # More aggressive for frequent trades
            "rsi_sell": 60,
            "stoch_rsi_buy": 0.2,
            "stoch_rsi_sell": 0.8,
            "ema_fast_period": 5,  # Faster EMAs
            "ema_slow_period": 13,
            "momentum_period": 3,  # Shorter momentum
            "momentum_threshold": 0.05,  # Even lower for more frequent signals
            "ml_confidence_boost": 0.2,  # Boost confidence with ML
            "auto_adjust_factor": 0.05,  # Auto-adjust thresholds based on performance
            "max_scalp_size": 0.8,  # Higher max position size (fraction of capital)
            # Risk-based sizing parameters
            "risk_pct_per_trade": 0.005,  # 0.5% of capital risk per trade
            "stop_atr_multiplier": 3.0,  # stop distance = ATR * multiplier (wider to reduce whipsaws)
            "profit_target_multiplier": 4.0,  # take-profit distance = stop_dist * multiplier (covers fees/slippage)
            "min_order_quantity": 0.001,
            # Exploration (intentionally aggressive early on)
            "exploration_rsi_relax": 12.0,
            "exploration_stoch_relax": 0.25,
            "exploration_ema_slack": 0.05,
            "exploration_momentum_slack": 0.5,
            "exploration_confidence_floor": 0.65,
            "exploration_risk_multiplier": 2.5,
            "exploration_random_entry_prob": 0.15,
            "exploration_random_exit_prob": 0.10,
        }

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        min_period = max(self.config["ema_slow_period"], self.config["momentum_period"]) + 2
        if len(closes) < min_period:
            return self._build_signal(
                decision="hold",
                price=closes[-1] if closes else 0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Insufficient data for scalping signals."],
            )

        price = float(closes[-1])
        rsi_series = indicators.get("rsi") or []
        stoch_rsi = _stoch_rsi(rsi_series)
        ema_fast_series = indicators.get("ema_fast")
        ema_slow_series = indicators.get("ema_slow")
        if isinstance(ema_fast_series, list) and ema_fast_series:
            ema_fast = float(ema_fast_series[-1])
        else:
            ema_fast = _ema(closes, self.config["ema_fast_period"])
        if isinstance(ema_slow_series, list) and ema_slow_series:
            ema_slow = float(ema_slow_series[-1])
        else:
            ema_slow = _ema(closes, self.config["ema_slow_period"])
        atr_series = indicators.get("atr") or []
        atr_val = self._latest(atr_series, default=price * 0.01)  # Higher default ATR for scalping
        momentum_value = _momentum(closes, self.config["momentum_period"])

        # Scalping-specific indicators
        doji_series = indicators.get("doji") or []
        engulfing_series = indicators.get("engulfing") or []
        volume_series = indicators.get("volumes") or []
        volume_sma = indicators.get("volume_sma") or []

        latest_rsi = self._latest(rsi_series, default=50.0)
        ema_cross = ema_fast - ema_slow
        latest_doji = self._latest(doji_series, default=False)
        latest_engulfing = self._latest(engulfing_series, default=0)
        latest_volume = self._latest(volume_series, default=0.0)
        latest_volume_sma = self._latest(volume_sma, default=0.0)
        volume_spike = latest_volume > latest_volume_sma * 1.5 if latest_volume_sma > 0 else False

        # Context (passed from scheduler via StrategyManager)
        ctx = context or {}
        exploration = bool(ctx.get("exploration", False))
        position_qty = float(ctx.get("position_qty", 0.0) or 0.0)

        # ML context integration (optional)
        ml_ctx = ctx.get("ml", {}) if ctx else {}
        ml_confidence = ml_ctx.get("confidence", 0.0)
        probability_up = ml_ctx.get("probability_up", 0.5)
        model_name = str(ml_ctx.get("model_name", "") or "")
        ml_available = bool(ml_ctx) and model_name not in {"", "dummy_model"}

        # Auto-adjust thresholds based on recent performance (simplified)
        # In a real implementation, track win/loss and adjust
        adjusted_rsi_buy = self.config["rsi_buy"] - (ml_confidence * self.config["auto_adjust_factor"] * 10)
        adjusted_rsi_sell = self.config["rsi_sell"] + (ml_confidence * self.config["auto_adjust_factor"] * 10)

        # Exploration mode: intentionally permissive to generate trades early.
        rsi_relax = float(self.config.get("exploration_rsi_relax", 12.0)) if exploration else 0.0
        stoch_relax = float(self.config.get("exploration_stoch_relax", 0.25)) if exploration else 0.0
        ema_slack = float(self.config.get("exploration_ema_slack", 0.05)) if exploration else 0.0
        mom_slack = float(self.config.get("exploration_momentum_slack", 0.5)) if exploration else 0.0

        rsi_buy_th = adjusted_rsi_buy + rsi_relax
        rsi_sell_th = adjusted_rsi_sell - rsi_relax
        stoch_buy_th = min(1.0, float(self.config["stoch_rsi_buy"]) + stoch_relax)
        stoch_sell_th = max(0.0, float(self.config["stoch_rsi_sell"]) - stoch_relax)

        ema_buy_ok = ema_cross > (-ema_slack if exploration else 0.0)
        ema_sell_ok = ema_cross < (ema_slack if exploration else 0.0)

        mom_th = float(self.config["momentum_threshold"])
        mom_buy_ok = momentum_value >= (-(mom_th * mom_slack) if exploration else mom_th)
        mom_sell_ok = momentum_value <= ((mom_th * mom_slack) if exploration else -mom_th)

        # ML gating is only applied when ML is available and we're not exploring.
        ml_buy_ok = True
        ml_sell_ok = True
        if ml_available and not exploration:
            ml_buy_ok = probability_up > 0.55
            ml_sell_ok = probability_up < 0.45

        # Trigger gating is ignored while exploring.
        trigger_buy_ok = True if exploration else (latest_engulfing == 1 or volume_spike)
        trigger_sell_ok = True if exploration else (latest_engulfing == -1 or volume_spike)

        buy_condition = (
            latest_rsi <= rsi_buy_th
            and stoch_rsi <= stoch_buy_th
            and ema_buy_ok
            and mom_buy_ok
            and ml_buy_ok
            and trigger_buy_ok
        )
        sell_condition = (
            latest_rsi >= rsi_sell_th
            and stoch_rsi >= stoch_sell_th
            and ema_sell_ok
            and mom_sell_ok
            and ml_sell_ok
            and trigger_sell_ok
        )

        reasons: list[str] = []
        if exploration:
            entry_prob = float(self.config.get("exploration_random_entry_prob", 0.15))
            exit_prob = float(self.config.get("exploration_random_exit_prob", 0.10))
            if position_qty <= 0 and random.random() < max(0.0, min(1.0, entry_prob)):
                buy_condition = True
                sell_condition = False
                reasons = ["Exploration: random entry."]
            elif position_qty > 0 and random.random() < max(0.0, min(1.0, exit_prob)):
                sell_condition = True
                buy_condition = False
                reasons = ["Exploration: random exit."]

        if buy_condition:
            base_confidence = self._confidence(latest_rsi, stoch_rsi, ema_cross, momentum_value, direction=1)
            boosted_confidence = min(1.0, base_confidence + ml_confidence * self.config["ml_confidence_boost"])
            if exploration:
                boosted_confidence = max(float(self.config.get("exploration_confidence_floor", 0.65)), boosted_confidence)
            if not reasons:
                reasons = [
                    f"RSI {latest_rsi:.2f} <= {rsi_buy_th:.1f}",
                    f"StochRSI {stoch_rsi:.2f} <= {stoch_buy_th:.2f}",
                    f"EMA spread {ema_cross:.4f}",
                    f"Momentum {momentum_value:.4f}",
                ]
            # Auto-balance: higher size in low volatility, but aggressive
            size = self._scalping_position_size(atr_val, price, boosted_confidence, ctx.get("capital", 10000))
            # compute stop and take-profit
            stop_dist = atr_val * self.config.get("stop_atr_multiplier", 1.5)
            stop_price = max(0.0, price - stop_dist)
            take_profit = price + stop_dist * self.config.get("profit_target_multiplier", 1.5)
            risk_pct = float(self.config.get("risk_pct_per_trade", 0.005))
            if exploration:
                risk_pct = risk_pct * float(self.config.get("exploration_risk_multiplier", 2.5))
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=boosted_confidence,
                suggested_size=size,
                reasons=reasons,
                metadata=self._metadata(price, latest_rsi, stoch_rsi, ema_cross, momentum_value, ml_ctx)
                | {"stop_loss": stop_price, "take_profit": take_profit, "stop_dist": stop_dist, "exploration": exploration, "risk_pct_per_trade": risk_pct},
            )

        if sell_condition:
            base_confidence = self._confidence(latest_rsi, stoch_rsi, ema_cross, momentum_value, direction=-1)
            boosted_confidence = min(1.0, base_confidence + ml_confidence * self.config["ml_confidence_boost"])
            if exploration:
                boosted_confidence = max(float(self.config.get("exploration_confidence_floor", 0.65)), boosted_confidence)
            if not reasons:
                reasons = [
                    f"RSI {latest_rsi:.2f} >= {rsi_sell_th:.1f}",
                    f"StochRSI {stoch_rsi:.2f} >= {stoch_sell_th:.2f}",
                    f"EMA spread {ema_cross:.4f}",
                    f"Momentum {momentum_value:.4f}",
                ]
            size = self._scalping_position_size(atr_val, price, boosted_confidence, ctx.get("capital", 10000))
            stop_dist = atr_val * self.config.get("stop_atr_multiplier", 1.5)
            stop_price = price + stop_dist
            take_profit = max(0.0, price - stop_dist * self.config.get("profit_target_multiplier", 1.5))
            risk_pct = float(self.config.get("risk_pct_per_trade", 0.005))
            if exploration:
                risk_pct = risk_pct * float(self.config.get("exploration_risk_multiplier", 2.5))
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=boosted_confidence,
                suggested_size=size,
                reasons=reasons,
                metadata=self._metadata(price, latest_rsi, stoch_rsi, ema_cross, momentum_value, ml_ctx)
                | {"stop_loss": stop_price, "take_profit": take_profit, "stop_dist": stop_dist, "exploration": exploration, "risk_pct_per_trade": risk_pct},
            )

        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.1,
            suggested_size=0.0,
            reasons=["No scalping opportunity detected."],
            metadata=self._metadata(price, latest_rsi, stoch_rsi, ema_cross, momentum_value, ml_ctx),
        )

    def _scalping_position_size(self, atr_val: float, price: float, confidence: float, capital: float) -> float:
        """Aggressive position sizing for scalping, auto-balanced by confidence and volatility."""
        if price <= 0 or capital <= 0:
            return self.config.get("min_order_quantity", 0.001)
        # Risk-per-trade sizing: qty = (capital * risk_pct) / (stop_dist * price)
        risk_pct = float(self.config.get("risk_pct_per_trade", 0.005))
        stop_atr_mult = float(self.config.get("stop_atr_multiplier", 1.5))
        stop_dist = (atr_val if atr_val and atr_val > 0 else price * 0.01) * stop_atr_mult
        if stop_dist <= 0:
            stop_dist = price * 0.01
        risk_amount = capital * risk_pct
        qty = (risk_amount) / (stop_dist * price)
        # cap by max_scalp_size fraction of capital
        max_by_cap = (capital * float(self.config.get("max_scalp_size", 0.8))) / price
        qty = min(qty, max_by_cap)
        # scale by confidence (more confidence -> allow more size)
        qty = qty * max(0.1, min(1.0, confidence))
        min_qty = float(self.config.get("min_order_quantity", 0.001))
        # round to sensible precision
        qty = max(min_qty, round(qty, 6))
        return qty

    def _confidence(
        self,
        rsi_value: float,
        stoch_rsi: float,
        ema_cross: float,
        momentum_value: float,
        *,
        direction: int,
    ) -> float:
        rsi_component = abs(self.config["rsi_buy" if direction == 1 else "rsi_sell"] - rsi_value) / 25  # Tighter
        stoch_component = abs(stoch_rsi - (self.config["stoch_rsi_buy"] if direction == 1 else self.config["stoch_rsi_sell"]))
        ema_component = min(abs(ema_cross) / 0.1, 1.0)  # Smaller EMA spread for scalping
        try:
            mom_th = float(self.config.get("momentum_threshold", 0.05))
        except Exception:
            mom_th = 0.05
        momentum_component = min(abs(momentum_value) / max(mom_th, 1e-6), 1.0)
        return min(1.0, 0.3 * (rsi_component + stoch_component + ema_component + momentum_component))

    def _metadata(
        self,
        price: float,
        rsi_value: float,
        stoch_rsi: float,
        ema_cross: float,
        momentum_value: float,
        ml_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "price": price,
            "rsi": rsi_value,
            "stoch_rsi": stoch_rsi,
            "ema_spread": ema_cross,
            "momentum": momentum_value,
            "ml_confidence": ml_ctx.get("confidence", 0.0),
            "ml_probability_up": ml_ctx.get("probability_up", 0.5),
            "profile": self.description,
        }


def _ema(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        return float(values[-1]) if values else 0.0
    multiplier = 2 / (period + 1)
    ema_val = float(values[0])
    for price in values[1:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return float(ema_val)


def _stoch_rsi(rsi_series: Sequence[float], period: int = 14) -> float:
    if len(rsi_series) < period:
        return 0.5
    window = rsi_series[-period:]
    highest = max(window)
    lowest = min(window)
    denominator = highest - lowest if highest != lowest else 1.0
    return float((window[-1] - lowest) / denominator)


def _momentum(closes: Sequence[float], period: int) -> float:
    if len(closes) <= period:
        return 0.0
    return float(closes[-1] - closes[-period])
