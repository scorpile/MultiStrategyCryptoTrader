"""Aggressive strategy favoring momentum + oscillator confluence."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .strategy_base import BaseStrategy


class AggressiveStrategy(BaseStrategy):
    """Higher-frequency strategy leveraging RSI, StochRSI, EMA crosses, momentum."""

    description = "RSI(14) + StochRSI + EMA cross + momentum, tuned for rapid trades."

    def name(self) -> str:
        return "aggressive"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "rsi_buy": 40,
            "rsi_sell": 60,
            "stoch_rsi_buy": 0.25,
            "stoch_rsi_sell": 0.75,
            "ema_fast_period": 9,
            "ema_slow_period": 21,
            "momentum_period": 8,
            "momentum_threshold": 0.15,
        }

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        if len(closes) < max(self.config["ema_slow_period"], self.config["momentum_period"]) + 2:
            return self._build_signal(
                decision="hold",
                price=closes[-1] if closes else 0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Need more history for aggressive signals."],
            )

        price = float(closes[-1])
        rsi_series = indicators.get("rsi") or []
        stoch_rsi = _stoch_rsi(rsi_series)
        ema_fast = _ema(closes, self.config["ema_fast_period"])
        ema_slow = _ema(closes, self.config["ema_slow_period"])
        atr_series = indicators.get("atr") or []
        atr_val = self._latest(atr_series, default=price * 0.008)
        momentum_value = _momentum(closes, self.config["momentum_period"])

        latest_rsi = self._latest(rsi_series, default=50.0)
        ema_cross = ema_fast - ema_slow

        buy_condition = (
            latest_rsi <= self.config["rsi_buy"]
            and stoch_rsi <= self.config["stoch_rsi_buy"]
            and ema_cross > 0
            and momentum_value >= self.config["momentum_threshold"]
        )
        sell_condition = (
            latest_rsi >= self.config["rsi_sell"]
            and stoch_rsi >= self.config["stoch_rsi_sell"]
            and ema_cross < 0
            and momentum_value <= -self.config["momentum_threshold"]
        )

        reasons: list[str] = []
        if buy_condition:
            confidence = self._confidence(latest_rsi, stoch_rsi, ema_cross, momentum_value, direction=1)
            reasons = [
                f"RSI {latest_rsi:.2f} <= {self.config['rsi_buy']}",
                f"StochRSI {stoch_rsi:.2f} <= {self.config['stoch_rsi_buy']}",
                f"EMA{self.config['ema_fast_period']} above EMA{self.config['ema_slow_period']}",
                f"Momentum {momentum_value:.2f} exceeds threshold",
            ]
            size = self._volatility_position_size(atr_value=atr_val, price=price, aggressiveness=1.3)
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=reasons,
                metadata=self._metadata(price, latest_rsi, stoch_rsi, ema_cross, momentum_value),
            )

        if sell_condition:
            confidence = self._confidence(latest_rsi, stoch_rsi, ema_cross, momentum_value, direction=-1)
            reasons = [
                f"RSI {latest_rsi:.2f} >= {self.config['rsi_sell']}",
                f"StochRSI {stoch_rsi:.2f} >= {self.config['stoch_rsi_sell']}",
                f"EMA{self.config['ema_fast_period']} below EMA{self.config['ema_slow_period']}",
                f"Momentum {momentum_value:.2f} below -threshold",
            ]
            size = self._volatility_position_size(atr_value=atr_val, price=price, aggressiveness=1.3)
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=reasons,
                metadata=self._metadata(price, latest_rsi, stoch_rsi, ema_cross, momentum_value),
            )

        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.2,
            suggested_size=0.0,
            reasons=["Signals misaligned for aggressive trade."],
            metadata=self._metadata(price, latest_rsi, stoch_rsi, ema_cross, momentum_value),
        )

    def _confidence(
        self,
        rsi_value: float,
        stoch_rsi: float,
        ema_cross: float,
        momentum_value: float,
        *,
        direction: int,
    ) -> float:
        rsi_component = abs(self.config["rsi_buy" if direction == 1 else "rsi_sell"] - rsi_value) / 30
        stoch_component = abs(stoch_rsi - (self.config["stoch_rsi_buy"] if direction == 1 else self.config["stoch_rsi_sell"]))
        ema_component = min(abs(ema_cross) / 0.2, 1.0)
        momentum_component = min(abs(momentum_value) / self.config["momentum_threshold"], 1.0)
        return min(1.0, 0.25 * (rsi_component + stoch_component + ema_component + momentum_component))

    def _metadata(
        self,
        price: float,
        rsi_value: float,
        stoch_rsi: float,
        ema_cross: float,
        momentum_value: float,
    ) -> Dict[str, Any]:
        return {
            "price": price,
            "rsi": rsi_value,
            "stoch_rsi": stoch_rsi,
            "ema_spread": ema_cross,
            "momentum": momentum_value,
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
