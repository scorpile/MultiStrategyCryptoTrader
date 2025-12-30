"""Moderate strategy blending RSI, MACD, and Bollinger signals."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .strategy_base import BaseStrategy


class ModerateStrategy(BaseStrategy):
    """Balanced configuration (legacy EntryExit logic)."""

    description = "RSI + MACD + Bollinger blend for balanced trade frequency."

    def name(self) -> str:
        return "moderate"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "rsi_buy": 30,
            "rsi_sell": 70,
            "macd_confirmation": 0.0,
            "bollinger_bias": 0.01,
        }

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        if not closes:
            return self._build_signal(
                decision="hold",
                price=0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["No closing prices provided."],
            )
        price = float(closes[-1])

        rsi_series = indicators.get("rsi") or []
        macd = indicators.get("macd") or {}
        boll = indicators.get("bollinger") or {}
        atr_series = indicators.get("atr") or []

        latest_rsi = self._latest(rsi_series, default=50.0)
        hist = self._latest(macd.get("histogram"), default=0.0)
        lower_band = self._latest(boll.get("lower"), default=price)
        upper_band = self._latest(boll.get("upper"), default=price)
        atr_val = self._latest(atr_series, default=price * 0.01)

        buy_condition = (
            latest_rsi <= self.config["rsi_buy"]
            and hist >= self.config["macd_confirmation"]
            and price <= lower_band * (1 + self.config["bollinger_bias"])
        )
        sell_condition = (
            latest_rsi >= self.config["rsi_sell"]
            and hist <= -self.config["macd_confirmation"]
            and price >= upper_band * (1 - self.config["bollinger_bias"])
        )

        size = self._volatility_position_size(atr_value=atr_val, price=price, aggressiveness=0.9)
        metadata = {
            "last_price": price,
            "rsi": latest_rsi,
            "macd_hist": hist,
            "strategy": self.name(),
        }

        if buy_condition:
            confidence = min(1.0, (self.config["rsi_buy"] - latest_rsi) / 30 + abs(hist))
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=[
                    f"RSI {latest_rsi:.2f} <= {self.config['rsi_buy']}",
                    "MACD histogram supportive.",
                    "Price near lower Bollinger band.",
                ],
                metadata=metadata,
            )

        if sell_condition:
            confidence = min(1.0, (latest_rsi - self.config["rsi_sell"]) / 30 + abs(hist))
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=[
                    f"RSI {latest_rsi:.2f} >= {self.config['rsi_sell']}",
                    "MACD histogram negative.",
                    "Price approaching upper Bollinger band.",
                ],
                metadata=metadata,
            )

        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.25,
            suggested_size=0.0,
            reasons=["No aligned signals detected."],
            metadata=metadata,
        )
