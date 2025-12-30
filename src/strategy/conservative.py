"""Conservative strategy with strict confirmations and low trade frequency."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .strategy_base import BaseStrategy


class ConservativeStrategy(BaseStrategy):
    """Deep-oversold / overbought detector prioritizing accuracy over frequency."""

    description = (
        "Deep RSI + MACD confirmation with Bollinger validation. Targets low drawdown, "
        "few trades, and high selectivity."
    )

    def name(self) -> str:
        return "conservative"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "rsi_buy": 25,
            "rsi_sell": 75,
            "macd_histogram_floor": 0.15,
            "bollinger_band_margin": 0.01,
            "confidence_scale": 0.6,
        }

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        if len(closes) < 5:
            return self._hold_signal(price=closes[-1] if closes else 0.0, reason="Insufficient data window.")

        price = float(closes[-1])
        rsi_series = indicators.get("rsi") or []
        macd = indicators.get("macd") or {}
        boll = indicators.get("bollinger") or {}
        atr_series = indicators.get("atr") or []

        latest_rsi = self._latest(rsi_series, default=50.0)
        hist = self._latest(macd.get("histogram"), default=0.0)
        macd_line = self._latest(macd.get("macd"), default=0.0)
        signal_line = self._latest(macd.get("signal"), default=0.0)
        lower_band = self._latest(boll.get("lower"), default=price)
        upper_band = self._latest(boll.get("upper"), default=price)
        atr_val = self._latest(atr_series, default=price * 0.01)

        buy_condition = (
            latest_rsi <= self.config["rsi_buy"]
            and price <= lower_band * (1 + self.config["bollinger_band_margin"])
            and hist >= self.config["macd_histogram_floor"]
            and macd_line > signal_line
        )
        sell_condition = (
            latest_rsi >= self.config["rsi_sell"]
            and price >= upper_band * (1 - self.config["bollinger_band_margin"])
            and hist <= -self.config["macd_histogram_floor"]
            and macd_line < signal_line
        )

        size = self._volatility_position_size(atr_value=atr_val, price=price, aggressiveness=0.6)
        reasons: list[str]

        if buy_condition:
            confidence = self._confidence_from_distance(latest_rsi, self.config["rsi_buy"], hist)
            reasons = [
                f"RSI {latest_rsi:.2f} <= {self.config['rsi_buy']}",
                "Price testing lower Bollinger band.",
                "MACD histogram positive with line > signal.",
            ]
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=reasons,
                metadata=self._metadata(price, latest_rsi, hist),
            )

        if sell_condition:
            confidence = self._confidence_from_distance(latest_rsi, self.config["rsi_sell"], hist)
            reasons = [
                f"RSI {latest_rsi:.2f} >= {self.config['rsi_sell']}",
                "Price stretching upper Bollinger band.",
                "MACD histogram negative with line < signal.",
            ]
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=reasons,
                metadata=self._metadata(price, latest_rsi, hist),
            )

        return self._hold_signal(price=price, reason="Conditions not extreme enough")

    def _confidence_from_distance(self, rsi_value: float, threshold: float, hist: float) -> float:
        distance = abs(rsi_value - threshold) / 50
        hist_component = min(abs(hist), 1.0)
        return min(1.0, (distance + hist_component) * self.config["confidence_scale"])

    def _metadata(self, price: float, rsi_value: float, hist: float) -> Dict[str, Any]:
        return {
            "last_price": price,
            "rsi": rsi_value,
            "macd_histogram": hist,
            "profile": self.description,
        }

    def _hold_signal(self, *, price: float, reason: str) -> Dict[str, Any]:
        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.15,
            suggested_size=0.0,
            reasons=[reason],
            metadata={"profile": self.description},
        )
