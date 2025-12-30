"""Multi-factor strategy combining technical + ML + RL signals."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import math

from .strategy_base import BaseStrategy


class MultiFactorStrategy(BaseStrategy):
    """Weighted scoring engine ingesting multiple indicators and ML/RL context."""

    description = (
        "RSI + MACD + Bollinger + EMA trend + volume spike + ATR regime + candle pattern + ML/RL."
    )

    def name(self) -> str:
        return "multifactor"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "score_buy_threshold": 0.35,
            "score_sell_threshold": -0.35,
            "weights": {
                "rsi": 0.15,
                "macd": 0.15,
                "bollinger": 0.1,
                "trend": 0.2,
                "volume": 0.1,
                "atr": 0.1,
                "candle": 0.05,
                "ml": 0.15,
            },
            "ema_fast_period": 20,
            "ema_slow_period": 50,
            "volume_window": 20,
        }

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        volumes = indicators.get("volumes") or []
        ohlcv_rows = indicators.get("ohlcv") or []
        if len(closes) < self.config["ema_slow_period"]:
            return self._build_signal(
                decision="hold",
                price=closes[-1] if closes else 0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Need more candles for multi-factor scoring."],
            )

        price = float(closes[-1])
        rsi_series = indicators.get("rsi") or []
        macd = indicators.get("macd") or {}
        boll = indicators.get("bollinger") or {}
        atr_series = indicators.get("atr") or []
        ema_fast = indicators.get("ema_fast") or _ema_series(closes, self.config["ema_fast_period"])
        ema_slow = indicators.get("ema_slow") or _ema_series(closes, self.config["ema_slow_period"])

        latest_rsi = self._latest(rsi_series, default=50.0)
        macd_hist = self._latest(macd.get("histogram"), default=0.0)
        middle_band = self._latest(boll.get("middle"), default=price)
        upper_band = self._latest(boll.get("upper"), default=price)
        lower_band = self._latest(boll.get("lower"), default=price)
        atr_val = self._latest(atr_series, default=price * 0.01)
        ema_fast_latest = ema_fast[-1] if isinstance(ema_fast, list) else float(ema_fast)
        ema_slow_latest = ema_slow[-1] if isinstance(ema_slow, list) else float(ema_slow)
        volume_spike = _volume_spike(volumes, window=self.config["volume_window"])
        atr_regime = _atr_regime(atr_val, price)
        candle_score = _candle_pattern_score(ohlcv_rows)

        ml_context = (context or {}).get("ml") or {}
        rl_policy = (context or {}).get("rl") or {}

        weights = self._compose_weights(rl_policy)
        contributions = {
            "rsi": weights["rsi"] * _rsi_component(latest_rsi),
            "macd": weights["macd"] * _macd_component(macd_hist),
            "bollinger": weights["bollinger"] * _bollinger_component(price, lower_band, upper_band, middle_band),
            "trend": weights["trend"] * _trend_component(ema_fast_latest, ema_slow_latest),
            "volume": weights["volume"] * _volume_component(volume_spike),
            "atr": weights["atr"] * _atr_component(atr_regime),
            "candle": weights["candle"] * candle_score,
            "ml": weights["ml"] * _ml_component(ml_context),
        }

        total_score = sum(contributions.values())
        decision = "hold"
        reasons: list[str] = []
        if total_score >= self.config["score_buy_threshold"]:
            decision = "buy"
            reasons.append(f"Score {total_score:.2f} >= buy threshold {self.config['score_buy_threshold']}")
        elif total_score <= self.config["score_sell_threshold"]:
            decision = "sell"
            reasons.append(f"Score {total_score:.2f} <= sell threshold {self.config['score_sell_threshold']}")
        else:
            reasons.append("Score within neutral band.")

        suggested_size = self._volatility_position_size(
            atr_value=atr_val,
            price=price,
            aggressiveness=1.0 + max(0.0, abs(total_score)),
        )
        confidence = min(1.0, abs(total_score))

        metadata = {
            "score": total_score,
            "contributions": contributions,
            "ml_context": ml_context,
            "rl_policy": rl_policy,
            "volume_spike": volume_spike,
            "atr_regime": atr_regime,
            "candle_score": candle_score,
            "price_position": _normalize(price, lower_band, upper_band),
        }

        return self._build_signal(
            decision=decision,
            price=price,
            confidence=confidence,
            suggested_size=suggested_size,
            reasons=reasons,
            metadata=metadata,
        )

    def _compose_weights(self, rl_policy: Dict[str, Any]) -> Dict[str, float]:
        weights = dict(self.config["weights"])
        rl_weights = rl_policy.get("weights") if isinstance(rl_policy, dict) else None
        if rl_weights:
            for key, value in rl_weights.items():
                if key in weights:
                    weights[key] = max(0.0, weights[key] + float(value))
        # Normalize so total weight is <= 1 to keep score bounded.
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    multiplier = 2 / (period + 1)
    ema_values = [float(values[0])]
    for price in values[1:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _volume_spike(volumes: Sequence[float], window: int) -> float:
    if not volumes or len(volumes) < window:
        return 0.0
    recent = volumes[-window:]
    avg = sum(recent) / len(recent)
    if avg == 0:
        return 0.0
    return (recent[-1] - avg) / avg


def _atr_regime(atr_value: float, price: float) -> float:
    if price <= 0:
        return 0.0
    atr_pct = atr_value / price
    return min(max(atr_pct / 0.02, 0.0), 1.0)


def _candle_pattern_score(ohlcv_rows: Sequence[Dict[str, float]]) -> float:
    if not ohlcv_rows or len(ohlcv_rows) < 2:
        return 0.0
    last = ohlcv_rows[-1]
    prev = ohlcv_rows[-2]
    last_body = last["close"] - last["open"]
    prev_body = prev["close"] - prev["open"]
    if last_body > 0 and prev_body < 0 and last["close"] > prev["open"]:
        return 0.6  # simple bullish engulfing
    if last_body < 0 and prev_body > 0 and last["close"] < prev["open"]:
        return -0.6  # bearish engulfing
    return 0.0


def _normalize(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return (value - lower) / (upper - lower)


def _rsi_component(rsi_value: float) -> float:
    return (50 - rsi_value) / 50


def _macd_component(hist: float) -> float:
    return math.tanh(hist)


def _bollinger_component(price: float, lower: float, upper: float, middle: float) -> float:
    if upper == lower:
        return 0.0
    if price <= middle:
        return min(1.0, (middle - price) / (middle - lower + 1e-9))
    return -min(1.0, (price - middle) / (upper - middle + 1e-9))


def _trend_component(ema_fast: float, ema_slow: float) -> float:
    if ema_slow == 0:
        return 0.0
    return math.tanh((ema_fast - ema_slow) / ema_slow)


def _volume_component(spike_value: float) -> float:
    return max(-1.0, min(1.0, spike_value))


def _atr_component(atr_regime: float) -> float:
    # Higher volatility reduces conviction (lean towards hold).
    return -(atr_regime - 0.5)


def _ml_component(ml_context: Dict[str, Any]) -> float:
    if not ml_context:
        return 0.0
    probability = float(ml_context.get("probability_up", 0.5))
    confidence = float(ml_context.get("confidence", 1.0))
    return (probability - 0.5) * 2 * min(1.0, confidence)
