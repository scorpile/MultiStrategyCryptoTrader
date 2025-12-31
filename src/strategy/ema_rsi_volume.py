"""EMA crossover + RSI + volume confirmation scalping strategy.

Based on the document `ScalpingStrategies.md`:
- Entry: EMA fast crosses above EMA slow, RSI confirms momentum, and volume spikes.
  (Spot-only bot => we only take LONG entries.)
- Exit: EMA fast crosses below EMA slow, or RSI rolls over from overbought.

Stops/TP:
- The scheduler provides ATR-based stop-loss/take-profit when the strategy exposes
  `stop_atr_multiplier` and `profit_target_multiplier` in its config.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .strategy_base import BaseStrategy
from . import technicals


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    p = max(int(period), 1)
    multiplier = 2 / (p + 1)
    ema_values = [float(values[0])]
    for price in values[1:]:
        ema_values.append((float(price) - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


class EmaRsiVolumeStrategy(BaseStrategy):
    """Trend scalping: EMA cross + RSI confirmation + volume spike."""

    description = "EMA(5/20) cross + RSI(7) confirmation + volume spike (spot long-only, 5m signals + 1m risk monitoring)."

    def name(self) -> str:
        return "ema_rsi_volume"

    def config_schema(self) -> Dict[str, Any]:
        return {
            # Signal timeframe (scheduler uses this to fetch indicators)
            "signal_interval": "5m",
            "ema_fast_period": 5,
            "ema_slow_period": 20,
            "rsi_period": 7,
            "rsi_buy_threshold": 30.0,
            "rsi_sell_threshold": 70.0,
            "require_rsi_cross": True,  # require RSI cross above buy threshold
            "volume_sma_period": 20,
            "require_volume_spike": True,
            "volume_spike_multiplier": 1.5,  # vol >= sma * mult
            "min_ema_separation_pct": 0.0,  # avoid tiny crosses (0 disables)
            # Risk-based sizing + exits (used by scheduler)
            "risk_pct_per_trade": 0.005,  # 0.5% equity risk per trade
            "stop_atr_multiplier": 2.0,
            "profit_target_multiplier": 3.0,
            # Trailing stop (scheduler-managed, paper mode)
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 2.0,
            "trailing_stop_activation_atr": 1.0,  # start trailing once +1 ATR in profit
            "trailing_stop_min_step_atr": 0.25,  # move stop only if it advances by >= 0.25 ATR
            "trailing_stop_ignore_take_profit": True,  # let winners run; exits via trail/strategy sell
            # Optional ML filter/boost
            "ml_filter_enabled": False,
            "ml_min_probability_up": 0.55,
            "ml_min_confidence": 0.15,
        }

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        volumes = indicators.get("volumes") or []
        if not closes:
            return self._build_signal(
                decision="hold",
                price=0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["No closing prices provided."],
            )

        ema_fast_period = int(self.config.get("ema_fast_period", 5) or 5)
        ema_slow_period = int(self.config.get("ema_slow_period", 20) or 20)
        rsi_period = int(self.config.get("rsi_period", 7) or 7)
        vol_sma_period = int(self.config.get("volume_sma_period", 20) or 20)

        min_len = max(ema_slow_period + 2, rsi_period + 2, vol_sma_period + 1)
        if len(closes) < min_len:
            return self._build_signal(
                decision="hold",
                price=float(closes[-1]),
                confidence=0.0,
                suggested_size=0.0,
                reasons=[f"Need >= {min_len} candles for EMA/RSI/volume confirmation."],
            )

        price = float(closes[-1])

        ema_fast_series = indicators.get("ema_fast")
        ema_slow_series = indicators.get("ema_slow")
        if isinstance(ema_fast_series, list) and len(ema_fast_series) == len(closes):
            ema_fast = ema_fast_series
        else:
            ema_fast = _ema_series(closes, ema_fast_period)
        if isinstance(ema_slow_series, list) and len(ema_slow_series) == len(closes):
            ema_slow = ema_slow_series
        else:
            ema_slow = _ema_series(closes, ema_slow_period)
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["EMA series unavailable."],
            )

        fast_prev, fast_now = float(ema_fast[-2]), float(ema_fast[-1])
        slow_prev, slow_now = float(ema_slow[-2]), float(ema_slow[-1])
        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        rsi_series = indicators.get("rsi")
        if not isinstance(rsi_series, list) or len(rsi_series) != len(closes):
            rsi_series = technicals.compute_rsi(closes, period=rsi_period)
        rsi_prev = float(rsi_series[-2]) if len(rsi_series) >= 2 else 50.0
        rsi_now = float(rsi_series[-1]) if rsi_series else 50.0
        rsi_buy_th = float(self.config.get("rsi_buy_threshold", 30.0) or 30.0)
        rsi_sell_th = float(self.config.get("rsi_sell_threshold", 70.0) or 70.0)

        if bool(self.config.get("require_rsi_cross", True)):
            rsi_buy_ok = rsi_prev < rsi_buy_th <= rsi_now
        else:
            rsi_buy_ok = rsi_now >= rsi_buy_th and rsi_now >= rsi_prev
        rsi_sell_ok = rsi_now >= rsi_sell_th and rsi_now < rsi_prev

        volume_now = float(volumes[-1]) if volumes else 0.0
        volume_sma = None
        vol_sma_series = indicators.get("volume_sma")
        if isinstance(vol_sma_series, list) and len(vol_sma_series) == len(closes):
            try:
                volume_sma = float(vol_sma_series[-1])
            except Exception:
                volume_sma = None
        if volume_sma is None:
            try:
                sma_series = technicals.compute_volume_sma(volumes, period=vol_sma_period)
                volume_sma = float(sma_series[-1]) if sma_series else 0.0
            except Exception:
                volume_sma = 0.0

        volume_ratio = (volume_now / volume_sma) if volume_sma and volume_sma > 0 else 1.0
        spike_mult = float(self.config.get("volume_spike_multiplier", 1.5) or 1.5)
        volume_ok = (volume_ratio >= spike_mult) if bool(self.config.get("require_volume_spike", True)) else True

        min_sep = float(self.config.get("min_ema_separation_pct", 0.0) or 0.0)
        sep_pct = abs(fast_now - slow_now) / price if price > 0 else 0.0
        ema_sep_ok = (sep_pct >= min_sep) if min_sep > 0 else True

        # Optional ML filter (if enabled).
        ml_ctx = (context or {}).get("ml") or {}
        if bool(self.config.get("ml_filter_enabled", False)):
            try:
                prob_up = float(ml_ctx.get("probability_up", 0.5))
                ml_conf = float(ml_ctx.get("confidence", 0.0))
            except Exception:
                prob_up, ml_conf = 0.5, 0.0
            ml_ok = prob_up >= float(self.config.get("ml_min_probability_up", 0.55) or 0.55) and ml_conf >= float(
                self.config.get("ml_min_confidence", 0.15) or 0.15
            )
        else:
            ml_ok = True

        ctx = context or {}
        position_qty = float(ctx.get("position_qty", 0.0) or 0.0)

        metadata = {
            "price": price,
            "ema_fast": fast_now,
            "ema_slow": slow_now,
            "ema_sep_pct": sep_pct,
            "rsi": rsi_now,
            "volume": volume_now,
            "volume_sma": volume_sma,
            "volume_ratio": volume_ratio,
            "ml": {"probability_up": ml_ctx.get("probability_up"), "confidence": ml_ctx.get("confidence"), "model": ml_ctx.get("model_name")},
        }

        if position_qty <= 0 and cross_up and rsi_buy_ok and volume_ok and ema_sep_ok and ml_ok:
            cross_strength = abs(fast_now - slow_now) / price if price > 0 else 0.0
            rsi_strength = max(0.0, min(1.0, (rsi_now - rsi_buy_th) / max(1.0, 100.0 - rsi_buy_th)))
            vol_strength = max(0.0, min(1.0, (volume_ratio - 1.0) / max(1e-9, spike_mult - 1.0)))
            confidence = _clamp01(0.15 + 4.0 * cross_strength + 0.35 * rsi_strength + 0.35 * vol_strength)
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=confidence,
                suggested_size=0.0,  # scheduler uses risk-based sizing
                reasons=[
                    f"EMA{ema_fast_period} crossed above EMA{ema_slow_period}.",
                    f"RSI{rsi_period} confirmed (>= {rsi_buy_th:.0f}).",
                    f"Volume spike x{volume_ratio:.2f} (>= {spike_mult:.2f})." if bool(self.config.get("require_volume_spike", True)) else "Volume filter disabled.",
                ],
                metadata=metadata
                | {
                    "risk_pct_per_trade": float(self.config.get("risk_pct_per_trade", 0.005) or 0.005),
                },
            )

        if position_qty > 0 and (cross_down or rsi_sell_ok):
            confidence = _clamp01(0.6 if cross_down else 0.45)
            reasons = []
            if cross_down:
                reasons.append(f"EMA{ema_fast_period} crossed below EMA{ema_slow_period}.")
            if rsi_sell_ok:
                reasons.append(f"RSI{rsi_period} rolling over from overbought (>= {rsi_sell_th:.0f}).")
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=confidence,
                suggested_size=0.0,  # sell full position
                reasons=reasons or ["Exit condition met."],
                metadata=metadata,
            )

        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.15,
            suggested_size=0.0,
            reasons=["No entry/exit conditions met."],
            metadata=metadata,
        )
