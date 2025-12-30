"""Bollinger Bands range scalping strategy (mean reversion).

Based on `ScalpingStrategies.md` — Estrategia 2: "Scalping en Rangos con Bandas de Bollinger".

Idea (spot long-only):
- Entry (BUY): price touches/penetrates the lower Bollinger band while RSI is oversold,
  optionally confirming a reversal (engulfing / bounce).
- Exit (SELL): close near the mid/upper band, or when RSI rolls over from overbought.

Stops/TP:
- The strategy emits absolute `stop_loss` / `take_profit` levels in metadata so the scheduler
  (and backtester) can use risk sizing + attach SL/TP metadata to the order.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .strategy_base import BaseStrategy
from . import technicals


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _last_ohlc_tuple(ohlcv: Sequence[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    if not ohlcv:
        return None
    last = ohlcv[-1] or {}
    try:
        return (float(last.get("open")), float(last.get("high")), float(last.get("low")), float(last.get("close")))
    except Exception:
        return None


def _bullish_reversal(indicators: Dict[str, Any]) -> bool:
    """Lightweight reversal confirmation compatible with live + backtesting payloads."""
    closes = indicators.get("closes") or []
    if len(closes) >= 2:
        try:
            if float(closes[-1]) > float(closes[-2]):
                return True
        except Exception:
            pass

    engulfing_series = indicators.get("engulfing")
    if isinstance(engulfing_series, list) and engulfing_series:
        try:
            return int(engulfing_series[-1] or 0) == 1
        except Exception:
            return False

    ohlcv = indicators.get("ohlcv") or []
    last2 = ohlcv[-2:] if len(ohlcv) >= 2 else []
    tuples = []
    for row in last2:
        try:
            tuples.append((float(row.get("open")), float(row.get("high")), float(row.get("low")), float(row.get("close"))))
        except Exception:
            return False
    if len(tuples) >= 2:
        patterns = technicals.detect_engulfing(tuples)
        try:
            return int(patterns[-1] or 0) == 1
        except Exception:
            return False
    return False


class BollingerRangeStrategy(BaseStrategy):
    """Mean-reversion scalper for sideways markets using Bollinger Bands."""

    description = "Bollinger range scalping: buy lower band w/ RSI oversold; exit mid/upper band (spot long-only)."

    def name(self) -> str:
        return "bollinger_range"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_stddev": 2.0,
            "touch_tolerance_pct": 0.001,  # allow near-touch (0.1%)
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "confirm_reversal": True,
            # Simple "range regime" filter (disable by setting to 0.0)
            "max_ema_separation_pct": 0.003,  # abs(EMA20-EMA50)/price <= 0.3%
            # Risk sizing + exits (scheduler/backtester)
            "risk_pct_per_trade": 0.005,
            "stop_atr_multiplier": 1.5,
            "profit_target_multiplier": 1.5,
            "take_profit_target": "middle",  # middle|upper
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
        ctx = context or {}
        position_qty = float(ctx.get("position_qty", 0.0) or 0.0)

        boll = indicators.get("bollinger") or {}
        try:
            lower_series = boll.get("lower") or []
            middle_series = boll.get("middle") or []
            upper_series = boll.get("upper") or []
            lower = float(lower_series[-1]) if lower_series else 0.0
            middle = float(middle_series[-1]) if middle_series else 0.0
            upper = float(upper_series[-1]) if upper_series else 0.0
        except Exception:
            lower = middle = upper = 0.0

        if lower <= 0.0 or middle <= 0.0 or upper <= 0.0:
            # Fallback: compute on demand.
            try:
                computed = technicals.compute_bollinger_bands(closes, period=int(self.config.get("bollinger_period", 20) or 20))
                lower = float((computed.get("lower") or [0.0])[-1] or 0.0)
                middle = float((computed.get("middle") or [0.0])[-1] or 0.0)
                upper = float((computed.get("upper") or [0.0])[-1] or 0.0)
            except Exception:
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=0.0,
                    suggested_size=0.0,
                    reasons=["Bollinger bands unavailable."],
                )

        rsi_period = int(self.config.get("rsi_period", 14) or 14)
        rsi_series = indicators.get("rsi")
        if not isinstance(rsi_series, list) or len(rsi_series) != len(closes):
            rsi_series = technicals.compute_rsi(closes, period=rsi_period)
        rsi_now = float(rsi_series[-1]) if rsi_series else 50.0
        rsi_prev = float(rsi_series[-2]) if len(rsi_series) >= 2 else rsi_now

        rsi_oversold = float(self.config.get("rsi_oversold", 30.0) or 30.0)
        rsi_overbought = float(self.config.get("rsi_overbought", 70.0) or 70.0)
        tol = max(0.0, float(self.config.get("touch_tolerance_pct", 0.001) or 0.001))

        # Range filter using EMA separation (EMA20/EMA50 provided by scheduler/backtester).
        ema_sep_ok = True
        max_sep = float(self.config.get("max_ema_separation_pct", 0.0) or 0.0)
        if max_sep > 0 and price > 0:
            try:
                ema_fast = float((indicators.get("ema_fast") or [0.0])[-1] or 0.0)
                ema_slow = float((indicators.get("ema_slow") or [0.0])[-1] or 0.0)
                sep_pct = abs(ema_fast - ema_slow) / price if price > 0 else 0.0
                ema_sep_ok = sep_pct <= max_sep
            except Exception:
                ema_sep_ok = True

        near_lower = price <= (lower * (1.0 + tol)) if lower > 0 else False
        oversold_ok = rsi_now <= rsi_oversold
        reversal_ok = _bullish_reversal(indicators) if bool(self.config.get("confirm_reversal", True)) else True

        # Exit trigger: mean reversion achieved (mid/upper) or RSI rollover.
        take_profit_target = str(self.config.get("take_profit_target", "middle") or "middle").lower()
        target_price = middle if take_profit_target == "middle" else upper
        rsi_rollover = rsi_now >= rsi_overbought and rsi_now < rsi_prev

        atr_series = indicators.get("atr") or []
        try:
            atr_val = float(atr_series[-1]) if atr_series else price * 0.008
        except Exception:
            atr_val = price * 0.008

        metadata = {
            "price": price,
            "bollinger": {"lower": lower, "middle": middle, "upper": upper},
            "rsi": rsi_now,
            "ema_fast": float((indicators.get("ema_fast") or [0.0])[-1] or 0.0) if price > 0 else 0.0,
            "ema_slow": float((indicators.get("ema_slow") or [0.0])[-1] or 0.0) if price > 0 else 0.0,
        }

        if position_qty <= 0 and near_lower and oversold_ok and reversal_ok and ema_sep_ok:
            stop_mult = float(self.config.get("stop_atr_multiplier", 1.5) or 1.5)
            profit_mult = float(self.config.get("profit_target_multiplier", 1.5) or 1.5)

            # Stop: ATR-based, nudged below the latest low for realism.
            stop_loss = price - max(0.0, atr_val * stop_mult)
            try:
                last_low = float((indicators.get("lows") or [price])[-1] or price)
                stop_loss = min(stop_loss, last_low - max(0.0, atr_val * 0.10))
            except Exception:
                pass
            stop_loss = max(0.0, float(stop_loss))

            # Take-profit: band target; if invalid, fall back to R-multiple.
            take_profit = float(target_price) if target_price > price else float(price + (abs(price - stop_loss) * max(1.0, profit_mult)))

            band_width = max(upper - lower, 1e-9)
            dist = max(0.0, lower - price) / band_width  # >0 when price penetrates lower band
            rsi_strength = max(0.0, (rsi_oversold - rsi_now) / max(1.0, rsi_oversold))
            confidence = _clamp01(0.25 + 0.7 * dist + 0.5 * rsi_strength + (0.1 if reversal_ok else 0.0))

            return self._build_signal(
                decision="buy",
                price=price,
                confidence=confidence,
                suggested_size=0.0,  # scheduler uses risk sizing
                reasons=[
                    f"Price near lower Bollinger band (<= {lower:.4f}).",
                    f"RSI{rsi_period} oversold ({rsi_now:.1f} <= {rsi_oversold:.0f}).",
                    "Reversal confirmation." if bool(self.config.get("confirm_reversal", True)) else "Reversal filter disabled.",
                ],
                metadata=metadata
                | {
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                    "risk_pct_per_trade": float(self.config.get("risk_pct_per_trade", 0.005) or 0.005),
                    "take_profit_target": take_profit_target,
                },
            )

        if position_qty > 0 and (price >= target_price or rsi_rollover):
            reasons = []
            if price >= target_price:
                reasons.append(f"Reached {take_profit_target} Bollinger target (>= {target_price:.4f}).")
            if rsi_rollover:
                reasons.append(f"RSI{rsi_period} rollover from overbought (>= {rsi_overbought:.0f}).")
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=_clamp01(0.6 if price >= target_price else 0.45),
                suggested_size=0.0,
                reasons=reasons or ["Exit condition met."],
                metadata=metadata,
            )

        hold_reasons = []
        if not ema_sep_ok:
            hold_reasons.append("Market looks trending (EMA separation too large).")
        else:
            hold_reasons.append("No entry/exit conditions met.")
        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.15,
            suggested_size=0.0,
            reasons=hold_reasons,
            metadata=metadata,
        )

