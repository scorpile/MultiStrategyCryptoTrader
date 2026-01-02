"""
EMA cross + RSI + volume + continuation scalping (redesigned).

Key changes vs previous version:
- Entry avoids "cross candle churn": requires cross happened recently + continuation.
- Minimum hold candles to avoid immediate exits.
- Trailing stop is enforced (emits sell when hit), not only updated.
- Exit no longer triggers on RSI dipping around 50 (noise). Uses structure loss + RSI weakness.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .strategy_base import BaseStrategy


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    p = max(int(period), 1)
    multiplier = 2 / (p + 1)
    ema_values = [float(values[0])]
    for price in values[1:]:
        ema_values.append((float(price) - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _sma(values: Sequence[float], period: int) -> list[float]:
    p = max(int(period), 1)
    out: list[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += float(v)
        if i >= p:
            s -= float(values[i - p])
        if i + 1 >= p:
            out.append(s / p)
        else:
            out.append(0.0)
    return out


def _clamp(value: float, low: float, high: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(low)
    return max(float(low), min(float(high), v))


def _last_swing_low(lows: Sequence[float], lookback: int) -> float:
    if not lows or len(lows) < 3:
        return float(lows[-1]) if lows else 0.0
    end = len(lows) - 2
    start = max(1, end - max(1, int(lookback)))
    for idx in range(end, start - 1, -1):
        prev_low = float(lows[idx - 1])
        curr_low = float(lows[idx])
        next_low = float(lows[idx + 1])
        if curr_low <= prev_low and curr_low <= next_low:
            return curr_low
    slice_lows = [float(v) for v in lows[start : end + 1]]
    return min(slice_lows) if slice_lows else float(lows[-1])


def _last_swing_high(highs: Sequence[float], lookback: int) -> float:
    if not highs or len(highs) < 3:
        return float(highs[-1]) if highs else 0.0
    end = len(highs) - 2
    start = max(1, end - max(1, int(lookback)))
    for idx in range(end, start - 1, -1):
        prev_high = float(highs[idx - 1])
        curr_high = float(highs[idx])
        next_high = float(highs[idx + 1])
        if curr_high >= prev_high and curr_high >= next_high:
            return curr_high
    slice_highs = [float(v) for v in highs[start : end + 1]]
    return max(slice_highs) if slice_highs else float(highs[-1])


class EmaRsiVolumeConfirmedStrategy(BaseStrategy):
    """
    Correct scalping structure:
    - Cross is a "regime change", but entry is on continuation (avoid cross candle).
    - Exits are based on (a) trailing stop hit or (b) structure invalidation after min hold.
    """

    description = (
        "EMA(5/20) regime + continuation entry + RSI momentum + volume filter; "
        "enforced trailing stop + structure-based exit."
    )

    def name(self) -> str:
        return "ema_rsi_volume_continuation"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "signal_interval": "3m",
            "ema_fast_period": 5,
            "ema_slow_period": 20,
            "rsi_period": 7,

            # Entry confirmations
            "rsi_entry_min": 55.0,                 # stronger than 50 to avoid noise
            "require_rsi_rising": True,
            "volume_sma_period": 20,
            "require_volume_above_sma": True,
            "require_close_above_emas": True,
            "require_ema_slope": True,

            # Cross / continuation logic
            "cross_lookback_candles": 3,           # cross must have happened within last N candles
            "continuation_min_body_pct": 0.0005,   # require some movement (avoid flat churn)

            # Anti-churn: minimum time in trade
            "min_hold_candles": 3,

            # Risk & orders
            "max_spread_pct": 0.002,
            "swing_lookback": 20,
            "max_loss_pct": 0.02,                  # typical scalping stop cap
            "take_profit_pct": 0.015,
            "take_profit_pct_min": 0.010,
            "take_profit_pct_max": 0.030,
            "risk_pct_per_trade": 0.01,

            # Trailing stop (enforced)
            "trailing_stop_enabled": True,
            "trailing_stop_pct": 0.012,            # 1.2%
            "trailing_stop_activation_pct": 0.008, # activate at +0.8%
            "trailing_stop_min_step_pct": 0.0015,  # avoid micro-adjust spam

            # Shorts optional
            "allow_shorts": False,

            # Optional confidence gating
            "min_confidence": 0.6,
        }

    def _compute_confidence(self, rsi: float, rsi_prev: float, vol_now: float, vol_avg: float) -> float:
        # Keep it simple but meaningful: we use it as a FILTER (min_confidence).
        confidence = 0.65
        if rsi >= 60 and rsi > rsi_prev:
            confidence += 0.10
        if vol_avg > 0 and vol_now >= 1.5 * vol_avg:
            confidence += 0.10
        return min(confidence, 0.95)

    def _update_and_check_trailing_long(
        self, ctx: Dict[str, Any], price: float, cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Returns dict with:
        - action: "none" | "update" | "hit"
        - trailing_stop: float|None
        - reason: str
        """
        entry_price = float(ctx.get("entry_price") or 0.0)
        if entry_price <= 0 or price <= 0:
            return {"action": "none", "trailing_stop": None, "reason": "No entry_price."}

        activation_pct = float(cfg.get("trailing_stop_activation_pct", 0.008) or 0.008)
        trail_pct = float(cfg.get("trailing_stop_pct", 0.012) or 0.012)
        min_step_pct = float(cfg.get("trailing_stop_min_step_pct", 0.0015) or 0.0015)

        # track peak
        peak = float(ctx.get("peak_price") or entry_price)
        if price > peak:
            peak = price
            ctx["peak_price"] = peak

        gain = (peak - entry_price) / entry_price
        if gain < activation_pct:
            return {"action": "none", "trailing_stop": None, "reason": "Trailing not active yet."}

        new_trail = peak * (1 - trail_pct)
        old_trail = ctx.get("trailing_stop")
        if old_trail:
            old_trail = float(old_trail)
            # only raise in meaningful steps
            step = abs(new_trail - old_trail) / max(1e-12, old_trail)
            if new_trail <= old_trail or step < min_step_pct:
                # still check hit
                if price <= old_trail:
                    return {"action": "hit", "trailing_stop": old_trail, "reason": "Trailing stop hit."}
                return {"action": "none", "trailing_stop": old_trail, "reason": "Trail unchanged."}
        # set/update trail
        ctx["trailing_stop"] = new_trail
        if price <= new_trail:
            return {"action": "hit", "trailing_stop": new_trail, "reason": "Trailing stop hit on update."}
        return {"action": "update", "trailing_stop": new_trail, "reason": "Trailing stop raised."}

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        opens = indicators.get("opens") or []
        highs = indicators.get("highs") or []
        lows = indicators.get("lows") or []
        volumes = indicators.get("volumes") or []

        if not closes or len(closes) < 3:
            return self._build_signal(
                decision="hold",
                price=float(closes[-1]) if closes else 0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Insufficient candle data."],
            )

        cfg = self.config
        price = float(closes[-1])
        ema_fast_period = int(cfg.get("ema_fast_period", 5) or 5)
        ema_slow_period = int(cfg.get("ema_slow_period", 20) or 20)
        rsi_period = int(cfg.get("rsi_period", 7) or 7)
        vol_sma_period = int(cfg.get("volume_sma_period", 20) or 20)

        min_len = max(ema_slow_period + 5, rsi_period + 5, vol_sma_period + 2, 10)
        if len(closes) < min_len:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=[f"Need >= {min_len} candles for stable confirmed scalping signals."],
            )

        # Indicators (use provided or compute)
        ema_fast_series = indicators.get("ema_fast")
        ema_slow_series = indicators.get("ema_slow")
        ema_fast = (
            ema_fast_series
            if isinstance(ema_fast_series, list) and len(ema_fast_series) == len(closes)
            else _ema_series(closes, ema_fast_period)
        )
        ema_slow = (
            ema_slow_series
            if isinstance(ema_slow_series, list) and len(ema_slow_series) == len(closes)
            else _ema_series(closes, ema_slow_period)
        )

        rsi_series = indicators.get("rsi") or []
        volume_sma = indicators.get("volume_sma") or []
        if not volume_sma:
            volume_sma = _sma(volumes, vol_sma_period)

        if len(ema_fast) < 5 or len(ema_slow) < 5 or len(rsi_series) < 3 or len(volume_sma) < 3:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Indicators not ready."],
            )

        ema_fast_now = float(ema_fast[-1])
        ema_fast_prev = float(ema_fast[-2])
        ema_slow_now = float(ema_slow[-1])
        ema_slow_prev = float(ema_slow[-2])

        rsi_now = float(rsi_series[-1])
        rsi_prev = float(rsi_series[-2])

        volume_now = float(volumes[-1]) if volumes else 0.0
        volume_avg = float(volume_sma[-1]) if volume_sma else 0.0

        # Context
        ctx = context or {}
        position_qty = float(ctx.get("position_qty", 0.0) or 0.0)
        capital = float(ctx.get("capital", 0.0) or 0.0)

        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_bps = ctx.get("spread_bps")
            if spread_bps is not None:
                try:
                    spread_pct = float(spread_bps) / 10_000.0
                except Exception:
                    spread_pct = None
        max_spread_pct = float(cfg.get("max_spread_pct", 0.002) or 0.002)

        # Common filters
        if spread_pct is not None and spread_pct > max_spread_pct:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=[f"Spread {spread_pct:.4f} > max {max_spread_pct:.4f}."],
            )

        require_volume_above_sma = bool(cfg.get("require_volume_above_sma", True))
        if require_volume_above_sma and (volume_avg <= 0 or volume_now <= volume_avg):
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Volume not above SMA."],
            )

        # ===========
        # POSITION MANAGEMENT (LONG)
        # ===========
        if position_qty > 0:
            # Track bars in trade
            bars_in_trade = int(ctx.get("bars_in_trade", 0) or 0) + 1
            ctx["bars_in_trade"] = bars_in_trade

            # Enforced trailing stop
            if bool(cfg.get("trailing_stop_enabled", True)):
                trail_info = self._update_and_check_trailing_long(ctx, price, cfg)
                if trail_info["action"] == "hit":
                    confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_avg)
                    return self._build_signal(
                        decision="sell",
                        price=price,
                        confidence=confidence,
                        suggested_size=abs(position_qty),
                        reasons=[trail_info["reason"]],
                        metadata={"trigger": "trailing_stop_hit", "trailing_stop": float(trail_info["trailing_stop"] or 0.0)},
                    )

            # Anti-churn: don't invalidate too early
            min_hold = int(cfg.get("min_hold_candles", 3) or 3)
            if bars_in_trade < min_hold:
                confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_avg)
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=confidence,
                    suggested_size=0.0,
                    reasons=[f"Min-hold active ({bars_in_trade}/{min_hold})."],
                    metadata={"trigger": "min_hold"},
                )

            # Structure invalidation (after min-hold)
            # We exit when price loses the fast EMA AND RSI weakens meaningfully (not around 50).
            close_below_fast_now = price < ema_fast_now
            close_below_fast_prev = float(closes[-2]) < float(ema_fast[-2])
            rsi_weak = rsi_now < 45.0 and rsi_now < rsi_prev

            if close_below_fast_now and close_below_fast_prev and rsi_weak:
                confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_avg)
                return self._build_signal(
                    decision="sell",
                    price=price,
                    confidence=confidence,
                    suggested_size=abs(position_qty),
                    reasons=["Invalidation: 2 closes below EMA fast + RSI < 45 weakening."],
                    metadata={"trigger": "structure_invalidation"},
                )

            # Otherwise hold
            confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_avg)
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=confidence,
                suggested_size=0.0,
                reasons=["Position open; managed by trailing/invalidation/TP/SL."],
            )

        # ===========
        # ENTRY LOGIC (FLAT)
        # ===========
        require_close_above_emas = bool(cfg.get("require_close_above_emas", True))
        require_ema_slope = bool(cfg.get("require_ema_slope", True))
        require_rsi_rising = bool(cfg.get("require_rsi_rising", True))
        rsi_entry_min = float(cfg.get("rsi_entry_min", 55.0) or 55.0)

        close_above = price > ema_fast_now and price > ema_slow_now
        ema_slope_up = ema_slow_now > ema_slow_prev

        rsi_ok = (rsi_now >= rsi_entry_min) and (rsi_now > rsi_prev if require_rsi_rising else True)

        # Cross must be recent (avoid late/lag entries), but we DON'T enter on the cross candle itself.
        cross_lookback = int(cfg.get("cross_lookback_candles", 3) or 3)
        cross_lookback = max(1, min(10, cross_lookback))
        crossed_recently = False
        cross_on_last_candle = False

        # Detect any cross-up inside last N candles
        for i in range(1, cross_lookback + 1):
            a_prev = float(ema_fast[-(i + 1)])
            a_now = float(ema_fast[-i])
            b_prev = float(ema_slow[-(i + 1)])
            b_now = float(ema_slow[-i])
            cross_up_i = (a_prev < b_prev) and (a_now >= b_now)
            if cross_up_i:
                crossed_recently = True
                if i == 1:
                    cross_on_last_candle = True
                break

        # Continuation requirement: green candle body with minimum size (avoid micro noise)
        continuation_ok = True
        body_pct_req = float(cfg.get("continuation_min_body_pct", 0.0005) or 0.0005)
        if opens and len(opens) == len(closes):
            body = abs(float(closes[-1]) - float(opens[-1]))
            body_pct = body / max(1e-12, price)
            is_green = float(closes[-1]) > float(opens[-1])
            continuation_ok = is_green and (body_pct >= body_pct_req)

        long_ok = crossed_recently and (not cross_on_last_candle) and rsi_ok and continuation_ok
        if require_close_above_emas:
            long_ok = long_ok and close_above
        if require_ema_slope:
            long_ok = long_ok and ema_slope_up

        if not long_ok:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["No valid continuation setup."],
            )

        # Confidence gating (NOW it actually matters)
        confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_avg)
        min_conf = float(cfg.get("min_confidence", 0.6) or 0.6)
        if confidence < min_conf:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=confidence,
                suggested_size=0.0,
                reasons=[f"Confidence {confidence:.2f} < min {min_conf:.2f}."],
            )

        # Risk-based sizing + SL/TP
        take_profit_pct = _clamp(
            float(cfg.get("take_profit_pct", 0.015) or 0.015),
            float(cfg.get("take_profit_pct_min", 0.010) or 0.010),
            float(cfg.get("take_profit_pct_max", 0.030) or 0.030),
        )

        swing_low = _last_swing_low(lows, int(cfg.get("swing_lookback", 20) or 20))
        max_loss_pct = float(cfg.get("max_loss_pct", 0.02) or 0.02)
        fallback_sl = price * (1.0 - max(0.0, max_loss_pct))
        stop_loss = min(swing_low or fallback_sl, fallback_sl)
        if stop_loss <= 0 or stop_loss >= price:
            stop_loss = fallback_sl

        take_profit = price * (1.0 + take_profit_pct)

        risk_pct = float(cfg.get("risk_pct_per_trade", 0.01) or 0.01)
        risk_amount = max(0.0, capital) * max(0.0, risk_pct)
        stop_distance = max(1e-9, price - stop_loss)
        size = risk_amount / stop_distance if risk_amount > 0 else self.config["min_position_size"]
        size = max(self.config["min_position_size"], min(self.config["max_position_size"], size))

        reasons = [
            f"Recent EMA{ema_fast_period}→EMA{ema_slow_period} cross (within {cross_lookback} candles) + continuation.",
            f"RSI({rsi_period}) >= {rsi_entry_min:.1f} and rising." if require_rsi_rising else f"RSI({rsi_period}) >= {rsi_entry_min:.1f}.",
            "Volume above SMA.",
            "Close above EMAs." if require_close_above_emas else "Close filter disabled.",
            "EMA slow slope up." if require_ema_slope else "Slope filter disabled.",
        ]

        # IMPORTANT: seed trade state via metadata (engine/context should persist it)
        return self._build_signal(
            decision="buy",
            price=price,
            confidence=confidence,
            suggested_size=size,
            reasons=reasons,
            metadata={
                "entry_type": "long",
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "risk_pct_per_trade": risk_pct,
                "ema_fast": float(ema_fast_now),
                "ema_slow": float(ema_slow_now),
                "rsi": float(rsi_now),
                "volume": float(volume_now),
                "volume_sma": float(volume_avg),
                "trigger": "continuation_entry",
                # trade-state seeds
                "entry_price": float(price),
                "bars_in_trade": 0,
                "peak_price": float(price),
                "trailing_stop": None,
            },
        )
