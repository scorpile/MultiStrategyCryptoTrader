"""EMA cross + RSI confirmation + volume filter (confirmed scalping).

Implements the rules from EMACrossRSIVolumeScalping.md with fixed TP/SL,
optional trailing stop, and risk-based sizing. Spot-only by default; short
signals can be enabled for future extensions.
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
    """EMA(5/20) + RSI(7) + volumen + trailing stop dinámico + confianza ajustable."""


    description = (
        "EMA(5/20) cross + RSI(7)>50 rising + volume>SMA20 + close above EMAs + EMA20 slope. "
        "Fixed TP/SL with dynamic trailing stop and confidence scaling."
    )


    def name(self) -> str:
        return "ema_rsi_volume_confirmed"


    def config_schema(self) -> Dict[str, Any]:
        return {
            "signal_interval": "3m",
            "ema_fast_period": 5,
            "ema_slow_period": 20,
            "rsi_period": 7,
            "rsi_confirm_threshold": 50.0,
            "rsi_exit_threshold": 50.0,
            "require_rsi_rising": True,
            "volume_sma_period": 20,
            "require_volume_above_sma": True,
            "require_close_above_emas": True,
            "require_ema_slope": True,
            "max_spread_pct": 0.002,
            "swing_lookback": 20,
            "max_loss_pct": 0.01,
            "take_profit_pct": 0.02,
            "take_profit_pct_min": 0.015,
            "take_profit_pct_max": 0.03,
            "risk_pct_per_trade": 0.01,
            "trailing_stop_enabled": True,
            "trailing_stop_pct": 0.0125,
            "trailing_stop_activation_pct": 0.01,
            "trailing_stop_min_step_pct": 0.001,
            "allow_shorts": False,
        }


    def _compute_confidence(self, rsi: float, rsi_prev: float, vol_now: float, vol_avg: float) -> float:
        confidence = 0.7
        if rsi > 60 and rsi > rsi_prev:
            confidence += 0.1
        if vol_now > 1.5 * vol_avg:
            confidence += 0.1
        return min(confidence, 0.95)


    def _apply_trailing_stop(self, context: Dict[str, Any], price: float, cfg: Dict[str, Any]) -> Optional[float]:
        entry_price = context.get("entry_price")
        if not entry_price or price <= 0:
            return None
        activation_pct = cfg.get("trailing_stop_activation_pct", 0.01)
        trail_pct = cfg.get("trailing_stop_pct", 0.0125)
        min_step_pct = cfg.get("trailing_stop_min_step_pct", 0.001)
        gain = (price - entry_price) / entry_price
        if gain < activation_pct:
            return None


        new_trail = price * (1 - trail_pct)
        old_trail = context.get("trailing_stop")
        if old_trail:
            step = abs(new_trail - old_trail) / old_trail
            if step < min_step_pct:
                return old_trail
        return new_trail

    def generate_signal(
        self,
        indicators: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        highs = indicators.get("highs") or []
        lows = indicators.get("lows") or []
        volumes = indicators.get("volumes") or []
        if not closes or len(closes) < 3:
            return self._build_signal(
                decision="hold",
                price=float(closes[-1]) if closes else 0.0,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Insufficient data for EMA/RSI/volume confirmation."],
            )

        ema_fast_period = int(self.config.get("ema_fast_period", 5) or 5)
        ema_slow_period = int(self.config.get("ema_slow_period", 20) or 20)
        rsi_period = int(self.config.get("rsi_period", 7) or 7)
        vol_sma_period = int(self.config.get("volume_sma_period", 20) or 20)

        min_len = max(ema_slow_period + 2, rsi_period + 2, vol_sma_period + 1, 3)
        if len(closes) < min_len:
            return self._build_signal(
                decision="hold",
                price=float(closes[-1]),
                confidence=0.0,
                suggested_size=0.0,
                reasons=[f"Need >= {min_len} candles for confirmed signals."],
            )

        price = float(closes[-1])

        ema_fast_series = indicators.get("ema_fast")
        ema_slow_series = indicators.get("ema_slow")
        ema_fast = ema_fast_series if isinstance(ema_fast_series, list) and len(ema_fast_series) == len(closes) else _ema_series(closes, ema_fast_period)
        ema_slow = ema_slow_series if isinstance(ema_slow_series, list) and len(ema_slow_series) == len(closes) else _ema_series(closes, ema_slow_period)

        rsi_series = indicators.get("rsi") or []
        volume_sma = indicators.get("volume_sma") or []

        if len(ema_fast) < 2 or len(ema_slow) < 2 or len(rsi_series) < 2 or len(volume_sma) < 1:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Indicators not ready for confirmed signals."],
            )

        ema_fast_prev = float(ema_fast[-2])
        ema_fast_now = float(ema_fast[-1])
        ema_slow_prev = float(ema_slow[-2])
        ema_slow_now = float(ema_slow[-1])
        rsi_prev = float(rsi_series[-2])
        rsi_now = float(rsi_series[-1])
        volume_now = float(volumes[-1]) if volumes else 0.0
        volume_sma_now = float(volume_sma[-1]) if volume_sma else 0.0

        cfg = self.config
        rsi_threshold = float(cfg.get("rsi_confirm_threshold", 50.0) or 50.0)
        require_rsi_rising = bool(cfg.get("require_rsi_rising", True))
        require_volume_above_sma = bool(cfg.get("require_volume_above_sma", True))
        require_close_above_emas = bool(cfg.get("require_close_above_emas", True))
        require_ema_slope = bool(cfg.get("require_ema_slope", True))
        allow_shorts = bool(cfg.get("allow_shorts", False))

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

        take_profit_pct = _clamp(
            float(cfg.get("take_profit_pct", 0.02) or 0.02),
            float(cfg.get("take_profit_pct_min", 0.015) or 0.015),
            float(cfg.get("take_profit_pct_max", 0.03) or 0.03),
        )
        cfg["take_profit_pct"] = take_profit_pct

        trail_pct = float(cfg.get("trailing_stop_pct", 0.0) or 0.0)
        if trail_pct > 0:
            trail_pct = _clamp(trail_pct, 0.01, 0.015)
            cfg["trailing_stop_pct"] = trail_pct

        # Exit logic for existing positions (always allow exits).
        if position_qty > 0:
            exit_threshold = float(cfg.get("rsi_exit_threshold", 50.0) or 50.0)
            exit_on_turn = (rsi_now < exit_threshold) and (rsi_now < rsi_prev)
            if exit_on_turn:
                confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_sma_now)
                min_conf = float(cfg.get("min_confidence", 0.6))
                if confidence < min_conf:
                    return self._build_signal(
                        decision="hold",
                        price=price,
                        confidence=confidence,
                        suggested_size=0.0,
                        reasons=[f"Signal confidence {confidence:.2f} < min {min_conf:.2f}."],
                    )                     
                return self._build_signal(
                    decision="sell",
                    price=price,
                    confidence=confidence,
                    suggested_size=abs(position_qty),
                    reasons=[f"RSI turning down below {exit_threshold:.1f} ({rsi_now:.2f} < {rsi_prev:.2f})."],
                    metadata={"trigger": "rsi_turn_down"},
                )

            if cfg.get("trailing_stop_enabled"):
                new_trail = self._apply_trailing_stop(ctx, price, cfg)
                old_trail = ctx.get("trailing_stop")
                if new_trail and (not old_trail or new_trail > old_trail):
                    confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_sma_now)
                    ctx["trailing_stop"] = new_trail            
                    min_conf = float(cfg.get("min_confidence", 0.6))
                    if confidence < min_conf:
                        return self._build_signal(
                            decision="hold",
                            price=price,
                            confidence=confidence,
                            suggested_size=0.0,
                            reasons=[f"Signal confidence {confidence:.2f} < min {min_conf:.2f}."],
                        )                    
                    return self._build_signal(
                        decision="sell",
                        price=price,
                        confidence=confidence,
                        suggested_size=abs(position_qty),
                        reasons=["Trailing stop updated (new > old)."],
                        metadata={"trigger": "trailing_stop_move"},
                    )

            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Position open; waiting for exit/SL/TP/trailing."],
            )

        if position_qty < 0:
            exit_threshold = float(cfg.get("rsi_exit_threshold", 50.0) or 50.0)
            exit_on_turn = (rsi_now > exit_threshold) and (rsi_now > rsi_prev)
            if exit_on_turn:
                confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_sma_now)
                min_conf = float(cfg.get("min_confidence", 0.6))
                if confidence < min_conf:
                    return self._build_signal(
                        decision="hold",
                        price=price,
                        confidence=confidence,
                        suggested_size=0.0,
                        reasons=[f"Signal confidence {confidence:.2f} < min {min_conf:.2f}."],
                    )                     
                return self._build_signal(
                    decision="buy",
                    price=price,
                    confidence=confidence,
                    suggested_size=abs(position_qty),
                    reasons=[f"RSI turning up above {exit_threshold:.1f} ({rsi_now:.2f} > {rsi_prev:.2f})."],
                    metadata={"trigger": "rsi_turn_up"},
                )

            if cfg.get("trailing_stop_enabled"):
                new_trail = self._apply_trailing_stop(ctx, price, cfg)
                old_trail = ctx.get("trailing_stop")
                if new_trail and (not old_trail or new_trail < old_trail):
                    confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_sma_now)
                    ctx["trailing_stop"] = new_trail
                    min_conf = float(cfg.get("min_confidence", 0.6))
                    if confidence < min_conf:
                        return self._build_signal(
                            decision="hold",
                            price=price,
                            confidence=confidence,
                            suggested_size=0.0,
                            reasons=[f"Signal confidence {confidence:.2f} < min {min_conf:.2f}."],
                        )                         
                    return self._build_signal(
                        decision="buy",
                        price=price,
                        confidence=confidence,
                        suggested_size=abs(position_qty),
                        reasons=["Trailing stop updated (new < old)."],
                        metadata={"trigger": "trailing_stop_move"},
                    )

            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Short position open; waiting for exit/SL/TP/trailing."],
            )

        # Entry filters (flat)
        if spread_pct is not None and spread_pct > max_spread_pct:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=[f"Spread {spread_pct:.4f} > max {max_spread_pct:.4f}."],
            )

        if require_volume_above_sma and (volume_sma_now <= 0 or volume_now <= volume_sma_now):
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Volume below SMA(20)."],
            )

        ema_cross_up = (ema_fast_prev < ema_slow_prev) and (ema_fast_now >= ema_slow_now)
        ema_cross_down = (ema_fast_prev > ema_slow_prev) and (ema_fast_now <= ema_slow_now)

        rsi_ok_long = (rsi_now > rsi_threshold) and (rsi_now > rsi_prev if require_rsi_rising else True)
        rsi_ok_short = (rsi_now < rsi_threshold) and (rsi_now < rsi_prev if require_rsi_rising else True)

        close_above = price > ema_fast_now and price > ema_slow_now
        close_below = price < ema_fast_now and price < ema_slow_now

        ema_slope_up = ema_slow_now > ema_slow_prev
        ema_slope_down = ema_slow_now < ema_slow_prev

        long_ok = ema_cross_up and rsi_ok_long
        short_ok = ema_cross_down and rsi_ok_short

        if require_close_above_emas:
            long_ok = long_ok and close_above
            short_ok = short_ok and close_below
        if require_ema_slope:
            long_ok = long_ok and ema_slope_up
            short_ok = short_ok and ema_slope_down

        if long_ok:
            swing_low = _last_swing_low(lows, int(cfg.get("swing_lookback", 20) or 20))
            max_loss_pct = float(cfg.get("max_loss_pct", 0.01) or 0.01)
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
                "EMA5 crossed above EMA20.",
                f"RSI rising above {rsi_threshold:.1f}.",
                "Volume above SMA20.",
                "Close above EMA5/EMA20.",
                "EMA20 slope positive.",
            ]
            confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_sma_now)
            min_conf = float(cfg.get("min_confidence", 0.6))
            if confidence < min_conf:
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=confidence,
                    suggested_size=0.0,
                    reasons=[f"Signal confidence {confidence:.2f} < min {min_conf:.2f}."],
                )
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
                    "swing_low": float(swing_low),
                    "ema_fast": ema_fast_now,
                    "ema_slow": ema_slow_now,
                    "rsi": rsi_now,
                    "volume": volume_now,
                    "volume_sma": volume_sma_now,
                },
            )

        if short_ok and allow_shorts:
            swing_high = _last_swing_high(highs, int(cfg.get("swing_lookback", 20) or 20))
            max_loss_pct = float(cfg.get("max_loss_pct", 0.01) or 0.01)
            fallback_sl = price * (1.0 + max(0.0, max_loss_pct))
            stop_loss = max(swing_high or fallback_sl, fallback_sl)
            if stop_loss <= 0 or stop_loss <= price:
                stop_loss = fallback_sl
            take_profit = price * (1.0 - take_profit_pct)
            risk_pct = float(cfg.get("risk_pct_per_trade", 0.01) or 0.01)
            risk_amount = max(0.0, capital) * max(0.0, risk_pct)
            stop_distance = max(1e-9, stop_loss - price)
            size = risk_amount / stop_distance if risk_amount > 0 else self.config["min_position_size"]
            size = max(self.config["min_position_size"], min(self.config["max_position_size"], size))
            reasons = [
                "EMA5 crossed below EMA20.",
                f"RSI falling below {rsi_threshold:.1f}.",
                "Volume above SMA20.",
                "Close below EMA5/EMA20.",
                "EMA20 slope negative.",
            ]
            confidence = self._compute_confidence(rsi_now, rsi_prev, volume_now, volume_sma_now)
            min_conf = float(cfg.get("min_confidence", 0.6))
            if confidence < min_conf:
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=confidence,
                    suggested_size=0.0,
                    reasons=[f"Signal confidence {confidence:.2f} < min {min_conf:.2f}."],
                )            
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=0.78,
                suggested_size=size,
                reasons=reasons,
                metadata={
                    "entry_type": "short",
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                    "risk_pct_per_trade": risk_pct,
                    "swing_high": float(swing_high),
                    "ema_fast": ema_fast_now,
                    "ema_slow": ema_slow_now,
                    "rsi": rsi_now,
                    "volume": volume_now,
                    "volume_sma": volume_sma_now,
                },
            )

        if short_ok and not allow_shorts:
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Short setup detected but shorts are disabled (spot-only)."],
            )

        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.0,
            suggested_size=0.0,
            reasons=["No confirmed EMA/RSI/volume setup."],
        )
