"""EMA cross + RSI confirmation + volume filter with ML/RL gating.

Implements EMACrossRSIVolumeScalping rules plus ML/RL filters for entries,
while keeping exits deterministic (never blocked by ML/RL).
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


class EmaRsiVolumeConfirmedMLRLStrategy(BaseStrategy):
    """Confirmed EMA cross + RSI + volume scalping with ML/RL gating."""

    description = (
        "EMA(5/20) cross + RSI(7)>50 rising + volume>SMA20 + close above EMAs + EMA20 slope. "
        "ML gate/boost + RL risk/confidence tuning (spot long-only)."
    )

    def name(self) -> str:
        return "ema_rsi_volume_confirmed_mlrl"

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
            "trailing_stop_min_step_pct": 0.0,
            "allow_shorts": False,
            # ML gating/boost
            "ml_filter_enabled": True,
            "ml_require_model": False,
            "ml_min_probability_up": 0.55,
            "ml_min_probability_down": 0.55,
            "ml_min_confidence": 0.15,
            "ml_confidence_boost": 0.15,
            "ml_risk_slope": 0.6,
            "ml_risk_multiplier_min": 0.6,
            "ml_risk_multiplier_max": 1.4,
            # RL tuning
            "rl_use_weights": True,
            "rl_confidence_boost": 0.2,
            "rl_score_scale": 0.002,
            "rl_risk_multiplier_min": 0.6,
            "rl_risk_multiplier_max": 1.4,
        }

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

        ml_ctx = ctx.get("ml") or {}
        model_name = str(ml_ctx.get("model_name", "") or "")
        ml_available = bool(ml_ctx) and model_name not in {"", "dummy_model"}
        ml_prob_up = float(ml_ctx.get("probability_up", 0.5) or 0.5)
        ml_confidence = float(ml_ctx.get("confidence", 0.0) or 0.0)
        ml_filter_enabled = bool(cfg.get("ml_filter_enabled", True))
        ml_require_model = bool(cfg.get("ml_require_model", False))
        ml_min_conf = float(cfg.get("ml_min_confidence", 0.15) or 0.15)
        ml_min_up = float(cfg.get("ml_min_probability_up", 0.55) or 0.55)
        ml_min_down = float(cfg.get("ml_min_probability_down", 0.55) or 0.55)
        ml_conf_boost = float(cfg.get("ml_confidence_boost", 0.15) or 0.15)
        ml_risk_slope = float(cfg.get("ml_risk_slope", 0.6) or 0.6)
        ml_risk_min = float(cfg.get("ml_risk_multiplier_min", 0.6) or 0.6)
        ml_risk_max = float(cfg.get("ml_risk_multiplier_max", 1.4) or 1.4)

        rl_policy = ctx.get("rl") or {}
        rl_score = float(rl_policy.get("score", 0.0) or 0.0)
        rl_use_weights = bool(cfg.get("rl_use_weights", True))
        rl_conf_boost = float(cfg.get("rl_confidence_boost", 0.2) or 0.2)
        rl_score_scale = float(cfg.get("rl_score_scale", 0.002) or 0.002)
        rl_risk_min = float(cfg.get("rl_risk_multiplier_min", 0.6) or 0.6)
        rl_risk_max = float(cfg.get("rl_risk_multiplier_max", 1.4) or 1.4)

        # Exits for open positions are never blocked by ML/RL.
        if position_qty > 0:
            exit_threshold = float(cfg.get("rsi_exit_threshold", 50.0) or 50.0)
            exit_on_turn = (rsi_now < exit_threshold) and (rsi_now < rsi_prev)
            if exit_on_turn:
                return self._build_signal(
                    decision="sell",
                    price=price,
                    confidence=0.9,
                    suggested_size=abs(position_qty),
                    reasons=[f"RSI turning down below {exit_threshold:.1f} ({rsi_now:.2f} < {rsi_prev:.2f})."],
                    metadata={"trigger": "rsi_turn_down"},
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
                return self._build_signal(
                    decision="buy",
                    price=price,
                    confidence=0.9,
                    suggested_size=abs(position_qty),
                    reasons=[f"RSI turning up above {exit_threshold:.1f} ({rsi_now:.2f} > {rsi_prev:.2f})."],
                    metadata={"trigger": "rsi_turn_up"},
                )
            return self._build_signal(
                decision="hold",
                price=price,
                confidence=0.0,
                suggested_size=0.0,
                reasons=["Short position open; waiting for exit/SL/TP/trailing."],
            )

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

        if ml_filter_enabled:
            if ml_require_model and not ml_available:
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=0.0,
                    suggested_size=0.0,
                    reasons=["ML model required but not available."],
                )
            if ml_available and ml_confidence < ml_min_conf:
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=0.0,
                    suggested_size=0.0,
                    reasons=[f"ML confidence {ml_confidence:.2f} < {ml_min_conf:.2f}."],
                )

        def _ml_gate_long() -> bool:
            if not ml_filter_enabled or not ml_available:
                return True
            return ml_prob_up >= ml_min_up

        def _ml_gate_short() -> bool:
            if not ml_filter_enabled or not ml_available:
                return True
            return (1.0 - ml_prob_up) >= ml_min_down

        base_confidence = 0.72
        confidence = base_confidence
        if ml_available:
            confidence = min(1.0, confidence + ml_confidence * ml_conf_boost)

        if rl_use_weights and isinstance(rl_policy, dict):
            rl_weights = rl_policy.get("weights")
            if isinstance(rl_weights, dict) and rl_weights:
                base = 1 / 8
                focus = (
                    float(rl_weights.get("trend", base))
                    + float(rl_weights.get("rsi", base))
                    + float(rl_weights.get("volume", base))
                    + float(rl_weights.get("ml", base))
                )
                bias = _clamp(focus - 4 * base, -0.3, 0.3)
                confidence = _clamp(confidence + bias * rl_conf_boost, 0.0, 1.0)

        ml_risk_mult = 1.0
        if ml_available:
            delta = (ml_prob_up - 0.5) * 2.0
            ml_risk_mult = 1.0 + delta * ml_risk_slope
        ml_risk_mult = _clamp(ml_risk_mult, ml_risk_min, ml_risk_max)

        rl_risk_mult = 1.0
        try:
            explicit = float(rl_policy.get("risk_multiplier", 0.0) or 0.0)
        except Exception:
            explicit = 0.0
        if explicit > 0:
            rl_risk_mult = explicit
        else:
            rl_risk_mult = 1.0 + _clamp(rl_score * rl_score_scale, -0.5, 0.5)
        rl_risk_mult = _clamp(rl_risk_mult, rl_risk_min, rl_risk_max)

        if long_ok and _ml_gate_long():
            swing_low = _last_swing_low(lows, int(cfg.get("swing_lookback", 20) or 20))
            max_loss_pct = float(cfg.get("max_loss_pct", 0.01) or 0.01)
            fallback_sl = price * (1.0 - max(0.0, max_loss_pct))
            stop_loss = min(swing_low or fallback_sl, fallback_sl)
            if stop_loss <= 0 or stop_loss >= price:
                stop_loss = fallback_sl
            take_profit = price * (1.0 + take_profit_pct)
            risk_pct = float(cfg.get("risk_pct_per_trade", 0.01) or 0.01)
            risk_pct = _clamp(risk_pct * ml_risk_mult * rl_risk_mult, 0.0001, 1.0)
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
            if ml_available:
                reasons.append(f"ML prob_up {ml_prob_up:.2f} conf {ml_confidence:.2f}.")
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
                    "ml": {"probability_up": ml_prob_up, "confidence": ml_confidence, "model": model_name},
                    "rl": {"score": rl_score},
                    "swing_low": float(swing_low),
                    "ema_fast": ema_fast_now,
                    "ema_slow": ema_slow_now,
                    "rsi": rsi_now,
                    "volume": volume_now,
                    "volume_sma": volume_sma_now,
                },
            )

        if short_ok and allow_shorts and _ml_gate_short():
            swing_high = _last_swing_high(highs, int(cfg.get("swing_lookback", 20) or 20))
            max_loss_pct = float(cfg.get("max_loss_pct", 0.01) or 0.01)
            fallback_sl = price * (1.0 + max(0.0, max_loss_pct))
            stop_loss = max(swing_high or fallback_sl, fallback_sl)
            if stop_loss <= 0 or stop_loss <= price:
                stop_loss = fallback_sl
            take_profit = price * (1.0 - take_profit_pct)
            risk_pct = float(cfg.get("risk_pct_per_trade", 0.01) or 0.01)
            risk_pct = _clamp(risk_pct * ml_risk_mult * rl_risk_mult, 0.0001, 1.0)
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
            if ml_available:
                reasons.append(f"ML prob_down {1.0 - ml_prob_up:.2f} conf {ml_confidence:.2f}.")
            return self._build_signal(
                decision="sell",
                price=price,
                confidence=confidence,
                suggested_size=size,
                reasons=reasons,
                metadata={
                    "entry_type": "short",
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                    "risk_pct_per_trade": risk_pct,
                    "ml": {"probability_up": ml_prob_up, "confidence": ml_confidence, "model": model_name},
                    "rl": {"score": rl_score},
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

        if ml_filter_enabled and ml_available:
            if long_ok and not _ml_gate_long():
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=0.0,
                    suggested_size=0.0,
                    reasons=[f"ML prob_up {ml_prob_up:.2f} < {ml_min_up:.2f}."],
                )
            if short_ok and allow_shorts and not _ml_gate_short():
                return self._build_signal(
                    decision="hold",
                    price=price,
                    confidence=0.0,
                    suggested_size=0.0,
                    reasons=[f"ML prob_down {1.0 - ml_prob_up:.2f} < {ml_min_down:.2f}."],
                )

        return self._build_signal(
            decision="hold",
            price=price,
            confidence=0.0,
            suggested_size=0.0,
            reasons=["No confirmed EMA/RSI/volume setup."],
        )
