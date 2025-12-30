"""Trading go/no-go gate for *new entries* (simulation only).

This module provides conservative checks to avoid trading when conditions are poor
or evidence is insufficient. It never places real orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    can_enter: bool
    reasons: List[str]
    updated_at: str
    snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "can_enter": self.can_enter,
            "reasons": self.reasons,
            "updated_at": self.updated_at,
            "snapshot": self.snapshot,
        }


def evaluate_trading_gate(
    *,
    now: datetime,
    gate_cfg: Dict[str, Any],
    risk_cfg: Dict[str, Any],
    metrics_today: Dict[str, Any],
    trades_today_count: int,
    volatility_regime: float,
    validation_mode: bool,
    is_training: bool,
    fear_greed_value: Optional[int] = None,
) -> GateDecision:
    """Return a conservative gate decision.

    Notes:
    - The gate is applied only to *new BUY entries*. Exits should always be allowed.
    - Defaults are intentionally conservative.
    """
    enabled = bool(gate_cfg.get("enabled", True))
    if not enabled:
        return GateDecision(
            enabled=False,
            can_enter=True,
            reasons=["Gate disabled."],
            updated_at=now.isoformat() + "Z",
            snapshot={},
        )

    blocking: List[str] = []
    info: List[str] = []

    if is_training:
        blocking.append("Training in progress.")
    if validation_mode:
        blocking.append("Validation mode enabled (no new entries).")

    min_trades_today = int(gate_cfg.get("min_trades_today", 5))
    if trades_today_count < min_trades_today:
        info.append(f"Insufficient trades today for strong stats ({trades_today_count} < {min_trades_today}).")

    pnl = float(metrics_today.get("pnl", 0.0) or 0.0)
    drawdown = float(metrics_today.get("drawdown", 0.0) or 0.0)
    sharpe = float(metrics_today.get("sharpe", 0.0) or 0.0)
    profit_factor = float(metrics_today.get("profit_factor", 0.0) or 0.0)
    expectancy = float(metrics_today.get("expectancy", 0.0) or 0.0)
    win_rate = float(metrics_today.get("win_rate", 0.0) or 0.0)

    max_daily_loss_usdt = float(risk_cfg.get("max_daily_loss_usdt", 0.0) or 0.0)
    if max_daily_loss_usdt > 0 and pnl <= -max_daily_loss_usdt:
        blocking.append(f"Daily loss limit hit (pnl {pnl:.2f} <= -{max_daily_loss_usdt:.2f}).")

    max_drawdown_usdt = float(risk_cfg.get("max_drawdown_usdt", 0.0) or 0.0)
    if max_drawdown_usdt > 0 and drawdown >= max_drawdown_usdt:
        blocking.append(f"Drawdown limit hit ({drawdown:.2f} >= {max_drawdown_usdt:.2f}).")

    # Performance thresholds need enough samples; otherwise they create a deadlock (no trades => no stats).
    try:
        configured = gate_cfg.get("min_trades_for_stats")
        if configured is None:
            min_trades_for_stats = max(10, min_trades_today)
        else:
            configured_i = int(configured)
            min_trades_for_stats = configured_i if configured_i > 0 else max(10, min_trades_today)
    except Exception:
        min_trades_for_stats = max(10, min_trades_today)
    if trades_today_count >= min_trades_for_stats:
        min_sharpe = float(gate_cfg.get("min_sharpe", 0.1))
        if sharpe < min_sharpe:
            blocking.append(f"Sharpe below threshold ({sharpe:.2f} < {min_sharpe:.2f}).")

        min_profit_factor = float(gate_cfg.get("min_profit_factor", 1.05))
        if profit_factor < min_profit_factor:
            blocking.append(f"Profit factor below threshold ({profit_factor:.2f} < {min_profit_factor:.2f}).")

        min_expectancy = float(gate_cfg.get("min_expectancy", 0.0))
        if expectancy < min_expectancy:
            blocking.append(f"Expectancy below threshold ({expectancy:.2f} < {min_expectancy:.2f}).")

        min_win_rate = float(gate_cfg.get("min_win_rate", 0.0))
        if min_win_rate > 0 and win_rate < min_win_rate:
            blocking.append(f"Win rate below threshold ({win_rate:.2f}% < {min_win_rate:.2f}%).")
    else:
        info.append(f"Performance thresholds not enforced until {min_trades_for_stats} trades.")

    min_vol = float(gate_cfg.get("min_volatility_regime", 0.0))
    max_vol = float(gate_cfg.get("max_volatility_regime", 1.0))
    if volatility_regime < min_vol:
        blocking.append(f"Volatility regime too low ({volatility_regime:.4f} < {min_vol:.4f}).")
    if volatility_regime > max_vol:
        blocking.append(f"Volatility regime too high ({volatility_regime:.4f} > {max_vol:.4f}).")

    if fear_greed_value is not None:
        fng_min = gate_cfg.get("fear_greed_min")
        fng_max = gate_cfg.get("fear_greed_max")
        try:
            fng = int(fear_greed_value)
            if fng_min is not None and fng < int(fng_min):
                blocking.append(f"Fear&Greed too low ({fng} < {int(fng_min)}).")
            if fng_max is not None and fng > int(fng_max):
                blocking.append(f"Fear&Greed too high ({fng} > {int(fng_max)}).")
        except Exception:
            info.append("Fear&Greed value unavailable.")

    can_enter = len(blocking) == 0
    reasons = blocking + info
    return GateDecision(
        enabled=True,
        can_enter=can_enter,
        reasons=reasons or ["OK"],
        updated_at=now.isoformat() + "Z",
        snapshot={
            "trades_today": trades_today_count,
            "pnl": pnl,
            "drawdown": drawdown,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "win_rate": win_rate,
            "volatility_regime": float(volatility_regime),
            "fear_greed": fear_greed_value,
        },
    )
