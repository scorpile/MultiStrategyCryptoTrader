"""Simple online auto-tuner for strategy parameters (simulation only).

Goal: adjust a handful of knobs based on recent closed-trade performance so the bot
can evolve without manual intervention. This is intentionally conservative and
bounded; it should never place real orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AutoTuneResult:
    changed: bool
    strategy: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    metrics: Dict[str, Any]
    reasons: List[str]
    changed_keys: List[str]
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "strategy": self.strategy,
            "before": self.before,
            "after": self.after,
            "metrics": self.metrics,
            "reasons": self.reasons,
            "changed_keys": self.changed_keys,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class Tunable:
    key: str
    lo: float
    hi: float
    step: float
    kind: str = "float"  # float|int

    def clamp(self, value: float) -> float:
        return _clamp(value, self.lo, self.hi)

    def apply(self, current: Any, delta_steps: float) -> Any:
        try:
            cur = float(current)
        except Exception:
            cur = 0.0
        updated = self.clamp(cur + (self.step * float(delta_steps)))
        if self.kind == "int":
            return int(round(updated))
        return round(float(updated), 6)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _float(cfg: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


def _int(cfg: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def compute_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple metrics from a list of closed trades (expects `pnl`)."""
    pnls = [float(t.get("pnl", 0.0) or 0.0) for t in trades]
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / max(len(pnls), 1)) * 100.0
    avg_win = sum(wins) / max(len(wins), 1)
    avg_loss = abs(sum(losses) / max(len(losses), 1))
    profit_factor = (sum(wins) / max(abs(sum(losses)), 1e-9)) if losses else float("inf")
    expectancy = (win_rate / 100.0 * avg_win) - ((100.0 - win_rate) / 100.0 * avg_loss)
    return {
        "trades": len(pnls),
        "pnl": total,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "expectancy": expectancy,
    }


def default_tunables_for_strategy(strategy_name: str) -> List[Tunable]:
    """Return a conservative list of tunables for known strategies."""
    base = [
        Tunable("risk_pct_per_trade", lo=0.001, hi=0.03, step=0.001, kind="float"),
        Tunable("stop_atr_multiplier", lo=0.8, hi=3.5, step=0.10, kind="float"),
        Tunable("profit_target_multiplier", lo=0.8, hi=4.0, step=0.10, kind="float"),
        Tunable("momentum_threshold", lo=0.01, hi=1.0, step=0.01, kind="float"),
        Tunable("rsi_buy", lo=5, hi=60, step=1, kind="int"),
        Tunable("rsi_sell", lo=40, hi=95, step=1, kind="int"),
        Tunable("stoch_rsi_buy", lo=0.05, hi=0.60, step=0.02, kind="float"),
        Tunable("stoch_rsi_sell", lo=0.40, hi=0.95, step=0.02, kind="float"),
    ]
    if strategy_name == "scalping_aggressive":
        base += [
            Tunable("exploration_random_entry_prob", lo=0.05, hi=0.80, step=0.05, kind="float"),
            Tunable("exploration_random_exit_prob", lo=0.05, hi=0.80, step=0.05, kind="float"),
            Tunable("exploration_risk_multiplier", lo=1.0, hi=5.0, step=0.25, kind="float"),
            Tunable("exploration_rsi_relax", lo=0.0, hi=25.0, step=1.0, kind="float"),
            Tunable("exploration_stoch_relax", lo=0.0, hi=0.60, step=0.05, kind="float"),
            Tunable("exploration_ema_slack", lo=0.0, hi=0.20, step=0.01, kind="float"),
            Tunable("exploration_momentum_slack", lo=0.0, hi=2.0, step=0.10, kind="float"),
        ]
    return base


def _enforce_invariants(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fix up obvious invalid combinations after tuning."""
    out = dict(cfg)
    try:
        rsi_buy = float(out.get("rsi_buy", 40))
        rsi_sell = float(out.get("rsi_sell", 60))
        if rsi_sell <= rsi_buy:
            # Keep a sensible gap.
            out["rsi_sell"] = int(round(_clamp(rsi_buy + 10.0, 40.0, 95.0)))
            out["rsi_buy"] = int(round(_clamp(rsi_buy, 5.0, 60.0)))
    except Exception:
        pass
    try:
        sb = float(out.get("stoch_rsi_buy", 0.2))
        ss = float(out.get("stoch_rsi_sell", 0.8))
        if ss <= sb:
            out["stoch_rsi_sell"] = round(_clamp(sb + 0.2, 0.4, 0.95), 4)
            out["stoch_rsi_buy"] = round(_clamp(sb, 0.05, 0.6), 4)
    except Exception:
        pass
    try:
        mt = float(out.get("momentum_threshold", 0.05))
        # Avoid division-by-zero in strategies; keep it small but positive.
        out["momentum_threshold"] = round(_clamp(mt, 0.01, 1.0), 6)
    except Exception:
        out["momentum_threshold"] = 0.05
    return out


def auto_tune_strategy(
    *,
    now: datetime,
    strategy_name: str,
    current_config: Dict[str, Any],
    recent_closed_trades: List[Dict[str, Any]],
    auto_cfg: Dict[str, Any],
    exploration_active: bool,
    trades_per_hour: Optional[float] = None,
    decision_stats: Optional[Dict[str, Any]] = None,
) -> AutoTuneResult:
    """Return an updated config (bounded) plus a structured explanation."""
    metrics = compute_trade_metrics(recent_closed_trades)
    reasons: List[str] = []
    changed_keys: List[str] = []

    min_closed_trades = _int(auto_cfg, "min_closed_trades", 10)
    if metrics["trades"] < min_closed_trades:
        # Not enough signal: keep exploration adventurous to gather data.
        target_prob = _float(auto_cfg, "bootstrap_entry_prob", 0.25)
        if exploration_active:
            before = dict(current_config)
            after = dict(current_config)
            old = float(after.get("exploration_random_entry_prob", 0.15))
            new = _clamp(old, 0.05, 0.70)
            if new < target_prob:
                after["exploration_random_entry_prob"] = round(_clamp(target_prob, 0.05, 0.70), 4)
                reasons.append(f"Bootstrap: raise random entry prob to {after['exploration_random_entry_prob']}.")
                changed_keys.append("exploration_random_entry_prob")
            changed = after != before
            return AutoTuneResult(
                changed=changed,
                strategy=strategy_name,
                before=before,
                after=after,
                metrics=metrics,
                reasons=reasons or [f"Bootstrap: waiting for >= {min_closed_trades} closed trades."],
                changed_keys=sorted(set(changed_keys)),
                updated_at=now.isoformat() + "Z",
            )

        return AutoTuneResult(
            changed=False,
            strategy=strategy_name,
            before=dict(current_config),
            after=dict(current_config),
            metrics=metrics,
            reasons=[f"Waiting for >= {min_closed_trades} closed trades."],
            changed_keys=[],
            updated_at=now.isoformat() + "Z",
        )

    before = dict(current_config)
    after = dict(current_config)

    # Thresholds
    good_pf = _float(auto_cfg, "good_profit_factor", 1.2)
    bad_pf = _float(auto_cfg, "bad_profit_factor", 0.95)
    good_wr = _float(auto_cfg, "good_win_rate", 55.0)
    bad_wr = _float(auto_cfg, "bad_win_rate", 40.0)
    good_exp = _float(auto_cfg, "good_expectancy", 0.0)

    is_good = metrics["profit_factor"] >= good_pf and metrics["win_rate"] >= good_wr and metrics["expectancy"] >= good_exp
    is_bad = metrics["profit_factor"] <= bad_pf or metrics["win_rate"] <= bad_wr or metrics["expectancy"] < 0.0

    # Trade-frequency shaping ("auto adjust everything" includes: make it trade when too quiet).
    target_tph = _float(auto_cfg, "target_trades_per_hour", 10.0)
    tph = float(trades_per_hour) if trades_per_hour is not None else None
    too_quiet = tph is not None and tph < (target_tph * 0.6)
    too_noisy = tph is not None and tph > (target_tph * 1.6)

    tunables = default_tunables_for_strategy(strategy_name)
    tunable_map = {t.key: t for t in tunables}

    def _set(key: str, value: Any) -> None:
        nonlocal after, changed_keys
        if after.get(key) != value:
            after[key] = value
            changed_keys.append(key)

    def _step(key: str, steps: float) -> None:
        t = tunable_map.get(key)
        if not t:
            return
        cur = after.get(key, before.get(key))
        _set(key, t.apply(cur, steps))

    if is_bad:
        # Reduce risk and tighten entry randomness; widen stops; raise take profit a bit less.
        _step("risk_pct_per_trade", -2)
        _step("stop_atr_multiplier", +1)
        _step("profit_target_multiplier", -1)
        _step("momentum_threshold", +1)  # require a bit more momentum
        _step("exploration_random_entry_prob", -1)
        _step("exploration_random_exit_prob", +1)
        _step("exploration_risk_multiplier", -1)
        _step("exploration_rsi_relax", -1)
        _step("exploration_stoch_relax", -1)
        reasons.append("Performance weak: reduce risk/noise, widen stops, demand slightly more momentum.")
    elif is_good:
        # Lean in: modestly increase risk/activity, tighten stop a bit.
        _step("risk_pct_per_trade", +1)
        _step("stop_atr_multiplier", -1)
        _step("profit_target_multiplier", +1)
        _step("momentum_threshold", -1)
        _step("exploration_random_entry_prob", +1)
        _step("exploration_random_exit_prob", -1)
        _step("exploration_risk_multiplier", +1)
        _step("exploration_rsi_relax", +1)
        _step("exploration_stoch_relax", +1)
        reasons.append("Performance strong: increase activity/risk slightly; tighten stops a bit.")
    else:
        reasons.append("Performance mixed: light adjustments driven by trade frequency.")

    if too_quiet:
        # Force more trades: relax thresholds.
        _step("rsi_buy", +2)
        _step("rsi_sell", -2)
        _step("stoch_rsi_buy", +1)
        _step("stoch_rsi_sell", -1)
        _step("momentum_threshold", -1)
        _step("exploration_random_entry_prob", +1)
        reasons.append("Too few trades: relax thresholds and increase entry probability.")
    elif too_noisy:
        # Reduce churn: tighten thresholds.
        _step("rsi_buy", -2)
        _step("rsi_sell", +2)
        _step("stoch_rsi_buy", -1)
        _step("stoch_rsi_sell", +1)
        _step("momentum_threshold", +1)
        _step("exploration_random_entry_prob", -1)
        reasons.append("Too many trades: tighten thresholds and reduce entry probability.")

    # Use blocked-signal feedback to shape exploration behavior.
    if decision_stats:
        try:
            blocked = decision_stats.get("blocked_by_reason") or {}
            no_pos = int(blocked.get("no_position", 0) or 0)
            throttle = int(blocked.get("throttle", 0) or 0)
            gate_blk = int(blocked.get("gate_blocked", 0) or 0)
            risk_pause = int(blocked.get("risk_pause", 0) or 0)
            total = int(decision_stats.get("total", 0) or 0)
            if total > 0:
                if no_pos / total >= 0.15:
                    _step("exploration_random_exit_prob", -1)
                    reasons.append("Many SELL signals while flat: reduce random exits.")
                if throttle / total >= 0.15 and tph is not None and tph < target_tph:
                    _step("exploration_random_entry_prob", -1)
                    reasons.append("Throttle blocks frequent: reduce entry randomness slightly (let controls adjust).")
                if gate_blk / total >= 0.15 and exploration_active:
                    _step("exploration_rsi_relax", +1)
                    _step("exploration_stoch_relax", +1)
                    _step("rsi_buy", +1)
                    _step("stoch_rsi_buy", +1)
                    _step("momentum_threshold", -1)
                    reasons.append("Gate blocks frequent during exploration: relax exploration thresholds slightly.")
                if risk_pause / total >= 0.05:
                    _step("risk_pct_per_trade", -2)
                    _step("exploration_risk_multiplier", -1)
                    reasons.append("Risk pause blocks seen: reduce risk per trade to recover more safely.")
        except Exception:
            pass

    after = _enforce_invariants(after)
    changed_keys = sorted(set(changed_keys))

    return AutoTuneResult(
        changed=after != before,
        strategy=strategy_name,
        before=before,
        after=after,
        metrics=metrics,
        reasons=reasons,
        changed_keys=changed_keys,
        updated_at=now.isoformat() + "Z",
    )
