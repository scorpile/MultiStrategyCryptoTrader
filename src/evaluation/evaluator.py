"""Evaluator helpers for daily performance analysis."""

from __future__ import annotations

import math
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Sequence


class Evaluator:
    """Aggregates daily performance metrics and probability estimates."""

    def __init__(self, state_manager: Any) -> None:
        """Store state manager reference for persisting metrics."""
        self.state_manager = state_manager

    def compile_daily_metrics(self, trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a dictionary with PnL, win rate, drawdown, Sharpe, profit factor, expectancy, etc."""
        # Metrics are computed on *closed* trades (SELL) to avoid buy-side noise.
        closed_trades = [t for t in trades if str(t.get("side", "")).upper() == "SELL"]
        pnl_values = [t.get("pnl", 0.0) for t in closed_trades]
        closed_pnls = [float(p or 0.0) for p in pnl_values if p is not None]
        total_pnl = sum(closed_pnls)
        wins = sum(1 for p in closed_pnls if p > 0)
        losses = sum(1 for p in closed_pnls if p < 0)
        win_rate = (wins / max(len(closed_pnls), 1)) * 100
        max_drawdown = self._calculate_drawdown(closed_pnls)
        sharpe = self._calculate_sharpe_ratio(closed_pnls)
        
        # Additional scalping metrics
        avg_win = sum(p for p in closed_pnls if p > 0) / max(wins, 1)
        avg_loss = abs(sum(p for p in closed_pnls if p < 0) / max(losses, 1))
        profit_factor = (wins * avg_win) / max(losses * avg_loss, 1)
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)
        
        metrics = {
            "pnl": total_pnl,
            "trades_count": len(closed_trades),
            "win_rate": win_rate,
            "drawdown": max_drawdown,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }
        return metrics

    def calculate_probability_of_profit(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the ≥30% profit probability flag using heuristics/ML with scalping metrics."""
        pnl = metrics.get("pnl", 0.0)
        win_rate = metrics.get("win_rate", 0.0)
        sharpe = metrics.get("sharpe", 0.0)
        profit_factor = metrics.get("profit_factor", 1.0)
        expectancy = metrics.get("expectancy", 0.0)
        
        normalized_pnl = min(max(pnl / 1000, 0.0), 1.0)  # Normalize to 0-1 assuming 10k capital
        normalized_win_rate = win_rate / 100
        normalized_sharpe = min(max(sharpe / 2, 0.0), 1.0)
        normalized_profit_factor = min(profit_factor / 2, 1.0)  # Profit factor >2 is excellent
        normalized_expectancy = min(max(expectancy / 50, 0.0), 1.0)  # Expectancy >50 is good
        
        # Weighted probability: emphasize expectancy and profit factor for scalping
        probability = (
            0.2 * normalized_pnl +
            0.2 * normalized_win_rate +
            0.2 * normalized_sharpe +
            0.2 * normalized_profit_factor +
            0.2 * normalized_expectancy
        )
        probability_percentage = probability * 100
        # For scalping, enforce real profitability: positive PnL, positive expectancy, profit factor >1
        flag_reached = pnl > 0 and expectancy > 0 and profit_factor > 1.0
        
        payload = {
            "probability": probability_percentage,
            "flag_reached": flag_reached,
            "details": {
                "normalized_pnl": normalized_pnl,
                "normalized_win_rate": normalized_win_rate,
                "normalized_sharpe": normalized_sharpe,
                "normalized_profit_factor": normalized_profit_factor,
                "normalized_expectancy": normalized_expectancy,
            },
        }
        return payload

    def build_daily_summary(self, metrics: Dict[str, Any], probability: Dict[str, Any]) -> Dict[str, Any]:
        """Create a payload consumed by the OpenAI optimizer and reports."""
        summary = {
            "metrics": metrics,
            "probability": probability,
        }
        return summary

    def calculate_readiness(
        self,
        *,
        recent_metrics: Sequence[Dict[str, Any]],
        backtest_state: Dict[str, Any],
        decision_stats_7d: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute a 'ready for micro-live review' score (0..100) from multiple signals.

        This never enables real trading; it only produces an advisory signal.
        """
        r_cfg = (cfg.get("readiness") or {}) if isinstance(cfg, dict) else {}
        min_days = int(r_cfg.get("min_days", 14) or 14)
        min_trades = int(r_cfg.get("min_trades", 200) or 200)
        min_pf = float(r_cfg.get("min_profit_factor", 1.15) or 1.15)
        min_sharpe = float(r_cfg.get("min_sharpe", 0.2) or 0.2)
        max_dd = float(r_cfg.get("max_drawdown", 300.0) or 300.0)
        max_exec_err_rate = float(r_cfg.get("max_execution_error_rate", 0.02) or 0.02)
        min_bt_folds = int(r_cfg.get("min_backtest_folds", 3) or 3)
        min_bt_sharpe = float(r_cfg.get("min_backtest_avg_sharpe", 0.1) or 0.1)
        min_score = float(r_cfg.get("min_score", 80) or 80)

        metrics_list = list(recent_metrics or [])
        days = len(metrics_list)
        total_trades = sum(int(m.get("trades_count", 0) or 0) for m in metrics_list)

        def _f(v: Any) -> float:
            try:
                return float(v or 0.0)
            except Exception:
                return 0.0

        pf_values = [_f(m.get("profit_factor")) for m in metrics_list if m.get("profit_factor") is not None]
        sharpe_values = [_f(m.get("sharpe")) for m in metrics_list if m.get("sharpe") is not None]
        dd_values = [_f(m.get("drawdown")) for m in metrics_list if m.get("drawdown") is not None]
        pnl_values = [_f(m.get("pnl")) for m in metrics_list if m.get("pnl") is not None]
        profitable_days = sum(1 for p in pnl_values if p > 0)

        avg_pf = sum(pf_values) / max(len(pf_values), 1)
        avg_sharpe = sum(sharpe_values) / max(len(sharpe_values), 1)
        worst_dd = max(dd_values) if dd_values else 0.0
        consistency = profitable_days / max(len(pnl_values), 1)

        blocked = (decision_stats_7d or {}).get("blocked_by_reason") or {}
        exec_err = int(blocked.get("execution_error", 0) or 0)
        total_events = int((decision_stats_7d or {}).get("total", 0) or 0)
        exec_err_rate = exec_err / max(total_events, 1)

        bt_summary = (backtest_state or {}).get("summary") or {}
        bt_folds = int(bt_summary.get("folds", 0) or 0)
        bt_avg_sharpe = _f(bt_summary.get("avg_test_sharpe"))

        score = 0.0
        reasons: list[str] = []
        checks: Dict[str, Any] = {}

        score += 25.0 * min(1.0, days / max(min_days, 1))
        checks["days"] = {"value": days, "min": min_days}
        if days < min_days:
            reasons.append(f"Not enough daily history ({days} < {min_days}).")

        score += 25.0 * min(1.0, total_trades / max(min_trades, 1))
        checks["trades"] = {"value": total_trades, "min": min_trades}
        if total_trades < min_trades:
            reasons.append(f"Not enough trades ({total_trades} < {min_trades}).")

        score += 20.0 * min(1.0, avg_pf / max(min_pf, 1e-9))
        checks["profit_factor"] = {"avg": round(avg_pf, 4), "min": min_pf}
        if avg_pf < min_pf:
            reasons.append(f"Profit factor too low (avg {avg_pf:.2f} < {min_pf:.2f}).")

        denom = max((min_sharpe - (-0.2)), 1e-9)
        score += 15.0 * min(1.0, (avg_sharpe - (-0.2)) / denom)
        checks["sharpe"] = {"avg": round(avg_sharpe, 4), "min": min_sharpe}
        if avg_sharpe < min_sharpe:
            reasons.append(f"Sharpe too low (avg {avg_sharpe:.2f} < {min_sharpe:.2f}).")

        score += 10.0 * min(1.0, consistency / 0.6)
        checks["consistency_days"] = {"ratio": round(consistency, 4), "target": 0.6}
        if consistency < 0.6:
            reasons.append(f"Low day consistency ({consistency*100:.0f}% profitable days).")

        dd_component = 1.0 - min(1.0, worst_dd / max(max_dd, 1e-9))
        score += 5.0 * max(0.0, dd_component)
        checks["drawdown"] = {"max": round(worst_dd, 4), "max_allowed": max_dd}
        if worst_dd > max_dd:
            reasons.append(f"Drawdown too high ({worst_dd:.2f} > {max_dd:.2f}).")

        score += 5.0 * (1.0 - min(1.0, exec_err_rate / max(max_exec_err_rate, 1e-9)))
        checks["execution_errors"] = {"rate": round(exec_err_rate, 4), "max_rate": max_exec_err_rate, "count": exec_err, "events": total_events}
        if exec_err_rate > max_exec_err_rate:
            reasons.append(f"Too many execution errors ({exec_err_rate*100:.1f}% > {max_exec_err_rate*100:.1f}%).")

        bt_ok = bt_folds >= min_bt_folds and bt_avg_sharpe >= min_bt_sharpe
        checks["backtest"] = {"folds": bt_folds, "min_folds": min_bt_folds, "avg_sharpe": round(bt_avg_sharpe, 4), "min_avg_sharpe": min_bt_sharpe}
        if bt_ok:
            score += 10.0
        else:
            reasons.append(f"Backtest not strong enough (folds {bt_folds}/{min_bt_folds}, sharpe {bt_avg_sharpe:.2f}/{min_bt_sharpe:.2f}).")

        score = max(0.0, min(100.0, score))

        ready = (
            score >= min_score
            and days >= min_days
            and total_trades >= min_trades
            and avg_pf >= min_pf
            and avg_sharpe >= min_sharpe
            and worst_dd <= max_dd
            and exec_err_rate <= max_exec_err_rate
            and bt_ok
        )

        return {
            "score": round(score, 2),
            "ready": bool(ready),
            "reasons": reasons[:12],
            "checks": checks,
            "window_days": days,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

    def _calculate_drawdown(self, pnl_values: Sequence[float]) -> float:
        """Calculate max drawdown from cumulative PnL."""
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnl_values:
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    def _calculate_sharpe_ratio(self, pnl_values: Sequence[float]) -> float:
        """Calculate a simple Sharpe ratio using daily returns."""
        if not pnl_values:
            return 0.0
        avg_return = mean(pnl_values)
        std_dev = math.sqrt(mean([(p - avg_return) ** 2 for p in pnl_values]) or 1e-9)
        sharpe = avg_return / std_dev if std_dev else 0.0
        return sharpe
