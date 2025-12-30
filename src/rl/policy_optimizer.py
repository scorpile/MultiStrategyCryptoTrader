"""Simple RL (Q-learning inspired) optimizer adjusting MultiFactor weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


FACTOR_KEYS = ["rsi", "macd", "bollinger", "trend", "volume", "atr", "candle", "ml"]


@dataclass
class PolicyResult:
    weights: Dict[str, float]
    score: float
    stats: Dict[str, Any]
    path: str


class PolicyOptimizer:
    """Learns lightweight policy weights using simulated trade history."""

    def __init__(
        self,
        *,
        policy_path: Path | str = Path("state/rl_policy.json"),
        learning_rate: float = 0.05,
        reward_scale: float = 100.0,
        strategy_manager: Optional[Any] = None,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.reward_scale = reward_scale
        self.strategy_manager = strategy_manager

    def optimize(self, trades: Sequence[Dict[str, Any]]) -> PolicyResult:
        stats = self._summarize_trades(trades)
        reward = stats["pnl"] - stats["max_drawdown"] * 0.4
        previous = self._load_policy()
        next_weights = self._update_weights(previous.get("weights", self._default_weights()), reward)
        policy = {
            "weights": next_weights,
            "score": reward,
            "stats": stats,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        self.policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        if self.strategy_manager:
            self.strategy_manager.persist_rl_policy(policy)
        return PolicyResult(weights=next_weights, score=reward, stats=stats, path=str(self.policy_path))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _summarize_trades(self, trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        pnl_values: List[float] = []
        wins = 0
        losses = 0
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for trade in trades:
            pnl = float(trade.get("pnl") or trade.get("trade_pnl") or 0.0)
            pnl_values.append(pnl)
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)
        total_pnl = sum(pnl_values)
        total = max(len(pnl_values), 1)
        win_rate = wins / total
        return {
            "pnl": total_pnl,
            "num_trades": total,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
        }

    def _update_weights(self, weights: Dict[str, float], reward: float) -> Dict[str, float]:
        adjustment = self.learning_rate * ((reward / self.reward_scale) if self.reward_scale else 0.0)
        for key in FACTOR_KEYS:
            if key not in weights:
                weights[key] = 1 / len(FACTOR_KEYS)
        weights["trend"] += adjustment * 0.3
        weights["ml"] += adjustment * 0.25
        weights["rsi"] += adjustment * 0.1

        if reward < 0:
            penalty = abs(adjustment)
            weights["atr"] += penalty * 0.3
            weights["bollinger"] += penalty * 0.1
            weights["macd"] -= penalty * 0.2

        total = sum(weights.values()) or 1.0
        normalized = {key: max(0.0, value) / total for key, value in weights.items()}
        return normalized

    def _load_policy(self) -> Dict[str, Any]:
        if not self.policy_path.exists():
            return {"weights": self._default_weights(), "score": 0.0}
        try:
            return json.loads(self.policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"weights": self._default_weights(), "score": 0.0}

    @staticmethod
    def _default_weights() -> Dict[str, float]:
        base = 1 / len(FACTOR_KEYS)
        return {key: base for key in FACTOR_KEYS}
