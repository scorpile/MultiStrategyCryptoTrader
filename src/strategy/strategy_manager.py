"""Runtime manager handling strategy instantiation, switching, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import STRATEGY_REGISTRY
from .strategy_base import BaseStrategy


class StrategyManager:
    """Keeps track of the active strategy and exposes a hot-swap API."""

    def __init__(
        self,
        *,
        risk_manager: Any,
        config: Optional[Dict[str, Any]] = None,
        state_config_path: Path | str = Path("state/config.json"),
    ) -> None:
        self.risk_manager = risk_manager
        self.config = config or {}
        self.config_path = Path(state_config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_state = self._load_runtime_state()
        self.runtime_state.setdefault("rl", {"enabled": True, "policy_path": "state/rl_policy.json"})
        self.runtime_state.setdefault("ml", {"enabled": True})
        self.active_strategy_name = self.runtime_state.get("active_strategy") or self.config.get("default", "moderate")
        self.active_strategy = self._instantiate(self.active_strategy_name)
        self._rl_policy_cache = self._load_rl_policy()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_active_strategy(self) -> BaseStrategy:
        return self.active_strategy

    def get_active_strategy_name(self) -> str:
        return self.active_strategy_name

    def set_active_strategy(self, name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
        if overrides:
            strategy_configs = self.runtime_state.setdefault("strategies", {})
            strategy_configs.setdefault(name, {}).update(overrides)
        self.active_strategy_name = name
        self.runtime_state["active_strategy"] = name
        self.active_strategy = self._instantiate(name)
        self._persist_runtime_state()
        return self.get_status_snapshot()

    def update_strategy_config(self, name: str, overrides: Dict[str, Any]) -> None:
        strategy_configs = self.runtime_state.setdefault("strategies", {})
        strategy_configs.setdefault(name, {}).update(overrides)
        if name == self.active_strategy_name:
            self.active_strategy.update_config(overrides)
        self._persist_runtime_state()

    def generate_decisions(
        self,
        indicators: Dict[str, Any],
        *,
        ml_context: Optional[Dict[str, Any]] = None,
        rl_context: Optional[Dict[str, Any]] = None,
        capital: Optional[float] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        strategy = self.get_active_strategy()
        runtime_rl = rl_context or self.get_rl_policy()
        context: Dict[str, Any] = {"capital": capital, "ml": ml_context or {}, "rl": runtime_rl}
        if extra_context:
            context.update(extra_context)
        signal = strategy.generate_signal(indicators, context=context)
        decision = {
            "decision": signal.get("decision", "hold"),
            "price": signal.get("price"),
            "suggested_size": signal.get("suggested_size", 0.0),
            "confidence": signal.get("confidence", 0.0),
            "reasons": signal.get("reasons", []),
            "metadata": signal.get("metadata", {}),
        }
        return [decision]

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for key, cls in STRATEGY_REGISTRY.items():
            instance = cls(config=self.runtime_state.get("strategies", {}).get(key), risk_manager=self.risk_manager)
            payload.append(
                {
                    "name": key,
                    "description": getattr(cls, "description", ""),
                    "config_schema": instance.config_schema(),
                }
            )
        return payload

    def get_status_snapshot(self) -> Dict[str, Any]:
        active_cfg: Dict[str, Any] = {}
        try:
            active_cfg = dict(self.active_strategy.config or {})
        except Exception:
            active_cfg = {}
        strategy_overrides = self.runtime_state.get("strategies", {}).get(self.active_strategy_name, {})
        return {
            "active_strategy": self.active_strategy_name,
            "active_config": active_cfg,
            "active_overrides": strategy_overrides,
            "available": self.get_available_strategies(),
            "ml": self.runtime_state.get("ml", {}),
            "rl": self.runtime_state.get("rl", {}),
        }

    def refresh_runtime_state(self) -> None:
        self.runtime_state = self._load_runtime_state()
        self.active_strategy_name = self.runtime_state.get("active_strategy", self.active_strategy_name)
        self.active_strategy = self._instantiate(self.active_strategy_name)
        self._rl_policy_cache = self._load_rl_policy()

    def persist_ml_state(self, ml_state: Dict[str, Any]) -> None:
        self.runtime_state["ml"] = ml_state
        self._persist_runtime_state()

    def persist_rl_policy(self, policy: Dict[str, Any]) -> None:
        policy_path = Path(self.runtime_state.get("rl", {}).get("policy_path", "state/rl_policy.json"))
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        self._rl_policy_cache = policy
        self.runtime_state.setdefault("rl", {})["last_score"] = policy.get("score")
        self._persist_runtime_state()

    def get_rl_policy(self) -> Dict[str, Any]:
        return self._rl_policy_cache or {}

    def get_ml_state(self) -> Dict[str, Any]:
        return self.runtime_state.get("ml", {})

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _instantiate(self, name: str) -> BaseStrategy:
        cls = STRATEGY_REGISTRY[name]
        strategy_configs = self.runtime_state.get("strategies", {})
        config = strategy_configs.get(name) or self.config.get("strategies", {}).get(name) or {}
        return cls(config=config, risk_manager=self.risk_manager)

    def _load_runtime_state(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            runtime_default = {
                "active_strategy": self.config.get("default", "moderate"),
                "strategies": {},
                "ml": {"enabled": True, "model": "lightgbm"},
                "rl": {"enabled": True, "policy_path": "state/rl_policy.json"},
            }
            self.config_path.write_text(json.dumps(runtime_default, indent=2), encoding="utf-8")
            return runtime_default
        with self.config_path.open("r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return {"active_strategy": "moderate", "strategies": {}}

    def _persist_runtime_state(self) -> None:
        self.config_path.write_text(json.dumps(self.runtime_state, indent=2), encoding="utf-8")

    def _load_rl_policy(self) -> Dict[str, Any]:
        policy_cfg = self.runtime_state.get("rl", {})
        policy_path = Path(policy_cfg.get("policy_path", "state/rl_policy.json"))
        if not policy_path.exists():
            return {}
        try:
            return json.loads(policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
