"""Scheduler coordinating trading cycles, ML/RL context, and daily automations."""

from __future__ import annotations

import logging
import json
import os
import threading
from datetime import date, datetime, time as time_cls, timedelta
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from datetime import timezone

from src.exchange.paper_trading import SimulatedOrder
from src.exchange.errors import OrderRejected
from src.exchange.symbol_rules import parse_symbol_rules
from src.data.market_data_store import MarketDataStore
from src.ml import feature_engineering
from src.strategy import technicals
from src.evaluation.trading_gate import evaluate_trading_gate
from src.evaluation.fear_greed_provider import FearGreedProvider
from src.evaluation.auto_tuner import auto_tune_strategy
from src.backtesting.walkforward_runner import run_walkforward, _default_grid


class Scheduler:
    """Central orchestrator for trading cadence, evaluation, and optimizations."""

    def __init__(
        self,
        *,
        state_manager: Any,
        exchange_client: Any,
        strategy_manager: Any,
        paper_trading_engine: Any,
        live_trading_engine: Optional[Any] = None,
        evaluator: Any,
        reporter: Any,
        ai_optimizer: Optional[Any],
        config: Dict[str, Any],
        ml_trainer: Optional[Any] = None,
        ml_predictor: Optional[Any] = None,
        rl_optimizer: Optional[Any] = None,
        backtesting_engine: Optional[Any] = None,
        market_data_store: Optional[MarketDataStore] = None,
        trading_interval_seconds: int = 30,  # Reduced for 1m scalping
        daily_run_time: str = "23:55",
    ) -> None:
        # ... existing code ...
        self.training_status: str = ""
        self.state_manager = state_manager
        self.exchange_client = exchange_client
        self.strategy_manager = strategy_manager
        self.paper_trading_engine = paper_trading_engine
        self.live_trading_engine = live_trading_engine
        self.evaluator = evaluator
        self.reporter = reporter
        self.ai_optimizer = ai_optimizer
        self.ml_trainer = ml_trainer
        self.ml_predictor = ml_predictor
        self.rl_optimizer = rl_optimizer
        self.backtesting_engine = backtesting_engine
        self.market_data_store = market_data_store
        self.config = config
        self.symbol = config.get("general", {}).get("base_symbol", "SOLUSDT")
        self.interval = config.get("general", {}).get("time_frame", "1h")
        self.trading_interval_seconds = trading_interval_seconds
        self.daily_run_time = self._parse_daily_time(daily_run_time)
        self.stop_event = threading.Event()
        self._admin_lock = threading.RLock()
        self.last_daily_run: Optional[date] = None
        self.latest_ml_context: Dict[str, Any] = {}
        self.last_decisions: List[Dict[str, Any]] = []
        self.last_daily_summary: Dict[str, Any] = {}
        self.last_rl_result: Optional[Dict[str, Any]] = None
        self.last_ai_response: Optional[Dict[str, Any]] = None
        self.last_indicator_payload: Dict[str, Any] = {}
        self.last_indicator_update: Optional[datetime] = None
        self.last_ml_training_summary: Dict[str, Any] = {}
        self._last_ml_training_date: Optional[date] = None
        self.validation_mode: bool = False  # Live validation without real trades
        self._last_symbol_sync: Optional[datetime] = None
        self.training_progress: int = 0
        self.is_training: bool = False
        self._last_buy_time: Optional[datetime] = None
        self._gate_status: Dict[str, Any] = {"enabled": True, "can_enter": True, "reasons": ["OK"], "updated_at": ""}
        self._fear_greed_provider = FearGreedProvider(
            provider=str(self.config.get("sentiment", {}).get("provider", "alternative_me") or "alternative_me"),
            coinmarketcap_api_key=str(self.config.get("sentiment", {}).get("coinmarketcap_api_key") or ""),
            min_refresh_seconds=int(self.config.get("sentiment", {}).get("refresh_seconds", 900) or 900),
        )
        self._fear_greed: Dict[str, Any] = {}
        self._sentiment_policy: Dict[str, Any] = {"regime": "unknown", "multipliers": {}}
        self._last_equity_snapshot_at: Optional[datetime] = None
        self._risk_pause: Dict[str, Any] = {}
        self._manual_pause: Dict[str, Any] = {}
        self._auto_tune_state: Dict[str, Any] = {}
        self._last_tuned_closed_count: int = 0
        self._last_control_tune_event_id: int = 0
        self._last_control_tune_at: Optional[datetime] = None
        self._last_signal_tune_event_id: int = 0
        self._last_signal_tune_at: Optional[datetime] = None
        self._last_gate_tune_event_id: int = 0
        self._last_gate_tune_at: Optional[datetime] = None
        self._backtest_state: Dict[str, Any] = {}
        self._paper_trailing_state: Dict[str, Any] = {}
        self.execution_mode: str = "paper"

        # Restore runtime toggles/snapshots from persisted parameters (best-effort).
        try:
            params = self.state_manager.load_parameters()
            try:
                configured_mode = str((self.config.get("execution", {}) or {}).get("mode") or "paper").lower().strip()
            except Exception:
                configured_mode = "paper"
            stored_mode = str(params.get("execution_mode") or configured_mode).lower().strip()
            self.execution_mode = "live" if stored_mode == "live" else "paper"
            try:
                active_symbol = params.get("active_symbol")
                if active_symbol:
                    normalized = self._normalize_symbol(active_symbol)
                    if normalized:
                        self.symbol = normalized
                        self.config.setdefault("general", {})["base_symbol"] = normalized
            except Exception:
                pass
            if "sentiment_enabled" in params:
                self.config.setdefault("sentiment", {})["enabled"] = bool(params["sentiment_enabled"])
            if isinstance(params.get("fear_greed_latest"), dict):
                self._fear_greed = params["fear_greed_latest"]
            if isinstance(params.get("trading_overrides"), dict):
                self.config.setdefault("trading", {}).update(params["trading_overrides"])
            if isinstance(params.get("gate_overrides"), dict):
                self.config.setdefault("gate", {}).update(params["gate_overrides"])
            if isinstance(params.get("auto_tune_last"), dict):
                self._auto_tune_state = params["auto_tune_last"]
            if isinstance(params.get("risk_pause"), dict):
                self._risk_pause = params["risk_pause"]
            if isinstance(params.get("manual_pause"), dict):
                self._manual_pause = params["manual_pause"]
            if isinstance(params.get("backtest_last"), dict):
                self._backtest_state = params["backtest_last"]
            if isinstance(params.get("paper_trailing_state"), dict):
                self._paper_trailing_state = params["paper_trailing_state"]
        except Exception:
            pass

        # Ensure symbol rules align with the (possibly restored) active symbol.
        try:
            self._refresh_symbol_rules(self.symbol)
        except Exception:
            pass

        # Learning mode: when not live, do not carry over any circuit-breaker pause.
        if self.execution_mode != "live" and self._risk_pause:
            self._risk_pause = {}
            try:
                self.state_manager.save_parameters({"risk_pause": self._risk_pause})
            except Exception:
                pass

        # Initial indicator fetch will be done lazily in get_runtime_snapshot

    def start(self) -> None:
        """Kick off the synchronous loop (trading cadence + daily tasks)."""
        logging.info("Scheduler started with trading cadence %ss", self.trading_interval_seconds)
        while not self.stop_event.is_set():
            now = datetime.utcnow()
            try:
                self._ensure_ml_retrained(now)
            except Exception:
                logging.exception("ML retraining check failed; continuing without retraining.")
            try:
                self.run_trading_cycle_once(now)
            except Exception:
                logging.exception("Trading cycle failed.")

            if self._should_run_daily(now):
                try:
                    self.run_daily_tasks(now)
                    self.last_daily_run = now.date()
                except Exception:
                    logging.exception("Daily task execution failed.")

            self.stop_event.wait(self.trading_interval_seconds)

    def stop(self) -> None:
        """Stop all periodic tasks gracefully and flush pending state."""
        logging.info("Stop signal received; shutting down scheduler.")
        self.stop_event.set()

    def run_trading_cycle_once(self, now: datetime) -> None:
        """Single trading iteration: fetch data, compute signals, simulate trades."""
        with self._admin_lock:
            self._sync_symbol_from_state()
            self._update_manual_pause(now=now)
            trading_cfg = self.config.get("trading", {}) or {}
            strategy_only_signals = bool(trading_cfg.get("strategy_only_signals", True))
            ohlcv = self.exchange_client.fetch_ohlcv(self.symbol, self.interval, limit=200)
            if not ohlcv:
                logging.warning("Received empty OHLCV data; skipping cycle.")
                return
            try:
                self._cache_recent_ohlcv(symbol=self.symbol, interval=self.interval, candles=ohlcv)
            except Exception:
                pass
            indicator_payload = self._build_indicator_payload(ohlcv)
            ml_context = self._build_ml_context(indicator_payload)
            self.last_indicator_payload = indicator_payload
            self.last_indicator_update = now
            engine = self._get_active_engine()

            # Log key indicators for debugging
            indicators_log = {k: (v[-1] if isinstance(v, list) and v else v) for k, v in indicator_payload.items() if k in ['close', 'rsi', 'ema_fast', 'ema_slow', 'atr', 'volume_sma', 'macd_histogram', 'bollinger_width']}
            logging.info(f"Indicators: {indicators_log}")

            price_for_equity = float(indicator_payload.get("closes", [0.0])[-1] or 0.0)

            # Ensure the engine state is in sync before evaluating stops/TP/trailing.
            try:
                self._get_portfolio_snapshot(engine, current_price=price_for_equity)
            except Exception:
                pass

            # Re-apply persisted paper trailing stop state (survives restarts).
            try:
                self._apply_paper_trailing_state(engine=engine)
            except Exception:
                pass

            # Optional scheduler-managed exits (disabled by default to keep strategy-only evaluation).
            if not strategy_only_signals:
                stop_loss_decisions = self._check_stop_loss_take_profit(indicator_payload)
                if stop_loss_decisions:
                    self._execute_decisions(stop_loss_decisions, now=now, indicators=indicator_payload)
                    logging.info("Executed stop-loss/take-profit: %s", stop_loss_decisions)

            portfolio = self._get_portfolio_snapshot(engine, current_price=price_for_equity)
            # Keep circuit breaker up-to-date even when dashboard is closed.
            try:
                equity_now = self._estimate_equity_usdt(portfolio, current_price=price_for_equity)
                self._update_risk_pause(now=now, current_equity=equity_now)
            except Exception:
                pass
            position_qty = 0.0
            try:
                pos = next((p for p in (portfolio.get("open_positions") or []) if p.get("symbol") == self.symbol), None)
                position_qty = float((pos or {}).get("quantity", 0.0) or 0.0)
            except Exception:
                position_qty = 0.0

            exploration = self._is_exploration_mode()

            try:
                spread_bps = float(trading_cfg.get("spread_bps", 0.0) or 0.0)
            except Exception:
                spread_bps = 0.0
            spread_pct = max(0.0, spread_bps) / 10_000.0

            # Some strategies prefer a different (less noisy) signal timeframe, while we still monitor 1m
            # for stops/trailing updates via `indicator_payload`.
            signal_payload = indicator_payload
            signal_interval = self.interval
            try:
                strategy = self.strategy_manager.get_active_strategy()
                signal_interval = self._get_strategy_signal_interval(strategy)
                if signal_interval and str(signal_interval) != str(self.interval):
                    ohlcv_signal = self.exchange_client.fetch_ohlcv(self.symbol, signal_interval, limit=200)
                    if ohlcv_signal:
                        try:
                            self._cache_recent_ohlcv(symbol=self.symbol, interval=signal_interval, candles=ohlcv_signal)
                        except Exception:
                            pass
                        ohlcv_signal = self._strip_open_candle(ohlcv_signal, now=now)
                        if ohlcv_signal:
                            signal_payload = self._build_indicator_payload(ohlcv_signal)
            except Exception:
                signal_payload = indicator_payload
                signal_interval = self.interval

            decisions = self.strategy_manager.generate_decisions(
                signal_payload,
                ml_context=ml_context,
                capital=portfolio["cash_balance"],
                extra_context={
                    "exploration": exploration,
                    "position_qty": position_qty,
                    "spread_bps": spread_bps,
                    "spread_pct": spread_pct,
                },
            )

            # Execute at the latest monitored price (1m), but keep the signal interval context for audit/debug.
            if signal_interval and str(signal_interval) != str(self.interval):
                for d in decisions:
                    if d.get("decision") not in {"buy", "sell"}:
                        continue
                    meta = d.get("metadata") or {}
                    try:
                        meta = dict(meta) if isinstance(meta, dict) else {}
                    except Exception:
                        meta = {}
                    meta.setdefault("signal_interval", str(signal_interval))
                    meta.setdefault("signal_price", d.get("price"))
                    meta["execution_price"] = float(price_for_equity)
                    d["metadata"] = meta
                    d["price"] = float(price_for_equity)
            if not self.validation_mode:
                self._execute_decisions(decisions, now=now, indicators=indicator_payload)
            else:
                self._record_blocked_decisions(decisions, now=now, reason="validation_mode")

            # Update trailing stop levels after any position changes (scheduler-managed).
            if not strategy_only_signals:
                try:
                    self._maybe_update_paper_trailing_stop(now=now, indicators=indicator_payload)
                except Exception:
                    pass
            self.latest_ml_context = ml_context
            self.last_decisions = decisions
            logging.info(
                "Strategy %s decisions: %s%s",
                self.strategy_manager.get_active_strategy_name(),
                [d.get("decision") for d in decisions],
                " (VALIDATION MODE - no trades executed)" if self.validation_mode else "",
            )
            try:
                self._maybe_auto_tune_from_signals(now=now)
            except Exception:
                pass
            logging.info("Completed trading cycle at %s UTC", now.isoformat())

    def run_daily_tasks(self, now: datetime) -> None:
        """Daily job: evaluation, RL/AI optimization, report generation."""
        with self._admin_lock:
            self.is_training = True
            self.training_progress = 0
            try:
                logging.info("Running daily tasks for %s", now.date().isoformat())
                self.training_status = "Evaluating daily metrics..."
                date_str = now.date().isoformat()
                trades = self.state_manager.get_trades_for_day(date_str, symbol=self.symbol)
                metrics = self.evaluator.compile_daily_metrics(trades)
                probability = self.evaluator.calculate_probability_of_profit(metrics)
                summary = self.evaluator.build_daily_summary(metrics, probability)
                self.last_daily_summary = summary

                self.training_status = "Saving metrics and probability..."
                self.state_manager.save_daily_metrics({"metric_date": date_str, **metrics})
                self.state_manager.set_probability_flag(probability["flag_reached"], probability["probability"], probability)

                # Activate validation mode if profitability target reached
                if probability["flag_reached"] and not self.validation_mode:
                    self.validation_mode = True
                    logging.info("Profitability target reached! Entering VALIDATION MODE (live data, no real trades).")
                    self.state_manager.save_parameters({"validation_mode": True})

                if self.rl_optimizer:
                    self.training_status = "Running RL optimization..."
                    rl_result = self.rl_optimizer.optimize(trades)
                    self.last_rl_result = rl_result.stats | {"score": rl_result.score}
                else:
                    self.last_rl_result = None

                if self.ai_optimizer:
                    self.training_status = "Consulting AI optimizer..."
                    try:
                        ai_response = self.ai_optimizer.run(summary)
                        self.last_ai_response = ai_response
                    except Exception:
                        logging.exception("AI optimizer failed.")
                        self.last_ai_response = {"error": "ai_optimizer_failure"}

                # Ensure ML model is retrained if needed
                self.training_status = "Retraining ML model..."
                self._ensure_ml_retrained(now)

                # Run daily backtest for validation
                self.training_status = "Running backtest optimization..."
                if self.backtesting_engine:
                    self._run_daily_backtest(now)

                # Compute "ready for micro-live review" advisory flag (never enables real trading).
                try:
                    readiness_cfg = self.config.get("readiness", {}) or {}
                    window_days = int(readiness_cfg.get("window_days", readiness_cfg.get("min_days", 14)) or 14)
                    recent_metrics = self.state_manager.get_recent_metrics(limit=max(1, window_days))
                    try:
                        decision_stats_7d = self.state_manager.get_decision_stats_since(datetime.utcnow() - timedelta(days=7), symbol=self.symbol)
                    except Exception:
                        decision_stats_7d = {}
                    readiness = self.evaluator.calculate_readiness(
                        recent_metrics=recent_metrics,
                        backtest_state=self._backtest_state or {},
                        decision_stats_7d=decision_stats_7d,
                        cfg=self.config,
                    )
                    self.state_manager.set_flag(
                        flag_name="ready_for_micro_live",
                        enabled=bool(readiness.get("ready", False)),
                        probability=float(readiness.get("score", 0.0) or 0.0),
                        metadata=readiness,
                    )
                except Exception:
                    pass

                self.training_status = "Generating reports..."
                report_context = {
                    "date": date_str,
                    "symbol": self.symbol,
                    "time_frame": self.interval,
                    "simulation_mode": True,
                    "pnl_today": metrics["pnl"],
                    "trades_count": metrics["trades_count"],
                    "win_rate": metrics["win_rate"],
                    "max_drawdown": metrics["drawdown"],
                    "sharpe_ratio": metrics["sharpe"],
                    "profit_probability": probability["probability"],
                    "probability_flag": probability["flag_reached"],
                    "ml_confidence": self.latest_ml_context.get("confidence"),
                    "rl_score": (self.last_rl_result or {}).get("score"),
                    "notes": (self.last_ai_response or {}).get("raw_response", {}).get("notes", ""),
                }
                self.reporter.save_daily_reports(report_context)
                self.training_progress = 100
                self.training_status = "Training completed successfully."
            finally:
                self.is_training = False

    def reload_configuration(self) -> None:
        """Reload strategy configuration from /state/config.json."""
        with self._admin_lock:
            logging.info("Reload configuration requested; refreshing StrategyManager runtime state.")
            self.strategy_manager.refresh_runtime_state()

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        raw = str(value or "").strip().upper()
        if not raw:
            return ""
        raw = raw.replace("/", "").replace("-", "").replace("_", "")
        if raw in {"SOL", "ETH", "BTC"}:
            return f"{raw}USDT"
        if raw.endswith("USDT") and len(raw) >= 6:
            return raw
        return raw

    def _get_allowed_symbols(self) -> List[str]:
        general_cfg = self.config.get("general", {}) or {}
        configured = general_cfg.get("allowed_symbols")
        source: List[Any] = configured if isinstance(configured, list) else []
        if not source:
            source = ["SOLUSDT", "ETHUSDT", "BTCUSDT"]
        allowed: List[str] = []
        seen: set[str] = set()
        for item in source:
            sym = self._normalize_symbol(item)
            if sym and sym not in seen:
                allowed.append(sym)
                seen.add(sym)
        return allowed

    def _refresh_symbol_rules(self, symbol: str) -> None:
        """Best-effort refresh of exchange trading rules for a symbol."""
        sym = str(symbol or "").upper()
        if not sym:
            return
        try:
            info = self.exchange_client.fetch_symbol_info(sym)
            rules = parse_symbol_rules(info) if isinstance(info, dict) else None
        except Exception:
            rules = None
        try:
            self.paper_trading_engine.set_symbol_rules(sym, rules)
        except Exception:
            pass
        try:
            if self.backtesting_engine is not None:
                self.backtesting_engine.symbol_rules = rules
        except Exception:
            pass

    def _apply_symbol_change(self, symbol: str, *, persist: bool) -> None:
        """Centralized symbol setter used by UI actions and background sync."""
        self.symbol = symbol
        self.config.setdefault("general", {})["base_symbol"] = symbol
        if persist:
            try:
                self.state_manager.save_parameters({"active_symbol": symbol})
            except Exception:
                pass
        try:
            self._refresh_symbol_rules(symbol)
        except Exception:
            pass
        # Force fresh indicators/decisions for the new symbol.
        self.last_indicator_payload = {}
        self.last_indicator_update = None
        self.last_decisions = []
        self._last_symbol_sync = datetime.utcnow()

    def _sync_symbol_from_state(self) -> None:
        """Keep the runtime symbol aligned with persisted active_symbol."""
        now = datetime.utcnow()
        if self._last_symbol_sync and (now - self._last_symbol_sync).total_seconds() < 2:
            return
        if self.execution_mode == "live" or self.is_training:
            self._last_symbol_sync = now
            return
        try:
            params = self.state_manager.load_parameters()
            active_symbol = params.get("active_symbol")
            normalized = self._normalize_symbol(active_symbol)
        except Exception:
            self._last_symbol_sync = now
            return
        self._last_symbol_sync = now
        if normalized and normalized != str(self.symbol or "").upper():
            try:
                open_positions = self.paper_trading_engine.get_open_positions()
                if open_positions:
                    logging.warning(
                        "Persisted active_symbol=%s differs from runtime=%s but positions are open; skipping sync.",
                        normalized,
                        self.symbol,
                    )
                    return
            except Exception:
                pass
            logging.info("Syncing active symbol from state: %s -> %s", self.symbol, normalized)
            self._apply_symbol_change(normalized, persist=False)

    def set_symbol(self, symbol: str) -> Dict[str, Any]:
        """Switch active trading symbol (paper only) and persist selection."""
        with self._admin_lock:
            requested = self._normalize_symbol(symbol)
            if not requested:
                raise ValueError("symbol is required")
            allowed = self._get_allowed_symbols()
            if requested not in allowed:
                raise ValueError(f"symbol '{requested}' not allowed (allowed: {', '.join(allowed)})")
            if self.execution_mode == "live":
                raise RuntimeError("Cannot change symbol while in LIVE mode.")
            if self.is_training:
                raise RuntimeError("Cannot change symbol while training is running.")
            try:
                snapshot = self.paper_trading_engine.get_portfolio_snapshot()
                open_positions = snapshot.get("open_positions", []) if isinstance(snapshot, dict) else []
            except Exception:
                try:
                    open_positions = self.paper_trading_engine.get_open_positions()
                except Exception:
                    open_positions = []
            if open_positions:
                symbols = []
                for pos in open_positions:
                    try:
                        sym = str(pos.get("symbol", "")).upper()
                        qty = float(pos.get("quantity", 0.0) or 0.0)
                    except Exception:
                        sym = ""
                        qty = 0.0
                    if sym and qty != 0.0 and sym not in symbols:
                        symbols.append(sym)
                suffix = f" ({', '.join(symbols)})" if symbols else ""
                raise RuntimeError(f"Cannot change symbol while positions are open{suffix}. Close positions first.")

            self._apply_symbol_change(requested, persist=True)
            return {"symbol": self.symbol, "allowed_symbols": allowed}

    def close_position(self, *, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Manually close an open position (SELL). Intended for dashboard recovery."""
        with self._admin_lock:
            now = datetime.utcnow()
            target = self._normalize_symbol(symbol) if symbol else str(self.symbol or "").upper()
            target = str(target or "").upper()
            if not target:
                raise ValueError("symbol is required")

            engine = self._get_active_engine()
            if self.execution_mode == "live":
                allowed = str(self._get_live_cfg().get("allow_symbol", self.symbol) or self.symbol).upper()
                if target != allowed:
                    raise RuntimeError(f"Can only close '{allowed}' in LIVE mode.")

            try:
                positions = engine.get_open_positions()
            except Exception:
                positions = []
            pos = next((p for p in positions if str(p.get("symbol", "")).upper() == target), None)
            qty = float((pos or {}).get("quantity", 0.0) or 0.0)
            min_qty = self._get_min_qty_for_symbol(engine, target)
            if qty <= 0.0 or (min_qty > 0 and abs(qty) < min_qty):
                raise RuntimeError(f"No open position to close for {target}.")

            price = 0.0
            if str(self.symbol or "").upper() == target:
                try:
                    price = float((self.last_indicator_payload.get("closes") or [0.0])[-1] or 0.0)
                except Exception:
                    price = 0.0
            if price <= 0.0:
                try:
                    candles = self.exchange_client.fetch_ohlcv(target, self.interval, limit=2)
                    if candles:
                        price = float(candles[-1].get("close", 0.0) or 0.0)
                except Exception:
                    price = 0.0
            if price <= 0.0:
                raise RuntimeError("Unable to determine current price for manual close.")

            decision = {
                "decision": "sell",
                "symbol": target,
                "price": float(price),
                "suggested_size": abs(qty),
                "confidence": 1.0,
                "reasons": [f"Manual close requested ({target})."],
                "metadata": {"trigger": "manual_close", "manual": True, "execution": {"force_full_fill": True}},
            }

            # If the symbol matches the scheduler's active symbol, reuse the standard execution path.
            if str(self.symbol or "").upper() == target:
                try:
                    self._execute_decisions([decision], now=now, indicators={"closes": [float(price)]})
                except Exception as e:
                    raise RuntimeError(f"Manual close failed: {str(e)}") from e
                return {"status": "submitted", "symbol": target, "quantity": abs(qty)}

            # Otherwise (paper-only), submit directly against the engine.
            order = SimulatedOrder(
                symbol=target,
                side="SELL",
                quantity=float(abs(qty)),
                price=float(price),
                metadata={
                    "trigger": "manual_close",
                    "manual": True,
                    "timestamp_utc": now.isoformat() + "Z",
                    "execution": {"force_full_fill": True},
                },
            )
            try:
                record = engine.submit_order(order)
                self._record_decision_event(decision, now=now, executed=True, trade_record=record, extra={"manual_close": True})
                return {"status": "closed", "symbol": target, "record": record}
            except OrderRejected as e:
                self._record_decision_event(
                    decision,
                    now=now,
                    executed=False,
                    blocked_reason=str(e.reason or "rejected"),
                    extra={"rejection": e.details or {}, "manual_close": True},
                )
                raise RuntimeError(f"Order rejected: {e.reason}") from e

    def admin_reset(self) -> Dict[str, Any]:
        """Hard reset: wipe SQLite runtime data and reset local runtime files."""
        with self._admin_lock:
            logging.warning("ADMIN RESET requested: wiping DB + runtime state.")

            # Preserve current selections to reapply after reset.
            preserved_symbol = str(self.symbol or "")
            try:
                preserved_state = json.loads(json.dumps(getattr(self.strategy_manager, "runtime_state", {})))
            except Exception:
                preserved_state = None

            # Reset scheduler runtime flags/caches (in-memory).
            self.validation_mode = False
            self.is_training = False
            self.training_progress = 0
            self.training_status = ""
            self.latest_ml_context = {}
            self.last_decisions = []
            self.last_daily_summary = {}
            self.last_rl_result = None
            self.last_ai_response = None
            self.last_indicator_payload = {}
            self.last_indicator_update = None
            self.last_ml_training_summary = {}
            self._last_ml_training_date = None
            self._last_buy_time = None
            self._gate_status = {"enabled": True, "can_enter": True, "reasons": ["OK"], "updated_at": ""}
            self._fear_greed = {}
            self._sentiment_policy = {"regime": "unknown", "multipliers": {}}
            self._last_equity_snapshot_at = None
            self._risk_pause = {}
            self._manual_pause = {}
            self._auto_tune_state = {}
            self._last_tuned_closed_count = 0
            self._last_control_tune_event_id = 0
            self._last_control_tune_at = None
            self.execution_mode = "paper"

            # Wipe persisted runtime data.
            try:
                self.state_manager.reset_db()
                # Ensure the file is fully recreated to avoid ghost rows.
                try:
                    self.state_manager.wipe_db_file()
                except Exception:
                    pass
                self.state_manager.init_db()
                # Validate that no positions/trades remain after reset.
                try:
                    leftover_trades = self.state_manager.count_total_trades()
                    leftover_exec_pos = self.state_manager.count_execution_positions()
                    if leftover_trades or leftover_exec_pos:
                        logging.warning(
                            "Post-reset validation found lingering rows (trades=%s, exec_positions=%s); forcing another wipe.",
                            leftover_trades,
                            leftover_exec_pos,
                        )
                        self.state_manager.reset_db()
                        self.state_manager.init_db()
                except Exception:
                    logging.exception("Post-reset validation failed; continuing.")
            except Exception:
                logging.exception("Failed to reset SQLite database.")

            # Remove runtime state files (they will be recreated with defaults).
            for path in (Path("state/config.json"), Path("state/rl_policy.json")):
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    logging.exception("Failed to delete runtime file: %s", path)

            # Refresh strategy/runtime config.
            try:
                self.strategy_manager.refresh_runtime_state()
            except Exception:
                logging.exception("Failed to refresh strategy runtime state after reset.")

            # Reapply preserved strategy runtime state (active strategy + overrides).
            try:
                if preserved_state:
                    cfg_path = getattr(self.strategy_manager, "config_path", None)
                    if cfg_path:
                        Path(cfg_path).write_text(json.dumps(preserved_state, indent=2), encoding="utf-8")
                        self.strategy_manager.refresh_runtime_state()
            except Exception:
                logging.exception("Failed to restore strategy selection after reset.")

            # Reset paper portfolio from (now empty) ledger.
            try:
                # Always clear in-memory paper state even if DB reset failed or rehydrate errors.
                try:
                    self.paper_trading_engine.cash_balance = float(getattr(self.paper_trading_engine, "initial_cash", 0.0) or 0.0)
                    self.paper_trading_engine.positions = {}
                    self.paper_trading_engine.realized_pnl = 0.0
                    self.paper_trading_engine.trade_history = []
                    self.paper_trading_engine._last_seen_trade_count = 0
                except Exception:
                    pass
                rehydrate = getattr(self.paper_trading_engine, "rehydrate_from_ledger", None)
                if callable(rehydrate):
                    rehydrate()
            except Exception:
                logging.exception("Failed to rehydrate paper portfolio after reset.")

            # Restore selected symbol and persist it.
            try:
                if preserved_symbol:
                    self._apply_symbol_change(self._normalize_symbol(preserved_symbol), persist=True)
            except Exception:
                logging.exception("Failed to restore symbol after reset.")

            return {"status": "ok", "reset_at": datetime.utcnow().isoformat() + "Z"}

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        """Expose metrics for UI/API endpoints."""
        with self._admin_lock:
            self._sync_symbol_from_state()
            now = datetime.utcnow()
            self._update_manual_pause(now=now)
            self._update_fear_greed(now)
            self._sentiment_policy = self._compute_sentiment_policy()

            # Ensure we have recent indicator data for dashboard (update every 10s or if empty)
            if (not self.last_indicator_payload or
                not self.last_indicator_update or
                (now - self.last_indicator_update).total_seconds() > 10):
                try:
                    ohlcv = self.exchange_client.fetch_ohlcv(self.symbol, self.interval, limit=200)
                    if ohlcv:
                        try:
                            self._cache_recent_ohlcv(symbol=self.symbol, interval=self.interval, candles=ohlcv)
                        except Exception:
                            pass
                        self.last_indicator_payload = self._build_indicator_payload(ohlcv)
                        self.last_indicator_update = now
                        logging.info("Fetched updated indicators for dashboard at %s", now.isoformat())
                except Exception:
                    logging.exception("Failed to fetch indicators")

            probability = self.state_manager.get_probability_flag()

            # Calculate current day's metrics
            date_str = now.date().isoformat()
            trades = self.state_manager.get_trades_for_day(date_str, symbol=self.symbol)
            current_metrics = self.evaluator.compile_daily_metrics(trades)

            # Calculate PnL series for last 7 days
            pnl_series = []
            for i in range(7):
                day = (now - timedelta(days=i)).date().isoformat()
                day_trades = self.state_manager.get_trades_for_day(day, symbol=self.symbol)
                day_metrics = self.evaluator.compile_daily_metrics(day_trades)
                pnl_series.append(day_metrics["pnl"])

            # Extract latest values for live indicators (coerce to numeric, handle None/NaN)
            live_indicators: Dict[str, float] = {}
            if self.last_indicator_payload:
                def _safe_last(key, default=0.0):
                    try:
                        val = self.last_indicator_payload.get(key)
                        if val is None:
                            return default
                        # handle list-like series
                        if isinstance(val, list):
                            if not val:
                                return default
                            candidate = val[-1]
                        else:
                            candidate = val
                        # coerce to float, guard against NaN/infinite
                        candidate_f = float(candidate)
                        if candidate_f != candidate_f:  # check NaN
                            return default
                        return candidate_f
                    except Exception:
                        return default

                live_indicators = {
                    "close": _safe_last("close", _safe_last("closes", 0.0)),
                    "rsi": _safe_last("rsi", 0.0),
                    "ema_fast": _safe_last("ema_fast", 0.0),
                    "ema_slow": _safe_last("ema_slow", 0.0),
                    "atr": _safe_last("atr", 0.0),
                    "volume_sma": _safe_last("volume_sma", 0.0),
                    "macd_histogram": _safe_last("macd_histogram", 0.0),
                    "bollinger_width": _safe_last("bollinger_width", 0.0),
                    "vwap": _safe_last("vwap", 0.0),
                }

            current_price = float(live_indicators.get("close", 0.0))
            engine = self._get_active_engine()
            portfolio = self._get_portfolio_snapshot(engine, current_price=current_price)
            try:
                total_value = self._estimate_equity_usdt(portfolio, current_price=current_price)
            except Exception:
                total_value = float(portfolio.get("cash_balance", 0.0) or 0.0)
            baseline_cash: Optional[float] = None
            if engine is self.paper_trading_engine:
                try:
                    params = self.state_manager.load_parameters() or {}
                    baseline_cash = float(params.get("paper_initial_cash", self.config.get("trading", {}).get("initial_cash", 10_000.0)))
                except Exception:
                    try:
                        baseline_cash = float(self.config.get("trading", {}).get("initial_cash", 10_000.0))
                    except Exception:
                        baseline_cash = 10_000.0

            pnl_since_reset = (float(total_value) - float(baseline_cash)) if baseline_cash is not None else None
            portfolio = {
                **portfolio,
                "total_value": total_value,
                "initial_cash": baseline_cash,
                "pnl_since_reset": pnl_since_reset,
            }

            gate = self._evaluate_gate(now=now, indicators=self.last_indicator_payload or {})

            self._maybe_save_equity_snapshot(now=now, portfolio=portfolio)
            self._update_risk_pause(now=now, current_equity=float(portfolio.get("total_value", 0.0) or 0.0))

            logging.info(f"Live indicators for dashboard: {live_indicators}")
            logging.info(f"Current price for dashboard: {current_price}")

            return {
                "symbol": self.symbol,
                "interval": self.interval,
                "execution": self.get_execution_status(),
                "ml_signal": self.latest_ml_context,
                "last_decisions": self.last_decisions,
                "strategy": self.strategy_manager.get_status_snapshot(),
                "probability_flag": probability,
                "last_daily_summary": current_metrics,  # Use current metrics for real-time updates
                "rl_result": self.last_rl_result,
                "ai_response": self.last_ai_response,
                "volatility_regime": self._calculate_volatility_regime(),
                "ml_last_training": self.last_ml_training_summary,
                "validation_mode": self.validation_mode,
                "training_progress": self.training_progress,
                "training_status": self.training_status,
                "is_training": self.is_training,
                "live_indicators": live_indicators,
                "current_price": current_price,
                "pnl_series": pnl_series,
                "portfolio": portfolio,
                "gate": gate,
                "fear_greed": self._fear_greed,
                "sentiment_policy": self._sentiment_policy,
                "auto_tune": self._auto_tune_state,
                "backtest": self._backtest_state,
                "risk_pause": self._risk_pause,
                "manual_pause": self._manual_pause,
                "equity_day": self._get_equity_day_status(now=now),
                "last_update": self.last_indicator_update.isoformat() + "Z" if self.last_indicator_update else "",
            }

    def _update_manual_pause(self, *, now: datetime) -> None:
        """Auto-clear manual pause when until_utc expires."""
        if not (self._manual_pause or {}).get("paused"):
            return
        until = (self._manual_pause or {}).get("until_utc")
        if not until:
            return
        if not isinstance(until, str):
            return
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except Exception:
            return
        try:
            now_aware = now
            if until_dt.tzinfo is not None:
                now_aware = now.replace(tzinfo=until_dt.tzinfo)
            if now_aware >= until_dt:
                self._manual_pause = {}
                try:
                    self.state_manager.save_parameters({"manual_pause": self._manual_pause})
                except Exception:
                    pass
        except Exception:
            return

    def _get_equity_day_status(self, *, now: datetime) -> Dict[str, Any]:
        risk_cfg = self.config.get("risk", {}) or {}
        tz_name = str(risk_cfg.get("pause_timezone") or self.config.get("general", {}).get("timezone") or "UTC")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")

        now_local = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        local_date = now_local.date()
        start_local = datetime.combine(local_date, time_cls(0, 0)).replace(tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

        summary = self.state_manager.get_equity_range_summary(start_utc, end_utc, symbol=self.symbol)
        start_val = summary.get("start_value")
        peak_val = summary.get("peak_value")
        latest_val = summary.get("latest_value")
        summary["timezone"] = tz_name
        summary["local_date"] = local_date.isoformat()
        if start_val is None or peak_val is None or latest_val is None:
            return summary
        summary["daily_pnl"] = float(latest_val) - float(start_val)
        summary["drawdown"] = float(peak_val) - float(latest_val)
        return summary

    def _update_risk_pause(self, *, now: datetime, current_equity: float) -> None:
        risk_cfg = self._get_effective_risk_cfg() or {}
        if not bool(risk_cfg.get("circuit_breaker_enabled", True)):
            return

        # Auto-clear pause when it expires (default next UTC day).
        if self._risk_pause.get("paused"):
            until = self._risk_pause.get("until_utc")
            if until and isinstance(until, str):
                try:
                    until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
                    if datetime.utcnow().replace(tzinfo=until_dt.tzinfo) >= until_dt:
                        self._risk_pause = {}
                        try:
                            self.state_manager.save_parameters({"risk_pause": self._risk_pause})
                        except Exception:
                            pass
                except Exception:
                    pass
            return

        day_status = self._get_equity_day_status(now=now)
        start_val = day_status.get("start_value")
        peak_val = day_status.get("peak_value")

        # If we don't have snapshots yet, use current equity as baseline.
        if start_val is None:
            start_val = float(current_equity)
        if peak_val is None:
            peak_val = float(current_equity)
        peak_val = max(float(peak_val), float(current_equity))

        daily_pnl = float(current_equity) - float(start_val)
        drawdown = float(peak_val) - float(current_equity)

        max_daily_loss = float(risk_cfg.get("max_daily_loss_usdt", 0.0) or 0.0)
        max_drawdown = float(risk_cfg.get("max_drawdown_usdt", 0.0) or 0.0)

        triggered = None
        if max_daily_loss > 0 and daily_pnl <= -max_daily_loss:
            triggered = "max_daily_loss"
        if max_drawdown > 0 and drawdown >= max_drawdown:
            triggered = triggered or "max_drawdown"

        if not triggered:
            return

        # Pause until next UTC day boundary by default.
        until_next_day = bool(risk_cfg.get("pause_until_next_day", True))
        if until_next_day:
            tz_name = str(risk_cfg.get("pause_timezone") or self.config.get("general", {}).get("timezone") or "UTC")
            try:
                tz = ZoneInfo(tz_name)
            except Exception:
                tz = ZoneInfo("UTC")
            now_local = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
            next_local_day = now_local.date() + timedelta(days=1)
            until_local = datetime.combine(next_local_day, time_cls(0, 0)).replace(tzinfo=tz)
            until_dt = until_local.astimezone(ZoneInfo("UTC"))
            until_iso = until_dt.isoformat().replace("+00:00", "Z")
        else:
            until_iso = ""

        self._risk_pause = {
            "paused": True,
            "reason": triggered,
            "triggered_at_utc": now.isoformat() + "Z",
            "until_utc": until_iso or None,
            "timezone": str(risk_cfg.get("pause_timezone") or self.config.get("general", {}).get("timezone") or "UTC"),
            "metrics": {
                "equity": float(current_equity),
                "day_start_equity": float(start_val),
                "day_peak_equity": float(peak_val),
                "daily_pnl": float(daily_pnl),
                "drawdown": float(drawdown),
                "max_daily_loss_usdt": max_daily_loss,
                "max_drawdown_usdt": max_drawdown,
            },
        }
        try:
            self.state_manager.save_parameters({"risk_pause": self._risk_pause})
        except Exception:
            pass

    def _maybe_save_equity_snapshot(self, *, now: datetime, portfolio: Dict[str, Any]) -> None:
        try:
            seconds = float((self.config.get("trading", {}) or {}).get("equity_snapshot_seconds", 30) or 30)
        except Exception:
            seconds = 30.0
        if seconds <= 0:
            return
        if self._last_equity_snapshot_at and (now - self._last_equity_snapshot_at).total_seconds() < seconds:
            return
        try:
            payload = {
                "snapshot_time": now.isoformat() + "Z",
                "symbol": self.symbol,
                "total_value": float(portfolio.get("total_value", 0.0) or 0.0),
                "cash_balance": float(portfolio.get("cash_balance", 0.0) or 0.0),
                "realized_pnl": float(portfolio.get("realized_pnl", 0.0) or 0.0),
                "open_positions_count": len(portfolio.get("open_positions") or []),
                "metadata": {"interval": self.interval},
            }
            self.state_manager.save_equity_snapshot(payload)
            self._last_equity_snapshot_at = now
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _execute_decisions(
        self,
        decisions: List[Dict[str, Any]],
        *,
        now: Optional[datetime] = None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = now or datetime.utcnow()
        try:
            self._update_manual_pause(now=now)
        except Exception:
            pass
        current_price = float((indicators or {}).get("closes", [0.0])[-1] or 0.0)
        engine = self._get_active_engine()
        portfolio = self._get_portfolio_snapshot(engine, current_price=current_price)
        equity = self._estimate_equity_usdt(portfolio, current_price=current_price)
        strategy = self.strategy_manager.get_active_strategy()
        risk_cfg = self._get_effective_risk_cfg()
        trading_cfg = self.config.get("trading", {})
        risk_pct_default = float(risk_cfg.get("risk_pct_per_trade", risk_cfg.get("risk_per_trade_pct", 0.01)))
        max_position_usdt = float(risk_cfg.get("max_position_size_usdt", trading_cfg.get("max_position_size_usdt", 100.0)))
        if self.execution_mode == "live":
            live_cfg = self._get_live_cfg()
            try:
                max_notional = float(live_cfg.get("max_notional_usdt_per_trade", 0.0) or 0.0)
                if max_notional > 0:
                    max_position_usdt = min(max_position_usdt, max_notional)
            except Exception:
                pass

        for decision in decisions:
            action = decision.get("decision")
            price = decision.get("price")
            metadata = decision.get("metadata") or {}
            size_hint = float(decision.get("suggested_size", 0.0))
            if action not in {"buy", "sell"} or not price:
                continue

            if action == "buy":
                # If there's an existing short (quantity < 0), allow BUY to close/reduce it.
                if (self._manual_pause or {}).get("paused"):
                    self._record_decision_event(decision, now=now, executed=False, blocked_reason="manual_pause", extra={"manual_pause": self._manual_pause})
                    continue
                if self.execution_mode == "live" and self._risk_pause.get("paused"):
                    self._record_decision_event(decision, now=now, executed=False, blocked_reason="risk_pause", extra={"risk_pause": self._risk_pause})
                    continue
                gate = self._evaluate_gate(now=now, indicators=indicators or {})
                if gate.get("enabled", True) and not gate.get("can_enter", True):
                    logging.info("BUY blocked by trading gate: %s", gate.get("reasons"))
                    self._record_decision_event(decision, now=now, executed=False, blocked_reason="gate_blocked", extra={"gate": gate})
                    continue
                exploration = bool(metadata.get("exploration", False))
                if self._should_throttle_buy(now, exploration=exploration):
                    self._record_decision_event(decision, now=now, executed=False, blocked_reason="throttle")
                    continue
                stop_loss_price, take_profit_price = self._resolve_stop_take_prices(
                    entry_price=float(price),
                    indicators=indicators or {},
                    decision_metadata=metadata,
                )
                risk_pct = float(strategy.config.get("risk_pct_per_trade", risk_pct_default))
                try:
                    meta_risk = metadata.get("risk_pct_per_trade")
                    if meta_risk is not None:
                        risk_pct = float(meta_risk)
                except Exception:
                    pass

                sp = self._sentiment_policy or self._compute_sentiment_policy()
                risk_mult = float(((sp.get("multipliers") or {}).get("risk_pct_multiplier")) or 1.0)
                risk_pct = max(0.0, risk_pct * max(0.0, risk_mult))
                sized = None
                risk_manager = getattr(self.strategy_manager, "risk_manager", None)
                if risk_manager is not None:
                    try:
                        sized = risk_manager.size_position_for_risk(
                            equity_usdt=equity,
                            entry_price=float(price),
                            stop_loss_price=stop_loss_price,
                            risk_pct_per_trade=risk_pct,
                            max_position_size_usdt=max_position_usdt,
                            max_quantity=size_hint if size_hint > 0 else None,
                        )
                    except Exception:
                        logging.exception("Risk sizing failed; falling back to strategy suggested size.")

                sized_qty = float(getattr(sized, "quantity", 0.0)) if sized is not None else size_hint
                order_metadata = dict(metadata)
                # Ensure trigger info is carried into trade metadata for later tooltips.
                if "trigger" not in order_metadata:
                    try:
                        first_reason = (decision.get("reasons") or [None])[0]
                    except Exception:
                        first_reason = None
                    if first_reason:
                        order_metadata["trigger"] = str(first_reason)
                if "reason" not in order_metadata:
                    try:
                        first_reason = (decision.get("reasons") or [None])[0]
                    except Exception:
                        first_reason = None
                    if first_reason:
                        order_metadata["reason"] = str(first_reason)
                # Attach market context for execution realism (spread/dynamic slippage).
                try:
                    atr_series = (indicators or {}).get("atr") or []
                    atr_val = float(atr_series[-1]) if atr_series else 0.0
                except Exception:
                    atr_val = 0.0
                try:
                    vols = (indicators or {}).get("volumes") or []
                    vol_val = float(vols[-1]) if vols else 0.0
                except Exception:
                    vol_val = 0.0
                try:
                    vsma = (indicators or {}).get("volume_sma") or []
                    vsma_val = float(vsma[-1]) if vsma else 0.0
                except Exception:
                    vsma_val = 0.0
                order_metadata.update(
                    {
                        "stop_loss": stop_loss_price,
                        "take_profit": take_profit_price,
                        "risk_pct_per_trade": risk_pct,
                        "exploration": exploration,
                        "sentiment": sp,
                        "market_context": {
                            "atr": atr_val,
                            "volume": vol_val,
                            "volume_sma": vsma_val,
                            "volatility_regime": self._calculate_volatility_regime(),
                        },
                        "equity_usdt": equity,
                        "sizing": {
                            "method": "risk",
                            "quantity": sized_qty,
                            "risk_amount_usdt": getattr(sized, "risk_amount_usdt", None),
                            "stop_distance_usdt": getattr(sized, "stop_distance_usdt", None),
                            "capped_by_max_notional": getattr(sized, "capped_by_max_notional", None),
                            "max_position_size_usdt": max_position_usdt,
                            "max_quantity_hint": size_hint if size_hint > 0 else None,
                        },
                        "timestamp_utc": now.isoformat() + "Z",
                    }
                )

                # Enforce max_position_usdt against the *total* open exposure (not per-order),
                # to avoid pyramiding beyond the configured cap.
                if sized_qty > 0 and max_position_usdt > 0 and float(price) > 0:
                    try:
                        current_qty = 0.0
                        positions_now = engine.get_open_positions()
                        pos_now = next((p for p in positions_now if p.get("symbol") == self.symbol), None)
                        current_qty = float((pos_now or {}).get("quantity", 0.0) or 0.0)
                    except Exception:
                        current_qty = 0.0
                    current_notional = max(0.0, current_qty) * float(price)
                    remaining_notional = float(max_position_usdt) - float(current_notional)
                    if remaining_notional <= 0:
                        self._record_decision_event(
                            decision,
                            now=now,
                            executed=False,
                            blocked_reason="max_position",
                            extra={"max_position_usdt": float(max_position_usdt), "current_notional_usdt": float(current_notional)},
                        )
                        continue
                    max_additional_qty = float(remaining_notional) / float(price)
                    if max_additional_qty > 0 and sized_qty > max_additional_qty:
                        sized_qty = max_additional_qty
                        try:
                            sizing_meta = order_metadata.get("sizing") if isinstance(order_metadata.get("sizing"), dict) else {}
                            sizing_meta.update(
                                {
                                    "quantity": sized_qty,
                                    "capped_by_existing_position": True,
                                    "current_notional_usdt": float(current_notional),
                                    "remaining_notional_usdt": float(remaining_notional),
                                }
                            )
                            order_metadata["sizing"] = sizing_meta
                        except Exception:
                            pass
                if sized_qty > 0:
                    order = SimulatedOrder(
                        symbol=self.symbol,
                        side="BUY",
                        quantity=sized_qty,
                        price=float(price),
                        metadata=order_metadata,
                    )
                    try:
                        record = engine.submit_order(order)
                        self._last_buy_time = now
                        self._record_decision_event(decision, now=now, executed=True, trade_record=record, extra={"sentiment": sp, "gate": gate})
                    except OrderRejected as e:
                        logging.info("BUY rejected: %s", str(e))
                        self._record_decision_event(
                            decision,
                            now=now,
                            executed=False,
                            blocked_reason=str(e.reason or "rejected"),
                            extra={"rejection": e.details or {}, "sentiment": sp, "gate": gate},
                        )
                    except Exception:
                        logging.exception("Failed to execute BUY order.")
                        self._record_decision_event(decision, now=now, executed=False, blocked_reason="execution_error", extra={"error": "buy_failed"})
                continue

            # SELL: always prefer to close (or reduce) existing exposure; do not block exits.
            available_qty = 0.0
            try:
                positions = engine.get_open_positions()
                pos = next((p for p in positions if p.get("symbol") == self.symbol), None)
                available_qty = float((pos or {}).get("quantity", 0.0) or 0.0)
            except Exception:
                available_qty = 0.0
            min_qty = self._get_min_qty_for_symbol(engine, self.symbol)

            entry_type = str((metadata or {}).get("entry_type") or "").lower()
            short_entry = entry_type == "short" and self.execution_mode != "live"

            # If flat and this is a short entry, allow it; otherwise require an open long.
            if not short_entry and (available_qty <= 0.0 or (min_qty > 0 and available_qty < min_qty)):
                logging.info("Ignoring SELL signal (no open position).")
                self._record_decision_event(decision, now=now, executed=False, blocked_reason="no_position")
                continue

            # Cap to available position size to avoid over-sell.
            sell_qty = float(size_hint) if size_hint > 0 else available_qty
            if short_entry and available_qty <= 0.0:
                sell_qty = float(size_hint) if size_hint > 0 else float(risk_cfg.get("default_short_qty", 0.001) or 0.001)
            else:
                sell_qty = min(sell_qty, available_qty)
                # If we'd leave an untradeable dust remainder, just close fully.
                if min_qty > 0 and (available_qty - sell_qty) < min_qty:
                    sell_qty = available_qty
                if min_qty > 0 and sell_qty < min_qty:
                    logging.info("Ignoring SELL signal (position dust below min order qty).")
                    self._record_decision_event(decision, now=now, executed=False, blocked_reason="no_position")
                    continue

            if sell_qty > 0:
                order_meta = dict(metadata)
                if "trigger" not in order_meta:
                    try:
                        first_reason = (decision.get("reasons") or [None])[0]
                    except Exception:
                        first_reason = None
                    if first_reason:
                        order_meta["trigger"] = str(first_reason)
                if "reason" not in order_meta:
                    try:
                        first_reason = (decision.get("reasons") or [None])[0]
                    except Exception:
                        first_reason = None
                    if first_reason:
                        order_meta["reason"] = str(first_reason)
                order = SimulatedOrder(
                    symbol=self.symbol,
                    side="SELL",
                    quantity=float(sell_qty),
                    price=float(price),
                    metadata=order_meta,
                )
                try:
                    record = engine.submit_order(order)
                    self._record_decision_event(decision, now=now, executed=True, trade_record=record)
                    if engine is self.paper_trading_engine:
                        self._maybe_auto_tune(now=now, last_trade_record=record)
                except OrderRejected as e:
                    logging.info("SELL rejected: %s", str(e))
                    self._record_decision_event(
                        decision,
                        now=now,
                        executed=False,
                        blocked_reason=str(e.reason or "rejected"),
                        extra={"rejection": e.details or {}},
                    )
                except Exception:
                    logging.exception("Failed to execute SELL order.")
                    self._record_decision_event(decision, now=now, executed=False, blocked_reason="execution_error", extra={"error": "sell_failed"})

    @staticmethod
    def _estimate_equity_usdt(portfolio: Dict[str, Any], *, current_price: float) -> float:
        cash = float(portfolio.get("cash_balance", 0.0))
        equity = cash
        for pos in portfolio.get("open_positions", []) or []:
            try:
                equity += float(pos.get("quantity", 0.0)) * float(current_price)
            except Exception:
                continue
        return float(equity)

    def _should_throttle_buy(self, now: datetime, *, exploration: bool = False) -> bool:
        trading_cfg = self.config.get("trading", {})
        cooldown_seconds = float(trading_cfg.get("cooldown_seconds", 0.0) or 0.0)
        max_trades_per_hour = int(trading_cfg.get("max_trades_per_hour", 0) or 0)

        sp = self._sentiment_policy or self._compute_sentiment_policy()
        mult = sp.get("multipliers") or {}
        try:
            max_trades_per_hour = int(round(max_trades_per_hour * float(mult.get("max_trades_per_hour_multiplier", 1.0))))
        except Exception:
            pass
        try:
            cooldown_seconds = float(cooldown_seconds) * float(mult.get("cooldown_seconds_multiplier", 1.0))
        except Exception:
            pass

        if exploration and bool(trading_cfg.get("exploration_ignore_cooldown", True)):
            cooldown_seconds = 0.0

        if cooldown_seconds > 0:
            if self._last_buy_time is None:
                self._last_buy_time = self.state_manager.get_last_trade_time(symbol=self.symbol, side="BUY")
            if self._last_buy_time is not None:
                last_buy = self._last_buy_time
                try:
                    # Normalize to naive so subtraction does not mix aware/naive datetimes.
                    if last_buy.tzinfo is not None:
                        last_buy = last_buy.replace(tzinfo=None)
                except Exception:
                    pass
                elapsed = (now - last_buy).total_seconds()
                if elapsed < cooldown_seconds:
                    logging.info("BUY throttled by cooldown (%.1fs < %.1fs).", elapsed, cooldown_seconds)
                    return True

        if max_trades_per_hour > 0:
            since = now - timedelta(hours=1)
            try:
                recent = self.state_manager.count_trades_since(since, symbol=self.symbol)
            except Exception:
                logging.exception("Unable to count recent trades for throttling; allowing BUY.")
                return False
            if recent >= max_trades_per_hour:
                logging.info("BUY throttled by max_trades_per_hour (%s/%s).", recent, max_trades_per_hour)
                return True

        return False

    def _compute_sentiment_policy(self) -> Dict[str, Any]:
        cfg = self.config.get("sentiment_policy", {}) or {}
        if not bool(cfg.get("enabled", True)):
            return {"regime": "disabled", "multipliers": {"risk_pct_multiplier": 1.0, "max_trades_per_hour_multiplier": 1.0, "cooldown_seconds_multiplier": 1.0}}

        fg = self._fear_greed if isinstance(self._fear_greed, dict) else {}
        if fg.get("enabled") is False:
            return {"regime": "no_data", "multipliers": {"risk_pct_multiplier": 1.0, "max_trades_per_hour_multiplier": 1.0, "cooldown_seconds_multiplier": 1.0}}
        if fg.get("error"):
            return {"regime": "error", "multipliers": {"risk_pct_multiplier": 1.0, "max_trades_per_hour_multiplier": 1.0, "cooldown_seconds_multiplier": 1.0}, "error": fg.get("error")}

        try:
            value = int(float(fg.get("value")))
        except Exception:
            return {"regime": "unknown", "multipliers": {"risk_pct_multiplier": 1.0, "max_trades_per_hour_multiplier": 1.0, "cooldown_seconds_multiplier": 1.0}}

        fear_th = int(cfg.get("fear_threshold", 25))
        greed_th = int(cfg.get("greed_threshold", 75))
        if value <= fear_th:
            regime = "fear"
        elif value >= greed_th:
            regime = "greed"
        else:
            regime = "neutral"

        mult = cfg.get(regime, {}) or {}
        def _m(key: str, default: float) -> float:
            try:
                return float(mult.get(key, default))
            except Exception:
                return float(default)

        return {
            "regime": regime,
            "value": value,
            "source": fg.get("source"),
            "timestamp_utc": fg.get("timestamp_utc") or fg.get("updated_at"),
            "multipliers": {
                "risk_pct_multiplier": _m("risk_pct_multiplier", 1.0),
                "max_trades_per_hour_multiplier": _m("max_trades_per_hour_multiplier", 1.0),
                "cooldown_seconds_multiplier": _m("cooldown_seconds_multiplier", 1.0),
            },
        }

    def _is_exploration_mode(self) -> bool:
        trading_cfg = self.config.get("trading", {}) or {}
        if not bool(trading_cfg.get("exploration_enabled", False)):
            return False
        try:
            strategy = self.strategy_manager.get_active_strategy()
            strat_cfg = getattr(strategy, "config", {}) or {}
            if not bool(strat_cfg.get("allow_exploration", False)):
                return False
        except Exception:
            return False
        target = int(trading_cfg.get("exploration_until_total_trades", 50) or 50)
        if target <= 0:
            return False
        try:
            total = self.state_manager.count_total_trades(symbol=self.symbol)
        except Exception:
            total = 0
        return total < target

    def _evaluate_gate(self, *, now: datetime, indicators: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_learning_mode():
            self._gate_status = {
                "enabled": True,
                "can_enter": True,
                "reasons": ["Learning mode (paper): entries allowed."],
                "updated_at": now.isoformat() + "Z",
                "snapshot": {"mode": "learning"},
            }
            return self._gate_status
        gate_cfg = self.config.get("gate", {})
        risk_cfg = self._get_effective_risk_cfg()
        try:
            date_str = now.date().isoformat()
            trades_today = self.state_manager.get_trades_for_day(date_str, symbol=self.symbol)
            metrics_today = self.evaluator.compile_daily_metrics(trades_today)
            trades_today_count = int(metrics_today.get("trades_count", len(trades_today)) or 0)
            volatility_regime = self._calculate_volatility_regime()
            decision = evaluate_trading_gate(
                now=now,
                gate_cfg=gate_cfg,
                risk_cfg=risk_cfg,
                metrics_today=metrics_today,
                trades_today_count=trades_today_count,
                volatility_regime=float(volatility_regime),
                validation_mode=bool(self.validation_mode),
                is_training=bool(self.is_training),
                fear_greed_value=(int(self._fear_greed.get("value")) if isinstance(self._fear_greed, dict) and self._fear_greed.get("value") is not None else None),
            )
            self._gate_status = decision.to_dict()
        except Exception:
            logging.exception("Trading gate evaluation failed; defaulting to allow entries.")
            self._gate_status = {"enabled": True, "can_enter": True, "reasons": ["gate_eval_error"], "updated_at": now.isoformat() + "Z", "snapshot": {}}
        return self._gate_status

    def _maybe_auto_tune(self, *, now: datetime, last_trade_record: Optional[Dict[str, Any]] = None) -> None:
        trading_cfg = self.config.get("trading", {}) or {}
        auto_cfg = self.config.get("auto_tune", {}) or {}
        enabled = bool(auto_cfg.get("enabled", trading_cfg.get("auto_tune_enabled", False)))
        if not enabled:
            return
        # Only tune on closed trades (SELL) to avoid overreacting.
        if last_trade_record and str(last_trade_record.get("side", "")).upper() != "SELL":
            return

        lookback = int(trading_cfg.get("auto_tune_lookback_closed_trades", 30) or 30)
        every = int(trading_cfg.get("auto_tune_every_closed_trades", 5) or 5)
        if every <= 0:
            return

        try:
            recent_sells = self.state_manager.get_recent_trades(limit=max(lookback, 10), symbol=self.symbol, side="SELL")
        except Exception:
            return

        closed_count = len(recent_sells)
        if closed_count <= 0:
            return
        # Throttle tuning frequency
        if closed_count < self._last_tuned_closed_count + every:
            return

        strategy_name = self.strategy_manager.get_active_strategy_name()
        current_cfg = dict(self.strategy_manager.get_active_strategy().config)
        exploration = self._is_exploration_mode()
        try:
            recent_per_hour = self.state_manager.count_trades_since(now - timedelta(hours=1), symbol=self.symbol)
            trades_per_hour = float(recent_per_hour)
        except Exception:
            trades_per_hour = None
        try:
            decision_stats_hour = self.state_manager.get_decision_stats_since(now - timedelta(hours=1), symbol=self.symbol)
        except Exception:
            decision_stats_hour = None

        result = auto_tune_strategy(
            now=now,
            strategy_name=strategy_name,
            current_config=current_cfg,
            recent_closed_trades=list(reversed(recent_sells[:lookback])),
            auto_cfg=auto_cfg,
            exploration_active=exploration,
            trades_per_hour=trades_per_hour,
            decision_stats=decision_stats_hour,
        )
        self._auto_tune_state = result.to_dict()
        self._last_tuned_closed_count = closed_count

        if result.changed:
            # Persist only overrides relevant to the active strategy.
            self.strategy_manager.update_strategy_config(strategy_name, result.after)
            try:
                self.state_manager.save_parameters({"auto_tune_last": self._auto_tune_state})
            except Exception:
                pass

        # Also tune global entry controls based on blocked reasons/execution rate.
        try:
            self._maybe_tune_controls(now=now, decision_stats=decision_stats_hour or {})
        except Exception:
            pass
        try:
            self._maybe_tune_gate(now=now, decision_stats=decision_stats_hour or {})
        except Exception:
            pass

    def _maybe_auto_tune_from_signals(self, *, now: datetime) -> None:
        """Use decision_events (including blocked signals) to tune even before any closed trades exist."""
        trading_cfg = self.config.get("trading", {}) or {}
        auto_cfg = self.config.get("auto_tune", {}) or {}
        enabled = bool(auto_cfg.get("enabled", trading_cfg.get("auto_tune_enabled", False)))
        if not enabled:
            return

        # Throttle how often we tune based on signals.
        min_minutes = int(auto_cfg.get("controls_min_interval_minutes", 5) or 5)
        if self._last_signal_tune_at and (now - self._last_signal_tune_at).total_seconds() < min_minutes * 60:
            return

        try:
            last_id = self.state_manager.get_last_decision_event_id()
        except Exception:
            last_id = 0

        min_new_events = int(auto_cfg.get("controls_min_new_events", 20) or 20)
        if last_id < (self._last_signal_tune_event_id + min_new_events):
            return

        try:
            decision_stats_hour = self.state_manager.get_decision_stats_since(now - timedelta(hours=1), symbol=self.symbol)
        except Exception:
            decision_stats_hour = {}

        min_events = int(auto_cfg.get("controls_min_events_hour", 10) or 10)
        if int(decision_stats_hour.get("total", 0) or 0) < min_events:
            return

        strategy_name = self.strategy_manager.get_active_strategy_name()
        current_cfg = dict(self.strategy_manager.get_active_strategy().config)
        exploration = self._is_exploration_mode()

        try:
            recent_per_hour = self.state_manager.count_trades_since(now - timedelta(hours=1), symbol=self.symbol)
            trades_per_hour = float(recent_per_hour)
        except Exception:
            trades_per_hour = None

        try:
            recent_sells = self.state_manager.get_recent_trades(limit=int(trading_cfg.get("auto_tune_lookback_closed_trades", 30) or 30), symbol=self.symbol, side="SELL")
        except Exception:
            recent_sells = []

        result = auto_tune_strategy(
            now=now,
            strategy_name=strategy_name,
            current_config=current_cfg,
            recent_closed_trades=list(reversed(recent_sells)),
            auto_cfg=auto_cfg,
            exploration_active=exploration,
            trades_per_hour=trades_per_hour,
            decision_stats=decision_stats_hour,
        )

        self._auto_tune_state = result.to_dict()
        if result.changed:
            self.strategy_manager.update_strategy_config(strategy_name, result.after)
            try:
                self.state_manager.save_parameters({"auto_tune_last": self._auto_tune_state})
            except Exception:
                pass

        try:
            self._maybe_tune_controls(now=now, decision_stats=decision_stats_hour or {})
        except Exception:
            pass
        try:
            self._maybe_tune_gate(now=now, decision_stats=decision_stats_hour or {})
        except Exception:
            pass

        self._last_signal_tune_event_id = last_id
        self._last_signal_tune_at = now

    def _maybe_tune_controls(self, *, now: datetime, decision_stats: Dict[str, Any]) -> None:
        trading_cfg = self.config.get("trading", {}) or {}
        auto_cfg = self.config.get("auto_tune", {}) or {}
        if not bool(auto_cfg.get("enabled", trading_cfg.get("auto_tune_enabled", False))):
            return

        # Throttle how often we change global controls.
        min_minutes = int(auto_cfg.get("controls_min_interval_minutes", 5) or 5)
        if self._last_control_tune_at and (now - self._last_control_tune_at).total_seconds() < min_minutes * 60:
            return

        try:
            last_id = self.state_manager.get_last_decision_event_id()
        except Exception:
            last_id = 0
        min_new_events = int(auto_cfg.get("controls_min_new_events", 20) or 20)
        if last_id < (self._last_control_tune_event_id + min_new_events):
            return

        total = int(decision_stats.get("total", 0) or 0)
        executed = int(decision_stats.get("executed", 0) or 0)
        if total < int(auto_cfg.get("controls_min_events_hour", 10) or 10):
            return

        exec_rate = executed / max(total, 1)
        target_exec = float(auto_cfg.get("target_execution_rate", 0.6) or 0.6)
        blocked = decision_stats.get("blocked_by_reason") or {}
        throttle_blocks = int(blocked.get("throttle", 0) or 0)

        max_trades = int(trading_cfg.get("max_trades_per_hour", 0) or 0)
        cooldown = float(trading_cfg.get("cooldown_seconds", 0.0) or 0.0)
        changed = False

        if exec_rate < target_exec and throttle_blocks > 0:
            # Make throttle less restrictive.
            if max_trades > 0:
                new_max = min(int(round(max_trades * 1.25)), int(auto_cfg.get("max_trades_per_hour_cap", 240) or 240))
                if new_max != max_trades:
                    trading_cfg["max_trades_per_hour"] = new_max
                    changed = True
            if cooldown > 0:
                new_cd = max(0.0, cooldown * 0.8)
                if abs(new_cd - cooldown) > 0.5:
                    trading_cfg["cooldown_seconds"] = round(new_cd, 2)
                    changed = True

        if changed:
            self._last_control_tune_event_id = last_id
            self._last_control_tune_at = now
            self.config.setdefault("trading", {}).update(trading_cfg)
            try:
                self.state_manager.save_parameters({"trading_overrides": {"max_trades_per_hour": trading_cfg.get("max_trades_per_hour"), "cooldown_seconds": trading_cfg.get("cooldown_seconds")}})
            except Exception:
                pass
            # Surface in auto_tune status for UI visibility.
            self._auto_tune_state.setdefault("controls", {})
            self._auto_tune_state["controls"] = {
                "execution_rate_hour": round(exec_rate, 4),
                "target_execution_rate": target_exec,
                "applied": {"max_trades_per_hour": trading_cfg.get("max_trades_per_hour"), "cooldown_seconds": trading_cfg.get("cooldown_seconds")},
                "updated_at": now.isoformat() + "Z",
            }

    def _maybe_tune_gate(self, *, now: datetime, decision_stats: Dict[str, Any]) -> None:
        """Relax the entry gate when it is the dominant block reason (bounded, simulation only)."""
        trading_cfg = self.config.get("trading", {}) or {}
        auto_cfg = self.config.get("auto_tune", {}) or {}
        if not bool(auto_cfg.get("enabled", trading_cfg.get("auto_tune_enabled", False))):
            return

        min_minutes = int(auto_cfg.get("controls_min_interval_minutes", 5) or 5)
        if self._last_gate_tune_at and (now - self._last_gate_tune_at).total_seconds() < min_minutes * 60:
            return

        try:
            last_id = self.state_manager.get_last_decision_event_id()
        except Exception:
            last_id = 0
        min_new_events = int(auto_cfg.get("controls_min_new_events", 20) or 20)
        if last_id < (self._last_gate_tune_event_id + min_new_events):
            return

        total = int(decision_stats.get("total", 0) or 0)
        executed = int(decision_stats.get("executed", 0) or 0)
        if total < int(auto_cfg.get("controls_min_events_hour", 10) or 10):
            return

        blocked = decision_stats.get("blocked_by_reason") or {}
        gate_blocks = int(blocked.get("gate_blocked", 0) or 0)
        if gate_blocks <= 0:
            return

        exec_rate = executed / max(total, 1)
        target_exec = float(auto_cfg.get("target_execution_rate", 0.6) or 0.6)
        gate_ratio = gate_blocks / max(total, 1)
        if gate_ratio < 0.25 or exec_rate >= target_exec:
            return

        gate_cfg = dict(self.config.get("gate", {}) or {})
        before = dict(gate_cfg)

        def _clamp(value: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, value))

        def _clamp_int(value: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, value))

        # Reduce sample requirements so the system can start trading and collecting evidence.
        gate_cfg["min_trades_today"] = _clamp_int(int(gate_cfg.get("min_trades_today", 5) or 5) - 1, 0, 50)
        gate_cfg["min_trades_for_stats"] = _clamp_int(int(gate_cfg.get("min_trades_for_stats", 15) or 15) - 2, 0, 200)

        # Relax performance thresholds slightly.
        gate_cfg["min_sharpe"] = round(_clamp(float(gate_cfg.get("min_sharpe", 0.1) or 0.1) - 0.05, -0.2, 2.0), 4)
        gate_cfg["min_profit_factor"] = round(_clamp(float(gate_cfg.get("min_profit_factor", 1.05) or 1.05) - 0.02, 0.9, 3.0), 4)
        gate_cfg["min_expectancy"] = round(_clamp(float(gate_cfg.get("min_expectancy", 0.0) or 0.0) - 0.5, -50.0, 50.0), 4)

        # Widen the acceptable volatility regime range.
        gate_cfg["min_volatility_regime"] = round(_clamp(float(gate_cfg.get("min_volatility_regime", 0.0) or 0.0) - 0.05, 0.0, 2.0), 4)
        gate_cfg["max_volatility_regime"] = round(_clamp(float(gate_cfg.get("max_volatility_regime", 1.0) or 1.0) + 0.10, 0.05, 2.0), 4)

        if gate_cfg == before:
            return

        self._last_gate_tune_event_id = last_id
        self._last_gate_tune_at = now
        self.config.setdefault("gate", {}).update(gate_cfg)
        try:
            self.state_manager.save_parameters({"gate_overrides": gate_cfg})
        except Exception:
            pass

        self._auto_tune_state.setdefault("gate_controls", {})
        self._auto_tune_state["gate_controls"] = {
            "gate_block_ratio_hour": round(gate_ratio, 4),
            "execution_rate_hour": round(exec_rate, 4),
            "target_execution_rate": target_exec,
            "before": before,
            "applied": gate_cfg,
            "updated_at": now.isoformat() + "Z",
        }

    def _record_decision_event(
        self,
        decision: Dict[str, Any],
        *,
        now: datetime,
        executed: bool,
        blocked_reason: Optional[str] = None,
        trade_record: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            symbol = decision.get("symbol") or self.symbol
            payload = {
                "event_time": now.isoformat() + "Z",
                "symbol": str(symbol or "").upper(),
                "strategy": decision.get("strategy") or self.strategy_manager.get_active_strategy_name(),
                "decision": decision.get("decision"),
                "price": decision.get("price"),
                "suggested_size": decision.get("suggested_size"),
                "confidence": decision.get("confidence"),
                "executed": executed,
                "blocked_reason": blocked_reason,
                "trade_time": (trade_record or {}).get("trade_time") if trade_record else None,
                "metadata": {
                    "decision_metadata": decision.get("metadata") or {},
                    "reasons": decision.get("reasons") or [],
                    "extra": extra or {},
                },
            }
            self.state_manager.save_decision_event(payload)
        except Exception:
            # Never break the trading loop due to telemetry persistence.
            return

    def _record_blocked_decisions(self, decisions: List[Dict[str, Any]], *, now: datetime, reason: str) -> None:
        for d in decisions:
            action = d.get("decision")
            if action in {"buy", "sell"}:
                self._record_decision_event(d, now=now, executed=False, blocked_reason=reason)

    def _update_fear_greed(self, now_utc_naive: datetime) -> None:
        cfg = self.config.get("sentiment", {}) or {}
        if not bool(cfg.get("enabled", False)):
            self._fear_greed = {"enabled": False}
            return
        provider = str(cfg.get("provider", "alternative_me") or "alternative_me").lower()
        if provider not in {"alternative_me", "alternative.me", "alternative", "cnn", "cnn_fng", "coinmarketcap", "cmc"}:
            self._fear_greed = {"enabled": True, "error": f"unsupported_provider:{provider}"}
            return
        # Recreate provider if config changed.
        try:
            if getattr(self._fear_greed_provider, "provider", None) != provider:
                self._fear_greed_provider = FearGreedProvider(
                    provider=provider,
                    coinmarketcap_api_key=str(cfg.get("coinmarketcap_api_key") or ""),
                    min_refresh_seconds=int(cfg.get("refresh_seconds", 900) or 900),
                )
        except Exception:
            pass
        try:
            from datetime import timezone

            now = now_utc_naive.replace(tzinfo=timezone.utc)
            snap = self._fear_greed_provider.get_latest(now=now)
            if snap is None:
                self._fear_greed = {"enabled": True, "error": "fetch_failed", "updated_at": now.isoformat().replace("+00:00", "Z")}
                return
            new_payload = {"enabled": True, **snap.to_dict()}
            if new_payload != self._fear_greed:
                self._fear_greed = new_payload
                try:
                    self.state_manager.save_parameters({"fear_greed_latest": self._fear_greed})
                except Exception:
                    pass
        except Exception:
            logging.exception("Failed to update Fear & Greed index.")
            self._fear_greed = {"enabled": True, "error": "exception", "updated_at": now_utc_naive.isoformat() + "Z"}

    def _resolve_stop_take_prices(
        self,
        *,
        entry_price: float,
        indicators: Dict[str, Any],
        decision_metadata: Dict[str, Any],
    ) -> tuple[float, float]:
        if entry_price <= 0:
            return (0.0, 0.0)

        strategy = self.strategy_manager.get_active_strategy()
        atr_series = (indicators or {}).get("atr") or []
        atr_val = float(atr_series[-1]) if atr_series else entry_price * 0.008

        # Strategy can provide explicit absolute levels (but we still enforce sane mins vs fees).
        stop_loss = decision_metadata.get("stop_loss") or decision_metadata.get("stop_loss_price")
        take_profit = decision_metadata.get("take_profit") or decision_metadata.get("take_profit_price")
        stop_loss_price = float(stop_loss) if stop_loss is not None else None
        take_profit_price = float(take_profit) if take_profit is not None else None

        if stop_loss_price is None or take_profit_price is None:
            stop_atr_mult = strategy.config.get("stop_atr_multiplier")
            profit_target_mult = strategy.config.get("profit_target_multiplier")
            if stop_atr_mult is not None and profit_target_mult is not None and atr_val > 0:
                stop_dist = atr_val * float(stop_atr_mult)
                if stop_loss_price is None:
                    stop_loss_price = max(0.0, entry_price - stop_dist)
                if take_profit_price is None:
                    take_profit_price = max(0.0, entry_price + stop_dist * float(profit_target_mult))
            else:
                # Default percentage based.
                stop_loss_pct = float(strategy.config.get("stop_loss_pct", 0.05))
                take_profit_pct = float(strategy.config.get("take_profit_pct", 0.10))
                if stop_loss_price is None:
                    stop_loss_price = max(0.0, entry_price * (1.0 - max(0.0, stop_loss_pct)))
                if take_profit_price is None:
                    take_profit_price = max(0.0, entry_price * (1.0 + max(0.0, take_profit_pct)))

        stop_loss_price = float(stop_loss_price or 0.0)
        take_profit_price = float(take_profit_price or 0.0)

        # Sanity: long entries should have stop below and TP above.
        if stop_loss_price <= 0.0 or stop_loss_price >= entry_price:
            stop_loss_price = max(0.0, entry_price * (1.0 - 0.002))
        if take_profit_price <= entry_price:
            take_profit_price = max(0.0, entry_price * (1.0 + 0.003))

        # Cost-aware guardrails: ensure TP/SL distances aren't smaller than execution costs.
        trading_cfg = self.config.get("trading", {}) or {}
        if bool(trading_cfg.get("enforce_cost_aware_exits", True)):
            try:
                fee_rate = float(trading_cfg.get("fee_rate", 0.0) or 0.0)
            except Exception:
                fee_rate = 0.0
            try:
                spread_bps = float(trading_cfg.get("spread_bps", 0.0) or 0.0)
            except Exception:
                spread_bps = 0.0
            try:
                base_slip = float(trading_cfg.get("slippage", 0.0) or 0.0)
            except Exception:
                base_slip = 0.0

            dyn_slip = 0.0
            if bool(trading_cfg.get("dynamic_slippage_enabled", True)) and atr_val > 0 and entry_price > 0:
                try:
                    slip_atr_mult = float(trading_cfg.get("slippage_atr_multiplier", 0.0) or 0.0)
                except Exception:
                    slip_atr_mult = 0.0
                atr_pct = float(atr_val) / float(entry_price)
                dyn_slip = max(0.0, atr_pct * max(0.0, slip_atr_mult))

            spread_pct = max(0.0, spread_bps) / 10_000.0
            total_cost_pct = max(0.0, (2.0 * fee_rate) + spread_pct + 2.0 * (max(0.0, base_slip) + max(0.0, dyn_slip)))

            # Minimum stop distance: at least either configured floor or the round-trip cost.
            min_stop_pct = None
            try:
                min_stop_pct = trading_cfg.get("min_stop_distance_pct")
            except Exception:
                min_stop_pct = None
            if min_stop_pct is None:
                try:
                    rm = getattr(self.strategy_manager, "risk_manager", None)
                    min_stop_pct = getattr(rm, "min_stop_distance_pct", None) if rm is not None else None
                except Exception:
                    min_stop_pct = None
            try:
                min_stop_pct_f = float(min_stop_pct) if min_stop_pct is not None else 0.0
            except Exception:
                min_stop_pct_f = 0.0
            min_stop_pct_f = max(0.0, float(min_stop_pct_f), float(total_cost_pct))
            min_stop_dist = entry_price * min_stop_pct_f
            if (entry_price - stop_loss_price) < min_stop_dist:
                stop_loss_price = max(0.0, entry_price - min_stop_dist)

            # Minimum take-profit distance: cover costs + edge multiplier.
            try:
                edge_mult = float(trading_cfg.get("take_profit_edge_multiplier", 1.5) or 1.5)
            except Exception:
                edge_mult = 1.5
            min_tp_pct = max(0.0, float(total_cost_pct) * max(1.0, edge_mult))
            min_tp_price = entry_price * (1.0 + min_tp_pct)
            if take_profit_price < min_tp_price:
                take_profit_price = min_tp_price

        return (float(stop_loss_price), float(take_profit_price))

    def _build_indicator_payload(self, ohlcv: List[Dict[str, Any]]) -> Dict[str, Any]:
        closes = [row["close"] for row in ohlcv]
        highs = [row["high"] for row in ohlcv]
        lows = [row["low"] for row in ohlcv]
        volumes = [row.get("volume", 0.0) for row in ohlcv]

        strategy = self.strategy_manager.get_active_strategy()
        cfg = getattr(strategy, "config", {}) or {}
        try:
            rsi_period = int(cfg.get("rsi_period", 14) or 14)
        except Exception:
            rsi_period = 14
        try:
            atr_period = int(cfg.get("atr_period", 14) or 14)
        except Exception:
            atr_period = 14
        try:
            vol_sma_period = int(cfg.get("volume_sma_period", 20) or 20)
        except Exception:
            vol_sma_period = 20
        try:
            ema_fast_period = int(cfg.get("ema_fast_period", 20) or 20)
        except Exception:
            ema_fast_period = 20
        try:
            ema_slow_period = int(cfg.get("ema_slow_period", 50) or 50)
        except Exception:
            ema_slow_period = 50
        try:
            boll_period = int(cfg.get("bollinger_period", 20) or 20)
        except Exception:
            boll_period = 20
        try:
            boll_std = float(cfg.get("bollinger_stddev", 2.0) or 2.0)
        except Exception:
            boll_std = 2.0

        rsi = technicals.compute_rsi(closes, period=rsi_period)
        macd = technicals.compute_macd(closes)
        bollinger = technicals.compute_bollinger_bands(closes, period=boll_period, num_std_dev=boll_std)
        atr = technicals.compute_atr(highs, lows, closes, period=atr_period)
        # Scalping indicators
        ohlc_tuples = [(row["open"], row["high"], row["low"], row["close"]) for row in ohlcv]
        doji = technicals.detect_doji(ohlc_tuples)
        engulfing = technicals.detect_engulfing(ohlc_tuples)
        volume_sma = technicals.compute_volume_sma(volumes, period=vol_sma_period)
        macd_histogram = technicals.compute_macd_histogram(closes)
        bollinger_width = technicals.compute_bollinger_width(closes, period=boll_period, std_dev=boll_std)
        features_df = feature_engineering.generate_features(ohlcv)
        vwap = technicals.compute_vwap(highs, lows, closes, volumes)
        payload = {
            "ohlcv": ohlcv,
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "rsi": rsi,
            "macd": macd,
            "bollinger": bollinger,
            "atr": atr,
            "doji": doji,
            "engulfing": engulfing,
            "volume_sma": volume_sma,
            "macd_histogram": macd_histogram,
            "bollinger_width": bollinger_width,
            "vwap": vwap,
            "features_df": features_df,
            "ema_fast": self._ema_series(closes, ema_fast_period),
            "ema_slow": self._ema_series(closes, ema_slow_period),
            "close": closes[-1] if closes else 0,
        }
        return payload

    def _build_ml_context(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ml_predictor:
            return {}
        features_df = indicators.get("features_df")
        if features_df is None or features_df.empty:
            return {}
        try:
            return self.ml_predictor.build_signal_context(features_df)
        except Exception:
            logging.exception("ML predictor failed; ignoring ML context for this cycle.")
            return {}

    def _calculate_volatility_regime(self) -> float:
        closes = self.last_indicator_payload.get("closes") or []
        highs = self.last_indicator_payload.get("highs") or []
        lows = self.last_indicator_payload.get("lows") or []
        if not closes or not highs or not lows:
            return 0.0
        atr = technicals.compute_atr(highs, lows, closes)
        if not atr:
            return 0.0
        price = closes[-1]
        atr_value = atr[-1]
        if price <= 0:
            return 0.0
        return round(min(1.0, max(0.0, atr_value / price)), 4)

    def _ensure_ml_retrained(self, now: datetime) -> None:
        """Train/retrain the ML model if the configured cadence dictates it."""
        with self._admin_lock:
            if not self.ml_trainer:
                return
            ml_cfg = self.config.get("ml", {})
            if not ml_cfg.get("retrain_daily", False):
                return
            last_date = self._get_last_ml_training_date()
            if last_date == now.date():
                return
            self._run_ml_training_cycle(now, ml_cfg)

    def _run_ml_training_cycle(self, now: datetime, ml_cfg: Dict[str, Any]) -> None:
        history_days = max(int(ml_cfg.get("history_days", 30)), 1)
        interval = ml_cfg.get("interval") or self.interval
        symbol = ml_cfg.get("symbol") or self.symbol
        horizon = max(int(ml_cfg.get("horizon", 1)), 1)
        candles = self._get_training_candles(symbol=symbol, interval=interval, history_days=history_days, now=now, limit=None)
        if not candles:
            logging.warning("ML retrain skipped because no OHLCV data was returned.")
            return
        features, labels = feature_engineering.build_dataset(candles, horizon=horizon)
        if features.empty or labels.empty:
            logging.warning("ML retrain skipped because feature matrix or labels were empty.")
            return
        model_name = ml_cfg.get("model_name") or f"auto_{symbol}_{interval}_{now.date().isoformat()}"
        # Walk-forward training is very expensive on large histories; use chronological split by default.
        use_walk_forward = bool(ml_cfg.get("walk_forward", False))
        if use_walk_forward and len(features) <= int(ml_cfg.get("walk_forward_max_rows", 10_000) or 10_000):
            try:
                summary = self.ml_trainer.walk_forward_train(features, labels, model_name=model_name)
            except Exception:
                logging.exception("Walk-forward ML training failed, falling back to simple training.")
                summary = self.ml_trainer.train(features, labels, model_name=model_name, use_cv=False)
        else:
            summary = self.ml_trainer.train(features, labels, model_name=model_name, use_cv=False)

        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        status_payload = {
            "model": summary.model_name,
            "trained_at": timestamp,
            "metrics": summary.metrics,
        }
        self.state_manager.save_parameters(
            {"ml_last_trained_at": timestamp, "ml_status": status_payload, "training_progress": 100}
        )
        self.strategy_manager.persist_ml_state({**status_payload, "enabled": True, "training_progress": 100})
        if self.ml_predictor:
            try:
                self.ml_predictor.load(summary.model_name)
            except Exception:
                logging.exception("ML predictor failed to load the newly trained model.")
        self.last_ml_training_summary = {
            **status_payload,
            "history_days": history_days,
            "interval": interval,
            "symbol": symbol,
        }
        self._last_ml_training_date = now.date()
        logging.info("Automatic ML training completed and persisted (%s).", summary.metrics)

    def _run_daily_backtest(self, now: datetime) -> None:
        """Run walk-forward + grid backtest and (optionally) apply best config (simulation only)."""
        try:
            if self.backtesting_engine is None:
                return
            back_cfg = self.config.get("backtesting", {}) or {}
            auto_cfg = back_cfg.get("auto", {}) or {}
            if not bool(auto_cfg.get("enabled", True)):
                return

            self._backtest_state = {"running": True, "started_at": now.isoformat() + "Z"}
            try:
                self.state_manager.save_parameters({"backtest_last": self._backtest_state})
            except Exception:
                pass

            history_days = int(auto_cfg.get("history_days", 0) or 0)
            limit_raw = auto_cfg.get("limit_candles", None)
            limit_raw_int: Optional[int] = None
            if limit_raw is not None:
                try:
                    limit_raw_int = int(limit_raw)
                except Exception:
                    limit_raw_int = None

            limit_candles = int(limit_raw_int) if limit_raw_int is not None else int(self._estimate_candles(self.interval, 30))
            limit_candles = int(limit_candles)
            limit_used = None if limit_raw_int is not None and int(limit_raw_int) <= 0 else limit_candles
            candles = self._get_training_candles(
                symbol=self.symbol,
                interval=self.interval,
                history_days=history_days if history_days > 0 else 30,
                now=now,
                limit=limit_used,
            )
            if not candles:
                logging.warning("Backtest optimization skipped: no OHLCV data.")
                self._backtest_state = {"running": False, "ended_at": datetime.utcnow().isoformat() + "Z", "error": "no_ohlcv"}
                try:
                    self.state_manager.save_parameters({"backtest_last": self._backtest_state})
                except Exception:
                    pass
                return
            
            active_strategy = self.strategy_manager.get_active_strategy_name()
            base_config = dict(self.strategy_manager.get_active_strategy().config or {})
            grid = _default_grid(active_strategy, base_config)
            cap = int(auto_cfg.get("grid_cap", 30) or 30)
            if cap > 0 and len(grid) > cap:
                # Deterministic-ish sample per day so results are repeatable.
                seed = int(auto_cfg.get("grid_seed", 0) or 0) or int(now.strftime("%Y%m%d"))
                rng = random.Random(seed)
                grid = rng.sample(grid, k=cap)

            train_size = int(auto_cfg.get("train_size", 800) or 800)
            test_size = int(auto_cfg.get("test_size", 200) or 200)
            step_size = int(auto_cfg.get("step_size", test_size) or test_size)
            # Keep walk-forward tractable on long histories.
            max_folds = int(auto_cfg.get("max_folds", 0) or 0)
            if max_folds > 0 and len(candles) > (train_size + test_size) and step_size > 0:
                span = max(0, len(candles) - train_size - test_size)
                if span > 0:
                    step_size = max(step_size, max(1, int(span / max_folds)))
            metric = str(auto_cfg.get("metric", "sharpe") or "sharpe").lower()
            if metric not in {"sharpe", "pnl"}:
                metric = "sharpe"

            required = int(train_size) + int(test_size)
            if len(candles) < required:
                logging.warning(
                    "Backtest optimization skipped: not enough candles (%s < %s).",
                    len(candles),
                    required,
                )
                self._backtest_state = {
                    "running": False,
                    "ended_at": datetime.utcnow().isoformat() + "Z",
                    "skipped": True,
                    "reason": "not_enough_candles",
                    "available_candles": int(len(candles)),
                    "required_candles": int(required),
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "strategy": active_strategy,
                    "metric": metric,
                    "grid_size": len(grid),
                    "params": {"train_size": train_size, "test_size": test_size, "step_size": step_size, "limit_candles": limit_used, "history_days": history_days},
                }
                try:
                    self.state_manager.save_parameters({"backtest_last": self._backtest_state})
                except Exception:
                    pass
                return

            try:
                wf = run_walkforward(
                    engine=self.backtesting_engine,
                    candles=list(candles),
                    strategy_name=active_strategy,
                    grid=list(grid),
                    train_size=train_size,
                    test_size=test_size,
                    step_size=step_size,
                    metric=metric,
                )
            except ValueError as e:
                msg = str(e)
                if "Not enough candles" in msg:
                    self._backtest_state = {
                        "running": False,
                        "ended_at": datetime.utcnow().isoformat() + "Z",
                        "skipped": True,
                        "reason": "not_enough_candles",
                        "available_candles": int(len(candles)),
                        "required_candles": int(required),
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "strategy": active_strategy,
                        "metric": metric,
                        "grid_size": len(grid),
                        "params": {"train_size": train_size, "test_size": test_size, "step_size": step_size, "limit_candles": limit_used, "history_days": history_days},
                    }
                    try:
                        self.state_manager.save_parameters({"backtest_last": self._backtest_state})
                    except Exception:
                        pass
                    return
                raise

            folds = wf.get("folds") or []
            summary = wf.get("summary") or {}
            total_test_pnl = float(summary.get("total_test_pnl", 0.0) or 0.0)
            avg_test_sharpe = float(summary.get("avg_test_sharpe", 0.0) or 0.0)

            # Aggregate "best_config" across folds to pick a robust candidate.
            cfg_scores: Dict[str, Dict[str, Any]] = {}
            for f in folds:
                cfg = f.get("best_config") or {}
                key = str(sorted(cfg.items()))
                bucket = cfg_scores.setdefault(key, {"cfg": cfg, "pnl": 0.0, "sharpe": 0.0, "n": 0})
                tm = f.get("test_metrics") or {}
                bucket["pnl"] += float(tm.get("pnl", 0.0) or 0.0)
                bucket["sharpe"] += float(tm.get("sharpe", 0.0) or 0.0)
                bucket["n"] += 1

            best_cfg = None
            best_score = float("-inf")
            for bucket in cfg_scores.values():
                n = int(bucket.get("n") or 0)
                if n <= 0:
                    continue
                avg_sh = float(bucket["sharpe"]) / n
                pnl_sum = float(bucket["pnl"])
                score = avg_sh if metric == "sharpe" else pnl_sum
                # Prefer higher score; break ties by pnl.
                if score > best_score or (abs(score - best_score) < 1e-9 and pnl_sum > float((best_cfg or {}).get("_pnl_tie", -1e18))):
                    best_score = score
                    best_cfg = dict(bucket["cfg"])
                    best_cfg["_pnl_tie"] = pnl_sum

            if best_cfg and "_pnl_tie" in best_cfg:
                best_cfg.pop("_pnl_tie", None)

            apply_cfg = bool(auto_cfg.get("apply_best", True))
            min_folds = int(auto_cfg.get("min_folds", 2) or 2)
            min_avg_sharpe = float(auto_cfg.get("min_avg_test_sharpe", 0.05) or 0.05)
            min_total_pnl = float(auto_cfg.get("min_total_test_pnl", 0.0) or 0.0)
            should_apply = (
                apply_cfg
                and best_cfg is not None
                and int(summary.get("folds", 0) or 0) >= min_folds
                and total_test_pnl >= min_total_pnl
                and (avg_test_sharpe >= min_avg_sharpe if metric == "sharpe" else True)
            )

            applied = False
            if should_apply:
                logging.info("Backtest best config selected; applying to strategy '%s'.", active_strategy)
                self.strategy_manager.update_strategy_config(active_strategy, best_cfg)
                applied = True

            export_root = Path(back_cfg.get("export_dir", "backtests"))
            export_dir = export_root / now.strftime("%Y-%m-%d_%H%M%S") / "auto_backtest"
            try:
                export_dir.mkdir(parents=True, exist_ok=True)
                import json

                (export_dir / "walkforward_summary.json").write_text(json.dumps(wf, indent=2), encoding="utf-8")
            except Exception:
                pass

            self._backtest_state = {
                "running": False,
                "ended_at": datetime.utcnow().isoformat() + "Z",
                "symbol": self.symbol,
                "interval": self.interval,
                "strategy": active_strategy,
                "metric": metric,
                "grid_size": len(grid),
                "params": {"train_size": train_size, "test_size": test_size, "step_size": step_size, "limit_candles": limit_used},
                "summary": {"folds": int(summary.get("folds", 0) or 0), "total_test_pnl": total_test_pnl, "avg_test_sharpe": avg_test_sharpe},
                "best_config": best_cfg or {},
                "applied": applied,
                "export_path": str(export_dir),
            }
            try:
                self.state_manager.save_parameters({"backtest_last": self._backtest_state})
            except Exception:
                pass
                
        except Exception:
            logging.exception("Daily backtest optimization failed.")
            try:
                self._backtest_state = {"running": False, "error": "failed", "ended_at": datetime.utcnow().isoformat() + "Z"}
                self.state_manager.save_parameters({"backtest_last": self._backtest_state})
            except Exception:
                pass

    def _cache_recent_ohlcv(self, *, symbol: str, interval: str, candles: List[Dict[str, Any]]) -> None:
        store = self.market_data_store
        if store is None:
            return
        md_cfg = self.config.get("historical_data", {}) or {}
        if not bool(md_cfg.get("enabled", True)):
            return
        if not bool(md_cfg.get("cache_recent", True)):
            return
        store.upsert_candles(symbol=symbol, interval=interval, candles=candles)

    def _get_training_candles(
        self,
        *,
        symbol: str,
        interval: str,
        history_days: int,
        now: datetime,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Fetch candles for training/backtesting using the local cache when available."""
        if history_days <= 0:
            history_days = 30
        store = self.market_data_store
        md_cfg = self.config.get("historical_data", {}) or {}
        if store is None or not bool(md_cfg.get("enabled", True)):
            # Fallback to API-limited fetch.
            api_limit = self._estimate_candles(interval, history_days)
            if limit is not None and limit > 0:
                api_limit = min(int(api_limit), int(limit))
            return self.exchange_client.fetch_ohlcv(symbol, interval, limit=int(api_limit))

        request_limit = int(md_cfg.get("request_limit", 1000) or 1000)
        sleep_seconds = float(md_cfg.get("request_sleep_seconds", 0.1) or 0.0)
        max_requests = int(md_cfg.get("max_requests_per_sync", 10_000) or 10_000)

        end_dt = now.replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=int(history_days))
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        try:
            stats = store.ensure_range(
                exchange_client=self.exchange_client,
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
                request_limit=request_limit,
                sleep_seconds=sleep_seconds,
                max_requests=max_requests,
            )
            try:
                self.state_manager.save_parameters({"market_data_last_sync": stats.to_dict()})
            except Exception:
                pass
        except Exception:
            logging.exception("Market data sync failed; falling back to direct OHLCV fetch.")
            api_limit = self._estimate_candles(interval, history_days)
            if limit is not None and limit > 0:
                api_limit = min(int(api_limit), int(limit))
            return self.exchange_client.fetch_ohlcv(symbol, interval, limit=int(api_limit))

        candles = store.get_candles(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms, order="ASC")
        if limit is not None and limit > 0 and len(candles) > int(limit):
            candles = candles[-int(limit) :]
        return candles

    def _get_last_ml_training_date(self) -> Optional[date]:
        if self._last_ml_training_date is not None:
            return self._last_ml_training_date
        params = self.state_manager.load_parameters()
        trained_at = params.get("ml_last_trained_at")
        if not trained_at:
            return None
        try:
            parsed = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
        except Exception:
            return None
        self._last_ml_training_date = parsed.date()
        return self._last_ml_training_date

    @staticmethod
    def _estimate_candles(interval: str, days: int) -> int:
        mapping = {
            "1m": 60 * 24,
            "5m": 12 * 24,
            "15m": 4 * 24,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }
        per_day = mapping.get(interval, 24)
        return max(100, min(1200, per_day * max(days, 1)))

    def _should_run_daily(self, now: datetime) -> bool:
        """Return True when the configured daily time has been reached."""
        if self.last_daily_run == now.date():
            return False
        scheduled_time = datetime.combine(now.date(), self.daily_run_time)
        return now >= scheduled_time

    @staticmethod
    def _parse_daily_time(value: str) -> time_cls:
        hour, minute = value.split(":")
        return time_cls(hour=int(hour), minute=int(minute))

    # ------------------------------------------------------------------ #
    # Execution routing (paper vs live)
    # ------------------------------------------------------------------ #
    def _get_execution_cfg(self) -> Dict[str, Any]:
        return self.config.get("execution", {}) or {}

    def _get_live_cfg(self) -> Dict[str, Any]:
        return (self._get_execution_cfg().get("live", {}) or {})

    def _is_learning_mode(self) -> bool:
        return self.execution_mode != "live"

    def _get_effective_risk_cfg(self) -> Dict[str, Any]:
        """Return risk config with live overrides applied when in live mode."""
        base = dict(self.config.get("risk", {}) or {})
        if self._is_learning_mode():
            # In learning mode (paper), do not block entries via circuit-breaker pauses.
            base["circuit_breaker_enabled"] = False
            return base
        live = self._get_live_cfg()
        try:
            if "max_daily_loss_usdt" in live:
                base["max_daily_loss_usdt"] = float(live.get("max_daily_loss_usdt") or 0.0)
        except Exception:
            pass
        try:
            if "max_drawdown_usdt" in live:
                base["max_drawdown_usdt"] = float(live.get("max_drawdown_usdt") or 0.0)
        except Exception:
            pass
        return base

    def _is_live_armed(self) -> bool:
        live = self._get_live_cfg()
        env_var = str(live.get("require_env_var", "BINANCE_LIVE_ARMED") or "BINANCE_LIVE_ARMED")
        env_value = str(live.get("require_env_value", "1") or "1")
        return str(os.getenv(env_var, "")).strip() == env_value

    def get_execution_status(self) -> Dict[str, Any]:
        live = self._get_live_cfg()
        exchange_cfg = self.config.get("exchange", {}) or {}
        return {
            "mode": self.execution_mode,
            "learning_mode": bool(self._is_learning_mode()),
            "live_available": self.live_trading_engine is not None,
            "live_armed": bool(self._is_live_armed()),
            "sandbox": bool(exchange_cfg.get("use_sandbox", False)),
            "allowed_symbol": str(live.get("allow_symbol", self.symbol) or self.symbol).upper(),
            "limits": {
                "max_notional_usdt_per_trade": float(live.get("max_notional_usdt_per_trade", 0.0) or 0.0),
                "max_daily_loss_usdt": float(live.get("max_daily_loss_usdt", 0.0) or 0.0),
                "max_orders_per_day": int(live.get("max_orders_per_day", 0) or 0),
            },
        }

    def set_execution_mode(self, mode: str, *, force: bool = False) -> Dict[str, Any]:
        """Persist and activate an execution mode. Defaults to safe checks."""
        requested = str(mode or "").lower().strip()
        if requested not in {"paper", "live"}:
            raise ValueError("mode must be 'paper' or 'live'")

        if requested == "live":
            if self.live_trading_engine is None:
                raise RuntimeError("Live trading engine not configured.")
            if not self._is_live_armed():
                raise RuntimeError("Live trading not armed (missing env var).")
            live_cfg = self._get_live_cfg()
            allowed = str(live_cfg.get("allow_symbol", self.symbol) or self.symbol).upper()
            if allowed != str(self.symbol or "").upper():
                raise RuntimeError(f"Live allowed_symbol mismatch (configured {allowed}, bot symbol {self.symbol}).")
            if not force:
                try:
                    readiness = self.state_manager.get_flag("ready_for_micro_live") or {}
                    score = float(((readiness.get("details") or {}).get("score")) or 0.0)
                    min_score = float(live_cfg.get("require_min_readiness_score", self.config.get("readiness", {}).get("min_score", 80)) or 0.0)
                    if min_score > 0 and score < min_score:
                        raise RuntimeError(f"Readiness score too low for live ({score:.0f} < {min_score:.0f}).")
                except RuntimeError:
                    raise
                except Exception:
                    raise RuntimeError("Readiness flag missing; use force=true to override.")

        previous = self.execution_mode
        self.execution_mode = requested
        try:
            self.state_manager.save_parameters({"execution_mode": self.execution_mode})
        except Exception:
            pass

        # Clear any persisted circuit-breaker pause when switching out of live mode.
        if previous == "live" and self.execution_mode != "live":
            self._risk_pause = {}
            try:
                self.state_manager.save_parameters({"risk_pause": self._risk_pause})
            except Exception:
                pass
        return self.get_execution_status()

    def _get_active_engine(self) -> Any:
        if self.execution_mode == "live" and self.live_trading_engine is not None:
            return self.live_trading_engine
        return self.paper_trading_engine

    @staticmethod
    def _get_portfolio_snapshot(engine: Any, *, current_price: float) -> Dict[str, Any]:
        try:
            return engine.get_portfolio_snapshot(current_price=current_price)
        except TypeError:
            return engine.get_portfolio_snapshot()

    @staticmethod
    def _get_min_qty_for_symbol(engine: Any, symbol: str) -> float:
        sym = str(symbol or "").upper()
        if not sym:
            return 0.0
        try:
            rules_map = getattr(engine, "symbol_rules", None)
            if isinstance(rules_map, dict):
                rules = rules_map.get(sym)
                if rules is not None:
                    try:
                        candidate = float(getattr(rules, "min_qty", 0.0) or 0.0)
                        if candidate > 0:
                            return candidate
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            return float(getattr(engine, "min_order_quantity", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _check_stop_loss_take_profit(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check open positions for stop-loss or take-profit triggers."""
        decisions = []
        engine = self._get_active_engine()
        open_positions = engine.get_open_positions()
        last_candle = (indicators.get("ohlcv") or [])[-1] if indicators else None
        close_price = float(indicators.get("closes", [0.0])[-1] or 0.0)
        high_price = float((last_candle or {}).get("high", close_price) or close_price)
        low_price = float((last_candle or {}).get("low", close_price) or close_price)
        strategy = self.strategy_manager.get_active_strategy()
        stop_loss_pct = strategy.config.get("stop_loss_pct", 0.05)
        take_profit_pct = strategy.config.get("take_profit_pct", 0.10)
        trailing_enabled = bool(strategy.config.get("trailing_stop_enabled", False))
        ignore_take_profit = trailing_enabled and bool(strategy.config.get("trailing_stop_ignore_take_profit", False))
        
        for pos in open_positions:
            symbol = pos["symbol"]
            if str(symbol).upper() != str(self.symbol).upper():
                continue
            min_qty = self._get_min_qty_for_symbol(engine, symbol)
            qty = float(pos.get("quantity", 0.0))
            if min_qty > 0 and abs(qty) < min_qty:
                continue
            avg_price = float(pos.get("avg_price", 0.0))
            if qty > 0:  # Long position
                stop_price = float(pos.get("stop_loss") or (avg_price * (1 - stop_loss_pct)))
                take_price = float(pos.get("take_profit") or (avg_price * (1 + take_profit_pct)))
                if low_price <= stop_price:
                    decisions.append({
                        "decision": "sell",
                        "price": stop_price,
                        "suggested_size": abs(qty),
                        "confidence": 1.0,
                        "reasons": [f"Stop-loss triggered at {stop_price:.4f} (low {low_price:.4f})"],
                        "metadata": {"trigger": "stop_loss", "avg_price": avg_price, "stop_loss": stop_price, "take_profit": take_price},
                    })
                elif (not ignore_take_profit) and high_price >= take_price:
                    decisions.append({
                        "decision": "sell",
                        "price": take_price,
                        "suggested_size": abs(qty),
                        "confidence": 1.0,
                        "reasons": [f"Take-profit triggered at {take_price:.4f} (high {high_price:.4f})"],
                        "metadata": {"trigger": "take_profit", "avg_price": avg_price, "stop_loss": stop_price, "take_profit": take_price},
                    })
            elif qty < 0:  # Short position (if supported)
                stop_price = avg_price * (1 + stop_loss_pct)
                take_price = avg_price * (1 - take_profit_pct)
                if high_price >= stop_price:
                    decisions.append({
                        "decision": "buy",
                        "price": stop_price,
                        "suggested_size": abs(qty),
                        "confidence": 1.0,
                        "reasons": [f"Stop-loss triggered at {stop_price:.4f} (high {high_price:.4f})"],
                        "metadata": {"trigger": "stop_loss", "avg_price": avg_price},
                    })
                elif low_price <= take_price:
                    decisions.append({
                        "decision": "buy",
                        "price": take_price,
                        "suggested_size": abs(qty),
                        "confidence": 1.0,
                        "reasons": [f"Take-profit triggered at {take_price:.4f} (low {low_price:.4f})"],
                        "metadata": {"trigger": "take_profit", "avg_price": avg_price},
                    })
        return decisions

    @staticmethod
    def _ema_series(values: List[float], period: int) -> List[float]:
        if not values:
            return []
        multiplier = 2 / (period + 1)
        ema_values = [float(values[0])]
        for price in values[1:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values

    # ------------------------------------------------------------------ #
    # Multi-timeframe signals + trailing stop (paper)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_iso_z(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    def _get_strategy_signal_interval(self, strategy: Any) -> str:
        cfg = getattr(strategy, "config", {}) or {}
        raw = cfg.get("signal_interval") or cfg.get("signal_timeframe") or cfg.get("signal_time_frame")
        if raw is None:
            return str(self.interval)
        try:
            interval = str(raw).strip().lower().replace(" ", "")
        except Exception:
            return str(self.interval)
        if not interval:
            return str(self.interval)
        if interval.isdigit():
            interval = f"{interval}m"
        if len(interval) < 2:
            return str(self.interval)
        if not any(interval.endswith(suf) for suf in ("m", "h", "d", "w")):
            return str(self.interval)
        try:
            int(interval[:-1])
        except Exception:
            return str(self.interval)
        return interval

    def _strip_open_candle(self, candles: List[Dict[str, Any]], *, now: datetime) -> List[Dict[str, Any]]:
        """Return candles without the currently-open bar (for stable multi-minute signals)."""
        if not isinstance(candles, list) or len(candles) < 2:
            return candles
        last = candles[-1] or {}
        close_time = self._parse_iso_z(last.get("close_time"))
        if close_time is None:
            return candles
        now_aware = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        if close_time > now_aware:
            return candles[:-1]
        return candles

    def _apply_paper_trailing_state(self, *, engine: Any) -> None:
        if engine is not self.paper_trading_engine:
            return
        sym = str(self.symbol or "").upper()
        if not sym:
            return
        if not isinstance(self._paper_trailing_state, dict):
            self._paper_trailing_state = {}
        state = self._paper_trailing_state.get(sym)
        if not isinstance(state, dict):
            return

        positions_map = getattr(engine, "positions", None)
        pos = positions_map.get(sym) if isinstance(positions_map, dict) else None
        if not isinstance(pos, dict):
            self._paper_trailing_state.pop(sym, None)
            try:
                self.state_manager.save_parameters({"paper_trailing_state": self._paper_trailing_state})
            except Exception:
                pass
            return

        try:
            qty = float(pos.get("qty", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        min_qty = self._get_min_qty_for_symbol(engine, sym)
        if qty <= 0.0 or (min_qty > 0 and abs(qty) < min_qty):
            self._paper_trailing_state.pop(sym, None)
            try:
                self.state_manager.save_parameters({"paper_trailing_state": self._paper_trailing_state})
            except Exception:
                pass
            return

        stop_loss = state.get("stop_loss")
        if stop_loss is not None:
            try:
                sl = float(stop_loss)
                if sl > 0:
                    pos["stop_loss"] = sl
            except Exception:
                pass
        high_watermark = state.get("high_watermark")
        if high_watermark is not None:
            try:
                pos["trail_high"] = float(high_watermark)
            except Exception:
                pass
        trail_distance = state.get("trail_distance")
        if trail_distance is not None:
            try:
                pos["trail_distance"] = float(trail_distance)
            except Exception:
                pass
        last_candle_time = state.get("last_candle_time")
        if last_candle_time:
            try:
                pos["trail_last_candle_time"] = str(last_candle_time)
            except Exception:
                pass

    def _maybe_update_paper_trailing_stop(self, *, now: datetime, indicators: Dict[str, Any]) -> None:
        """Update trailing stop-loss for the active symbol (paper mode only)."""
        engine = self._get_active_engine()
        if engine is not self.paper_trading_engine:
            return

        strategy = self.strategy_manager.get_active_strategy()
        cfg = getattr(strategy, "config", {}) or {}
        if not bool(cfg.get("trailing_stop_enabled", False)):
            return

        sym = str(self.symbol or "").upper()
        if not sym:
            return

        positions_map = getattr(engine, "positions", None)
        pos = positions_map.get(sym) if isinstance(positions_map, dict) else None
        if not isinstance(pos, dict):
            if isinstance(self._paper_trailing_state, dict) and sym in self._paper_trailing_state:
                self._paper_trailing_state.pop(sym, None)
                try:
                    self.state_manager.save_parameters({"paper_trailing_state": self._paper_trailing_state})
                except Exception:
                    pass
            return

        try:
            qty = float(pos.get("qty", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        min_qty = self._get_min_qty_for_symbol(engine, sym)
        if qty <= 0.0 or (min_qty > 0 and abs(qty) < min_qty):
            if isinstance(self._paper_trailing_state, dict) and sym in self._paper_trailing_state:
                self._paper_trailing_state.pop(sym, None)
                try:
                    self.state_manager.save_parameters({"paper_trailing_state": self._paper_trailing_state})
                except Exception:
                    pass
            return

        try:
            avg_price = float(pos.get("avg_price", 0.0) or 0.0)
        except Exception:
            avg_price = 0.0
        try:
            current_stop = float(pos.get("stop_loss", 0.0) or 0.0)
        except Exception:
            current_stop = 0.0

        ohlcv = indicators.get("ohlcv") or []
        candle = None
        if isinstance(ohlcv, list) and ohlcv:
            candle = ohlcv[-1]
            if len(ohlcv) >= 2:
                close_time = self._parse_iso_z((ohlcv[-1] or {}).get("close_time"))
                now_aware = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
                if close_time is not None and close_time > now_aware:
                    candle = ohlcv[-2]
        candle = candle or {}
        candle_time = str(candle.get("open_time") or candle.get("close_time") or "")
        candle_high = float(candle.get("high", 0.0) or 0.0)
        candle_close = float(candle.get("close", 0.0) or 0.0)
        if candle_close <= 0:
            candle_close = float((indicators.get("closes") or [0.0])[-1] or 0.0)

        if not isinstance(self._paper_trailing_state, dict):
            self._paper_trailing_state = {}
        state = self._paper_trailing_state.get(sym) if isinstance(self._paper_trailing_state.get(sym), dict) else {}
        last_candle_seen = str(state.get("last_candle_time") or pos.get("trail_last_candle_time") or "")
        if candle_time and candle_time == last_candle_seen:
            return

        atr_series = indicators.get("atr") or []
        try:
            atr_val = float(atr_series[-1]) if atr_series else 0.0
        except Exception:
            atr_val = 0.0
        if atr_val <= 0 and candle_close > 0:
            atr_val = candle_close * 0.008

        prev_high = None
        for candidate in (
            pos.get("trail_high"),
            state.get("high_watermark"),
            avg_price,
            candle_close,
        ):
            try:
                v = float(candidate or 0.0)
                if v > 0:
                    prev_high = v
                    break
            except Exception:
                continue
        prev_high = float(prev_high or 0.0)
        high_now = max(prev_high, float(candle_high or 0.0), float(candle_close or 0.0))

        trail_pct = 0.0
        try:
            trail_pct = float(cfg.get("trailing_stop_pct", 0.0) or 0.0)
        except Exception:
            trail_pct = 0.0

        trail_dist = 0.0
        try:
            trail_dist = float(pos.get("trail_distance", 0.0) or 0.0)
        except Exception:
            trail_dist = 0.0
        if trail_pct > 0 and high_now > 0:
            trail_dist = max(0.0, float(high_now) * max(0.0, trail_pct))
        elif trail_dist <= 0.0:
            try:
                mult = float(cfg.get("trailing_stop_atr_multiplier", cfg.get("stop_atr_multiplier", 2.0)) or 2.0)
            except Exception:
                mult = 2.0
            trail_dist = max(0.0, float(atr_val) * max(0.0, mult))

        activated = True
        activation_pct = 0.0
        try:
            activation_pct = float(cfg.get("trailing_stop_activation_pct", 0.0) or 0.0)
        except Exception:
            activation_pct = 0.0
        if activation_pct > 0 and avg_price > 0:
            activated = high_now >= (avg_price * (1.0 + activation_pct))
        else:
            try:
                activation_atr = float(cfg.get("trailing_stop_activation_atr", 0.0) or 0.0)
            except Exception:
                activation_atr = 0.0
            if activation_atr > 0 and atr_val > 0 and avg_price > 0:
                activated = high_now >= (avg_price + activation_atr * atr_val)

        candidate_stop = (high_now - trail_dist) if (trail_dist > 0 and high_now > 0) else 0.0
        new_stop = float(current_stop)
        if activated and candidate_stop > 0:
            new_stop = max(float(current_stop), float(candidate_stop))

        min_step_pct = 0.0
        try:
            min_step_pct = float(cfg.get("trailing_stop_min_step_pct", 0.0) or 0.0)
        except Exception:
            min_step_pct = 0.0
        step = 0.0
        if min_step_pct > 0 and candle_close > 0:
            step = float(candle_close) * float(min_step_pct)
        else:
            try:
                min_step_atr = float(cfg.get("trailing_stop_min_step_atr", 0.0) or 0.0)
            except Exception:
                min_step_atr = 0.0
            step = float(atr_val) * float(min_step_atr) if (min_step_atr > 0 and atr_val > 0) else 0.0
        if step > 0 and (new_stop - float(current_stop)) < step:
            new_stop = float(current_stop)

        if candle_close > 0 and new_stop >= candle_close:
            new_stop = candle_close * (1.0 - 0.001)

        changed = False
        if new_stop > 0 and (float(current_stop) <= 0 or new_stop > float(current_stop) + 1e-9):
            pos["stop_loss"] = float(new_stop)
            changed = True
        pos["trail_high"] = float(high_now)
        pos["trail_distance"] = float(trail_dist)
        if candle_time:
            pos["trail_last_candle_time"] = candle_time

        state_changed = changed or (high_now > float(state.get("high_watermark") or 0.0) + 1e-9) or (trail_dist != float(state.get("trail_distance") or 0.0))
        if not candle_time:
            candle_time = now.isoformat() + "Z"
        if state_changed:
            self._paper_trailing_state[sym] = {
                "stop_loss": float(pos.get("stop_loss") or 0.0),
                "high_watermark": float(high_now),
                "trail_distance": float(trail_dist),
                "last_candle_time": candle_time,
                "updated_at": now.isoformat() + "Z",
                "strategy": self.strategy_manager.get_active_strategy_name(),
            }
            try:
                self.state_manager.save_parameters({"paper_trailing_state": self._paper_trailing_state})
            except Exception:
                pass
