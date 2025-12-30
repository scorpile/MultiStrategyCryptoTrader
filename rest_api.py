"""REST API endpoints for strategy control, ML training, and backtesting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException

from src.ml import feature_engineering


def attach_api_routes(
    app: FastAPI,
    *,
    state_manager: Optional[Any],
    strategy_manager: Optional[Any],
    scheduler: Optional[Any],
    exchange_client: Optional[Any],
    ml_trainer: Optional[Any],
    ml_predictor: Optional[Any],
    backtesting_engine: Optional[Any],
    market_data_store: Optional[Any],
    config: Dict[str, Any],
) -> None:
    router = APIRouter(prefix="/api")

    def _fetch_history(symbol: str, interval: str, *, days: int) -> list[dict[str, Any]]:
        _require(exchange_client, "Exchange client not configured.")
        days_i = max(int(days or 1), 1)
        if market_data_store is None or not bool((config.get("historical_data", {}) or {}).get("enabled", True)):
            candles_limit = _estimate_candles(interval, days_i)
            return exchange_client.fetch_ohlcv(symbol, interval, limit=candles_limit)

        md_cfg = config.get("historical_data", {}) or {}
        request_limit = int(md_cfg.get("request_limit", 1000) or 1000)
        sleep_seconds = float(md_cfg.get("request_sleep_seconds", 0.1) or 0.0)
        max_requests = int(md_cfg.get("max_requests_per_sync", 10_000) or 10_000)

        end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=days_i)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        market_data_store.ensure_range(
            exchange_client=exchange_client,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            request_limit=request_limit,
            sleep_seconds=sleep_seconds,
            max_requests=max_requests,
        )
        return market_data_store.get_candles(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms, order="ASC")

    @router.get("/strategy/list")
    async def list_strategies() -> Dict[str, Any]:
        _require(strategy_manager, "Strategy manager not configured.")
        return {"strategies": strategy_manager.get_available_strategies()}

    @router.get("/strategy/current")
    async def current_strategy() -> Dict[str, Any]:
        _require(strategy_manager, "Strategy manager not configured.")
        return strategy_manager.get_status_snapshot()

    @router.post("/strategy/set")
    async def set_strategy(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(strategy_manager, "Strategy manager not configured.")
        name = payload.get("strategy")
        overrides = payload.get("config")
        if not name:
            raise HTTPException(status_code=400, detail="Missing 'strategy' in payload.")
        snapshot = strategy_manager.set_active_strategy(name, overrides=overrides)
        return {"strategy": snapshot}

    @router.get("/symbol/list")
    async def list_symbols() -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        try:
            allowed = scheduler._get_allowed_symbols()
        except Exception:
            allowed = ["SOLUSDT", "ETHUSDT", "BTCUSDT"]
        current = getattr(scheduler, "symbol", None) or config.get("general", {}).get("base_symbol", "SOLUSDT")
        return {"symbols": allowed, "current": str(current).upper()}

    @router.post("/symbol/set")
    async def set_symbol(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        symbol = payload.get("symbol")
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing 'symbol' in payload.")
        try:
            return scheduler.set_symbol(symbol)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Symbol change failed: {str(e)}")

    @router.get("/ml/status")
    async def ml_status() -> Dict[str, Any]:
        _require(state_manager, "State manager not configured.")
        params = state_manager.load_parameters()
        status = params.get("ml_status", {})
        if strategy_manager:
            status["runtime"] = strategy_manager.get_ml_state()
        return status

    @router.post("/ml/train")
    async def ml_train(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _require(ml_trainer, "ML trainer not configured.")
        _require(state_manager, "State manager not configured.")
        _require(exchange_client, "Exchange client not configured.")
        payload = payload or {}
        history_days = int(payload.get("history_days") or config.get("ml", {}).get("history_days", 30))
        interval = payload.get("interval") or config.get("general", {}).get("time_frame", "1h")
        default_symbol = getattr(scheduler, "symbol", None) or config.get("general", {}).get("base_symbol", "SOLUSDT")
        symbol = payload.get("symbol") or default_symbol

        candles = _fetch_history(symbol, interval, days=history_days)
        if not candles:
            raise HTTPException(status_code=500, detail="No OHLCV data available for training.")
        features, labels = feature_engineering.build_dataset(candles)
        if features.empty:
            raise HTTPException(status_code=500, detail="Feature matrix is empty; cannot train.")

        summary = ml_trainer.train(features, labels, model_name=payload.get("model_name"))
        status_payload = {
            "model": summary.model_name,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "metrics": summary.metrics,
        }
        state_manager.save_parameters({"ml_status": status_payload})
        if strategy_manager:
            strategy_manager.persist_ml_state({"model": summary.model_name, "metrics": summary.metrics, "enabled": True})
        if ml_predictor:
            ml_predictor.load(summary.model_name)
        return status_payload

    @router.get("/backtest/run")
    async def run_backtest(strategy: Optional[str] = None, interval: Optional[str] = None, days: int = 60) -> Dict[str, Any]:
        _require(backtesting_engine, "Backtesting engine not configured.")
        _require(exchange_client, "Exchange client not configured.")
        target_strategy = strategy or (strategy_manager.get_active_strategy_name() if strategy_manager else "moderate")
        target_interval = interval or config.get("general", {}).get("time_frame", "1h")
        symbol = getattr(scheduler, "symbol", None) or config.get("general", {}).get("base_symbol", "SOLUSDT")
        candles = _fetch_history(symbol, target_interval, days=int(days or 60))
        if not candles:
            raise HTTPException(status_code=500, detail="Backtest requires historical candles.")
        result = backtesting_engine.run(target_strategy, candles)
        return result.to_dict()

    @router.post("/training/start")
    async def start_training() -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        if scheduler.is_training:
            raise HTTPException(status_code=409, detail="Training already in progress.")
        try:
            import asyncio
            await asyncio.to_thread(scheduler.run_daily_tasks, datetime.utcnow())
            return {"status": "Training started successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    @router.post("/sentiment/set")
    async def set_sentiment(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        enabled = bool(payload.get("enabled", False))
        scheduler.config.setdefault("sentiment", {})["enabled"] = enabled
        # Best-effort persist in state so UI can survive restarts if desired.
        if state_manager:
            try:
                state_manager.save_parameters({"sentiment_enabled": enabled})
            except Exception:
                pass
        # Update immediately.
        scheduler._update_fear_greed(datetime.utcnow())
        return {"enabled": enabled, "fear_greed": getattr(scheduler, "_fear_greed", {})}

    @router.post("/sentiment/refresh")
    async def refresh_sentiment() -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        try:
            # Force refresh regardless of cache.
            provider = getattr(scheduler, "_fear_greed_provider", None)
            if provider is not None:
                provider.get_latest(force_refresh=True)
            scheduler._update_fear_greed(datetime.utcnow())
            return {"fear_greed": getattr(scheduler, "_fear_greed", {})}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sentiment refresh failed: {str(e)}")

    @router.post("/risk/clear")
    async def clear_risk_pause() -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        try:
            scheduler._risk_pause = {}
            if state_manager:
                try:
                    state_manager.save_parameters({"risk_pause": {}})
                except Exception:
                    pass
            return {"risk_pause": {}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Risk clear failed: {str(e)}")

    @router.post("/trading/pause")
    async def set_manual_pause(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        paused = bool(payload.get("paused", True))
        reason = str(payload.get("reason") or "").strip()
        minutes = payload.get("minutes")
        until_utc = None
        if minutes is not None:
            try:
                minutes_f = float(minutes)
                if minutes_f > 0:
                    until_utc = (datetime.utcnow() + timedelta(minutes=minutes_f)).isoformat() + "Z"
            except Exception:
                until_utc = None

        try:
            now_iso = datetime.utcnow().isoformat() + "Z"
            if paused:
                scheduler._manual_pause = {
                    "paused": True,
                    "reason": reason or "manual",
                    "set_at_utc": now_iso,
                    "until_utc": until_utc,
                }
            else:
                scheduler._manual_pause = {}

            if state_manager:
                try:
                    state_manager.save_parameters({"manual_pause": scheduler._manual_pause})
                except Exception:
                    pass
            return {"manual_pause": scheduler._manual_pause}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Manual pause failed: {str(e)}")

    @router.post("/positions/close")
    async def close_position(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        confirm = str(payload.get("confirm") or "")
        symbol = payload.get("symbol")
        try:
            mode = str((scheduler.get_execution_status() or {}).get("mode") or "paper").lower().strip()
        except Exception:
            mode = "paper"
        expected = "LIVE SELL" if mode == "live" else "SELL"
        if confirm != expected:
            raise HTTPException(status_code=400, detail=f"Confirmation required: set JSON body {{'confirm': '{expected}'}}")
        try:
            return scheduler.close_position(symbol=symbol)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Close position failed: {str(e)}")

    @router.post("/admin/reset")
    async def admin_reset(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        confirm = str(payload.get("confirm") or "")
        if confirm != "RESET":
            raise HTTPException(status_code=400, detail="Confirmation required: set JSON body {'confirm': 'RESET'}")
        try:
            import asyncio

            result = await asyncio.to_thread(scheduler.admin_reset)
            return result if isinstance(result, dict) else {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

    @router.get("/execution/status")
    async def execution_status() -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        try:
            return scheduler.get_execution_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution status failed: {str(e)}")

    @router.post("/execution/mode")
    async def set_execution_mode(payload: Dict[str, Any]) -> Dict[str, Any]:
        _require(scheduler, "Scheduler not configured.")
        mode = str(payload.get("mode") or "").lower().strip()
        force = bool(payload.get("force", False))
        confirm = str(payload.get("confirm") or "")
        if mode not in {"paper", "live"}:
            raise HTTPException(status_code=400, detail="Invalid mode; expected 'paper' or 'live'.")
        if mode == "live" and confirm != "LIVE":
            raise HTTPException(status_code=400, detail="Confirmation required: set JSON body {'confirm': 'LIVE'}")
        try:
            return scheduler.set_execution_mode(mode, force=force)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Set execution mode failed: {str(e)}")

    app.include_router(router)


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


def _require(dependency: Any, message: str) -> None:
    if dependency is None:
        raise HTTPException(status_code=503, detail=message)
