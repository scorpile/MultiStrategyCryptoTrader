"""Frontend server implementation for the monitoring dashboard."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rest_api import attach_api_routes


def create_app(
    *,
    state_manager: Optional[Any] = None,
    strategy_manager: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    exchange_client: Optional[Any] = None,
    ml_trainer: Optional[Any] = None,
    ml_predictor: Optional[Any] = None,
    backtesting_engine: Optional[Any] = None,
    market_data_store: Optional[Any] = None,
    templates_dir: Optional[Path] = None,
    static_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> FastAPI:
    """Return a FastAPI app with dashboard + API routes."""
    templates_path = templates_dir or Path(__file__).with_suffix("").parent / "templates"
    static_path = static_dir or Path(__file__).with_suffix("").parent / "static"

    app = FastAPI(
        title="Autonomous Crypto Trading Bot (Simulation)",
        description="Dashboard server for monitoring the paper-trading bot.",
        version="0.1.0",
    )

    templates = Jinja2Templates(directory=str(templates_path))
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    attach_api_routes(
        app,
        state_manager=state_manager,
        strategy_manager=strategy_manager,
        scheduler=scheduler,
        exchange_client=exchange_client,
        ml_trainer=ml_trainer,
        ml_predictor=ml_predictor,
        backtesting_engine=backtesting_engine,
        market_data_store=market_data_store,
        config=config or {},
    )

    @app.get("/dashboard")
    async def dashboard(request: Request) -> HTMLResponse:
        """Render dashboard template using latest metrics and probability flag."""
        metrics = state_manager.get_recent_metrics(limit=7) if state_manager else []
        probability = state_manager.get_probability_flag() if state_manager else {}
        status_payload = _sanitize_for_json(_build_status_payload(metrics, probability, scheduler, strategy_manager, state_manager))
        context = {
            "request": request,
            "status": status_payload,
            "strategies": status_payload.get("strategy", {}).get("available", []),
        }
        return templates.TemplateResponse("index.html", context)

    @app.get("/")
    async def root(request: Request) -> HTMLResponse:
        """Root route: render the same dashboard for convenience."""
        metrics = state_manager.get_recent_metrics(limit=7) if state_manager else []
        probability = state_manager.get_probability_flag() if state_manager else {}
        status_payload = _sanitize_for_json(_build_status_payload(metrics, probability, scheduler, strategy_manager, state_manager))
        context = {
            "request": request,
            "status": status_payload,
            "strategies": status_payload.get("strategy", {}).get("available", []),
        }
        return templates.TemplateResponse("index.html", context)

    @app.get("/api/status", response_class=JSONResponse)
    async def api_status() -> JSONResponse:
        """Return JSON with bot status, latest metrics, and probability flag."""
        metrics = state_manager.get_recent_metrics(limit=7) if state_manager else []
        probability = state_manager.get_probability_flag() if state_manager else {}
        payload = _build_status_payload(metrics, probability, scheduler, strategy_manager, state_manager)
        safe_payload = _sanitize_for_json(jsonable_encoder(payload))
        return JSONResponse(safe_payload)

    return app


def _build_status_payload(
    metrics: List[Dict[str, Any]],
    probability: Optional[Dict[str, Any]],
    scheduler: Optional[Any],
    strategy_manager: Optional[Any],
    state_manager: Optional[Any],
) -> Dict[str, Any]:
    """Assemble a consistent JSON payload for both dashboard and API."""
    runtime = scheduler.get_runtime_snapshot() if scheduler else {}
    strategy_snapshot = runtime.get("strategy") if runtime else {}
    if not strategy_snapshot and strategy_manager:
        strategy_snapshot = strategy_manager.get_status_snapshot()

    today_iso = datetime.utcnow().date().isoformat()
    merged_metrics = list(metrics or [])
    live_daily = runtime.get("last_daily_summary") if isinstance(runtime, dict) else None
    if isinstance(live_daily, dict) and live_daily:
        live_row = {"metric_date": today_iso, **live_daily}
        merged_metrics = [m for m in merged_metrics if str(m.get("metric_date", "")) != today_iso]
        merged_metrics.insert(0, live_row)
    # `metrics` arrive newest->oldest; reverse for plotting left->right as older->newer.
    pnl_series = [float(item.get("pnl", 0.0) or 0.0) for item in reversed(merged_metrics)]
    ml_state = strategy_manager.get_ml_state() if strategy_manager else {}
    # Add live indicators and price
    current_price = runtime.get("current_price", 0)
    live_indicators = runtime.get("live_indicators", {})
    # Training progress (simplified)
    training_progress = runtime.get("training_progress", 0)
    validation_mode = runtime.get("validation_mode", False)
    payload = {
        "status": "online",
        "symbol": runtime.get("symbol"),
        "interval": runtime.get("interval"),
        "recent_metrics": merged_metrics,
        "probability_flag": probability or {},
        "strategy": strategy_snapshot,
        "execution": runtime.get("execution", {}),
        "ml_signal": runtime.get("ml_signal", {}),
        "ml_state": ml_state,
        "last_decisions": runtime.get("last_decisions", []),
        "rl_result": runtime.get("rl_result"),
        "volatility_regime": runtime.get("volatility_regime"),
        "gate": runtime.get("gate", {}),
        "fear_greed": runtime.get("fear_greed", {}),
        "sentiment_policy": runtime.get("sentiment_policy", {}),
        "auto_tune": runtime.get("auto_tune", {}),
        "backtest": runtime.get("backtest", {}),
        "risk_pause": runtime.get("risk_pause", {}),
        "manual_pause": runtime.get("manual_pause", {}),
        "is_training": runtime.get("is_training", False),
        "training_status": runtime.get("training_status", ""),
        "pnl_series": pnl_series,
        "portfolio": runtime.get("portfolio", {}),
        "current_price": current_price,
        "validation_mode": validation_mode,
        "training_progress": training_progress,
        "live_indicators": {
            "rsi": live_indicators.get("rsi", 0),
            "ema_fast": live_indicators.get("ema_fast", 0),
            "ema_slow": live_indicators.get("ema_slow", 0),
            "atr": live_indicators.get("atr", 0),
            "volume_sma": live_indicators.get("volume_sma", 0),
            "macd_histogram": live_indicators.get("macd_histogram", 0),
            "bollinger_width": live_indicators.get("bollinger_width", 0),
            "vwap": (live_indicators.get("vwap")[-1] if isinstance(live_indicators.get("vwap"), list) and live_indicators.get("vwap") else live_indicators.get("vwap", 0)),
        },
        "last_update": runtime.get("last_update", ""),
    }
    try:
        if state_manager:
            # Use UTC date from server time for daily grouping.
            from datetime import datetime as _dt, timedelta as _td

            day = _dt.utcnow().date().isoformat()
            payload["decision_stats_today"] = state_manager.get_decision_stats_for_day(day, symbol=runtime.get("symbol"))
            payload["decision_stats_last_hour"] = state_manager.get_decision_stats_since(
                _dt.utcnow() - _td(hours=1),
                symbol=runtime.get("symbol"),
            )
            payload["equity_series"] = state_manager.get_equity_since(
                _dt.utcnow() - _td(minutes=15),
                symbol=runtime.get("symbol"),
                limit=800,
            )
            try:
                payload["recent_round_trips"] = state_manager.get_recent_round_trips(
                    limit=25,
                    symbol=runtime.get("symbol"),
                )
            except Exception:
                payload["recent_round_trips"] = []
            try:
                payload["readiness_flag"] = state_manager.get_flag("ready_for_micro_live") or {}
            except Exception:
                payload["readiness_flag"] = {}
    except Exception:
        payload["decision_stats_today"] = {}
        payload["decision_stats_last_hour"] = {}
        payload["equity_series"] = []
        payload["recent_round_trips"] = []
        payload["readiness_flag"] = {}
    return payload


def _sanitize_for_json(value: Any) -> Any:
    """Best-effort JSON sanitizer for dashboard payloads.

    Starlette's `JSONResponse` uses `allow_nan=False`. If any NaN/inf sneaks into the
    status payload (often via metrics/model outputs), the whole `/api/status` call
    fails and the UI appears frozen. This function converts problematic values to
    JSON-safe primitives.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (datetime, date)):
        try:
            iso = value.isoformat()
            if isinstance(value, datetime) and value.tzinfo is None:
                iso += "Z"
            return iso
        except Exception:
            return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]

    # Common numeric-like values (Decimal / numpy scalars)
    try:
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    except Exception:
        pass

    try:
        return str(value)
    except Exception:
        return None
