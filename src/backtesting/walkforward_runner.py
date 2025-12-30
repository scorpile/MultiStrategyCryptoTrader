"""Walk-forward + grid-search backtesting runner (simulation only)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from src.backtesting.engine import BacktestingEngine
from src.exchange.binance_client import BinanceClient
from src.exchange.symbol_rules import parse_symbol_rules
from src.strategy.risk_manager import RiskManager


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _default_grid(strategy_name: str, base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_cfg = dict(base_cfg or {})
    if strategy_name == "scalping_aggressive":
        rsi_buy_values = [30, 35, 40]
        rsi_sell_values = [60, 65, 70]
        risk_values = [0.003, 0.005, 0.01]
        stop_mult_values = [1.0, 1.5, 2.0]
        profit_mult_values = [1.2, 1.5, 2.0]
        grid: List[Dict[str, Any]] = []
        for rsi_buy in rsi_buy_values:
            for rsi_sell in rsi_sell_values:
                if rsi_sell <= rsi_buy:
                    continue
                for risk_pct in risk_values:
                    for stop_mult in stop_mult_values:
                        for profit_mult in profit_mult_values:
                            cfg = dict(base_cfg)
                            cfg.update(
                                {
                                    "rsi_buy": rsi_buy,
                                    "rsi_sell": rsi_sell,
                                    "risk_pct_per_trade": risk_pct,
                                    "stop_atr_multiplier": stop_mult,
                                    "profit_target_multiplier": profit_mult,
                                }
                            )
                            grid.append(cfg)
        return grid

    if strategy_name == "ema_rsi_volume":
        rsi_buy_values = [25.0, 30.0, 35.0]
        rsi_sell_values = [65.0, 70.0, 75.0]
        vol_mult_values = [1.2, 1.5, 2.0]
        risk_values = [0.003, 0.005, 0.01]
        stop_mult_values = [1.5, 2.0, 2.5]
        profit_mult_values = [2.0, 3.0, 4.0]
        grid: List[Dict[str, Any]] = []
        for rsi_buy in rsi_buy_values:
            for rsi_sell in rsi_sell_values:
                if rsi_sell <= rsi_buy:
                    continue
                for vol_mult in vol_mult_values:
                    for risk_pct in risk_values:
                        for stop_mult in stop_mult_values:
                            for profit_mult in profit_mult_values:
                                cfg = dict(base_cfg)
                                cfg.update(
                                    {
                                        "rsi_buy_threshold": float(rsi_buy),
                                        "rsi_sell_threshold": float(rsi_sell),
                                        "volume_spike_multiplier": float(vol_mult),
                                        "risk_pct_per_trade": float(risk_pct),
                                        "stop_atr_multiplier": float(stop_mult),
                                        "profit_target_multiplier": float(profit_mult),
                                    }
                                )
                                grid.append(cfg)
        return grid

    if strategy_name == "bollinger_range":
        rsi_oversold_values = [25.0, 30.0, 35.0]
        rsi_overbought_values = [65.0, 70.0, 75.0]
        tol_values = [0.0, 0.001, 0.002]
        risk_values = [0.003, 0.005, 0.01]
        stop_mult_values = [1.0, 1.5, 2.0]
        profit_mult_values = [1.0, 1.5, 2.0]
        tp_targets = ["middle", "upper"]
        grid: List[Dict[str, Any]] = []
        for rsi_os in rsi_oversold_values:
            for rsi_ob in rsi_overbought_values:
                if rsi_ob <= rsi_os:
                    continue
                for tol in tol_values:
                    for risk_pct in risk_values:
                        for stop_mult in stop_mult_values:
                            for profit_mult in profit_mult_values:
                                for tp_target in tp_targets:
                                    cfg = dict(base_cfg)
                                    cfg.update(
                                        {
                                            "rsi_oversold": float(rsi_os),
                                            "rsi_overbought": float(rsi_ob),
                                            "touch_tolerance_pct": float(tol),
                                            "risk_pct_per_trade": float(risk_pct),
                                            "stop_atr_multiplier": float(stop_mult),
                                            "profit_target_multiplier": float(profit_mult),
                                            "take_profit_target": str(tp_target),
                                        }
                                    )
                                    grid.append(cfg)
        return grid

    if strategy_name in {"moderate", "aggressive", "conservative"}:
        rsi_buy_values = [20, 25, 30, 35, 40]
        rsi_sell_values = [60, 65, 70, 75, 80]
        grid = []
        for rsi_buy in rsi_buy_values:
            for rsi_sell in rsi_sell_values:
                if rsi_sell <= rsi_buy:
                    continue
                cfg = dict(base_cfg)
                cfg.update({"rsi_buy": rsi_buy, "rsi_sell": rsi_sell})
                grid.append(cfg)
        return grid

    return [base_cfg]


def _score(metrics: Any, metric_name: str) -> float:
    value = getattr(metrics, metric_name, None)
    try:
        return float(value)
    except Exception:
        return 0.0


def run_walkforward(
    *,
    engine: BacktestingEngine,
    candles: List[Dict[str, Any]],
    strategy_name: str,
    grid: List[Dict[str, Any]],
    train_size: int,
    test_size: int,
    step_size: int,
    metric: str,
) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []
    if len(candles) < train_size + test_size:
        raise ValueError("Not enough candles for the requested walk-forward split.")

    for test_start in range(train_size, len(candles) - test_size + 1, step_size):
        train = candles[test_start - train_size : test_start]
        test = candles[test_start : test_start + test_size]

        best_cfg: Dict[str, Any] | None = None
        best_score = float("-inf")
        best_train_metrics = None

        for cfg in grid:
            result = engine.run(
                strategy_name,
                train,
                strategy_config=cfg,
                collect_trades=False,
                collect_equity_curve=False,
            )
            score = _score(result.metrics, metric)
            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_train_metrics = result.metrics.__dict__

        if best_cfg is None:
            continue

        test_result = engine.run(
            strategy_name,
            test,
            strategy_config=best_cfg,
            collect_trades=False,
            collect_equity_curve=False,
        )
        folds.append(
            {
                "test_start": test[0]["open_time"],
                "test_end": test[-1]["open_time"],
                "best_config": best_cfg,
                "train_metrics": best_train_metrics,
                "test_metrics": test_result.metrics.__dict__,
            }
        )

    total_pnl = sum(float(f["test_metrics"].get("pnl", 0.0)) for f in folds)
    avg_sharpe = sum(float(f["test_metrics"].get("sharpe", 0.0)) for f in folds) / max(len(folds), 1)
    return {
        "strategy": strategy_name,
        "metric": metric,
        "train_size": train_size,
        "test_size": test_size,
        "step_size": step_size,
        "folds": folds,
        "summary": {
            "folds": len(folds),
            "total_test_pnl": total_pnl,
            "avg_test_sharpe": avg_sharpe,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward + grid-search backtester (simulation only).")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--metric", choices=["pnl", "sharpe"], default="sharpe")
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample candles (no HTTP).")
    parser.add_argument("--export-dir", default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    strategy_name = args.strategy or cfg.get("strategy", {}).get("default", "moderate")
    symbol = args.symbol or cfg.get("general", {}).get("base_symbol", "SOLUSDT")
    interval = args.interval or cfg.get("general", {}).get("time_frame", "1h")

    exchange_cfg = cfg.get("exchange", {})
    exchange = BinanceClient(
        api_key=exchange_cfg.get("api_key"),
        api_secret=exchange_cfg.get("api_secret"),
        sample_mode=bool(args.sample),
    )
    try:
        symbol_info = exchange.fetch_symbol_info(symbol)
        symbol_rules = parse_symbol_rules(symbol_info)
    except Exception:
        symbol_rules = None
    candles = exchange.fetch_ohlcv(symbol, interval, limit=int(args.limit))

    trading_cfg = cfg.get("trading", {})
    risk_cfg = cfg.get("risk", {})
    risk_manager = RiskManager(
        max_position_size_usdt=risk_cfg.get("max_position_size_usdt", trading_cfg.get("max_position_size_usdt", 100.0)),
        max_daily_loss_usdt=risk_cfg.get("max_daily_loss_usdt", 50.0),
        default_risk_pct_per_trade=float(risk_cfg.get("risk_pct_per_trade", risk_cfg.get("risk_per_trade_pct", 0.01))),
    )
    backtesting_cfg = cfg.get("backtesting", {})
    engine = BacktestingEngine(
        initial_cash=float(backtesting_cfg.get("initial_cash", 10_000.0)),
        fee_rate=float(backtesting_cfg.get("fee_rate", trading_cfg.get("fee_rate", 0.0005))),
        slippage=float(trading_cfg.get("slippage", 0.0005)),
        strategy_configs=cfg.get("strategy", {}).get("strategies"),
        risk_manager=risk_manager,
        symbol_rules=symbol_rules,
    )

    base_strategy_cfg = (cfg.get("strategy", {}).get("strategies", {}) or {}).get(strategy_name, {})
    grid = _default_grid(strategy_name, base_strategy_cfg)

    step = int(args.step) if args.step is not None else int(args.test)
    export_root = args.export_dir or backtesting_cfg.get("export_dir", "backtests")
    export_dir = Path(export_root) / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") / "walkforward"
    export_dir.mkdir(parents=True, exist_ok=True)

    result = run_walkforward(
        engine=engine,
        candles=candles,
        strategy_name=strategy_name,
        grid=grid,
        train_size=int(args.train),
        test_size=int(args.test),
        step_size=step,
        metric=args.metric,
    )
    (export_dir / "walkforward_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {export_dir / 'walkforward_summary.json'}")


if __name__ == "__main__":
    main()
