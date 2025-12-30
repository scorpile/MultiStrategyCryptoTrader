"""Entry point for the Autonomous Crypto Trading Bot."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict

import yaml

from src.core.scheduler import Scheduler
from src.core.state_manager import StateManager
from src.evaluation.evaluator import Evaluator
from src.exchange.binance_client import BinanceClient
from src.exchange.binance_spot_executor import BinanceSpotConfig, BinanceSpotExecutor
from src.exchange.paper_trading import PaperTradingEngine
from src.exchange.live_trading import LiveLimits, LiveTradingEngine
from src.exchange.symbol_rules import parse_symbol_rules
from src.frontend.server import create_app
from src.ml.model_manager import ModelManager
from src.ml.model_predictor import TrendModelPredictor
from src.ml.model_trainer import LightGBMTrainer
from src.reports.reporter import DailyReporter
from src.rl.policy_optimizer import PolicyOptimizer
from src.strategy.risk_manager import RiskManager
from src.strategy.strategy_manager import StrategyManager
from src.ai import AIAutoOptimizer
from src.backtesting.engine import BacktestingEngine
from src.data.market_data_store import MarketDataStore


def main() -> None:
    """Main entry point that wires together config, storage, and scheduler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = _load_config()

    state_manager = StateManager(Path(config.get("general", {}).get("db_path", "data/state.db")))
    state_manager.init_db()
    try:
        state_manager.migrate_simulated_pnl_net_v1()
    except Exception:
        logging.exception("Failed to migrate simulated PnL net; continuing.")

    exchange_cfg = config.get("exchange", {})
    exchange_client = BinanceClient(
        api_key=exchange_cfg.get("api_key"),
        api_secret=exchange_cfg.get("api_secret"),
        sample_mode=exchange_cfg.get("use_sample_data", False),
    )

    market_data_store = None
    try:
        md_cfg = config.get("historical_data", {}) or {}
        if bool(md_cfg.get("enabled", True)):
            market_data_store = MarketDataStore(Path(md_cfg.get("db_path", "data/market_data.sqlite")))
            market_data_store.init_db()
    except Exception:
        logging.exception("Market data store init failed; continuing without long-history cache.")

    trading_cfg = config.get("trading", {})
    base_symbol = str(config.get("general", {}).get("base_symbol", "SOLUSDT") or "SOLUSDT")
    try:
        symbol_info = exchange_client.fetch_symbol_info(base_symbol)
        symbol_rules = parse_symbol_rules(symbol_info)
    except Exception:
        symbol_rules = None

    live_trading_engine = None
    try:
        api_key = str(exchange_cfg.get("api_key") or "").strip()
        api_secret = str(exchange_cfg.get("api_secret") or "").strip()
        if api_key and api_secret:
            use_sandbox = bool(exchange_cfg.get("use_sandbox", False))
            base_url = "https://testnet.binance.vision" if use_sandbox else "https://api.binance.com"
            spot_executor = BinanceSpotExecutor(
                BinanceSpotConfig(
                    api_key=api_key,
                    api_secret=api_secret,
                    base_url=base_url,
                    timeout_seconds=float(exchange_cfg.get("timeout_seconds", 10.0) or 10.0),
                )
            )
            exec_cfg = config.get("execution", {}) or {}
            live_cfg = exec_cfg.get("live", {}) or {}
            live_limits = LiveLimits(
                allowed_symbol=str(live_cfg.get("allow_symbol", config.get("general", {}).get("base_symbol", "SOLUSDT")) or "SOLUSDT"),
                max_notional_usdt_per_trade=float(live_cfg.get("max_notional_usdt_per_trade", 10.0) or 10.0),
                max_daily_loss_usdt=float(live_cfg.get("max_daily_loss_usdt", 10.0) or 10.0),
                max_orders_per_day=int(live_cfg.get("max_orders_per_day", 20) or 20),
                min_order_quantity=float(trading_cfg.get("min_order_quantity", 0.001) or 0.001),
                require_env_var=str(live_cfg.get("require_env_var", "BINANCE_LIVE_ARMED") or "BINANCE_LIVE_ARMED"),
                require_env_value=str(live_cfg.get("require_env_value", "1") or "1"),
            )
            live_trading_engine = LiveTradingEngine(
                state_manager=state_manager,
                executor=spot_executor,
                limits=live_limits,
            )
            try:
                live_trading_engine.set_symbol_rules(base_symbol, symbol_rules)
            except Exception:
                pass
    except Exception:
        logging.exception("Live trading engine not available; continuing in paper mode only.")

    paper_trading_engine = PaperTradingEngine(
        state_manager=state_manager,
        initial_cash=trading_cfg.get("initial_cash", 10_000.0),
        fee_rate=trading_cfg.get("fee_rate", 0.0005),
        slippage=trading_cfg.get("slippage", 0.0005),
        spread_bps=trading_cfg.get("spread_bps", 5),
        dynamic_slippage_enabled=trading_cfg.get("dynamic_slippage_enabled", True),
        slippage_atr_multiplier=trading_cfg.get("slippage_atr_multiplier", 0.5),
        partial_fills_enabled=trading_cfg.get("partial_fills_enabled", True),
        partial_fill_probability=trading_cfg.get("partial_fill_probability", 0.3),
        partial_fill_min_ratio=trading_cfg.get("partial_fill_min_ratio", 0.3),
        partial_fill_max_ratio=trading_cfg.get("partial_fill_max_ratio", 1.0),
        min_order_quantity=trading_cfg.get("min_order_quantity", 0.001),
    )
    try:
        paper_trading_engine.set_symbol_rules(base_symbol, symbol_rules)
    except Exception:
        pass
    try:
        paper_trading_engine.rehydrate_from_ledger()
    except Exception:
        logging.exception("Failed to rehydrate paper portfolio from ledger; starting from initial cash.")

    risk_cfg = config.get("risk", {})
    risk_manager = RiskManager(
        max_position_size_usdt=risk_cfg.get("max_position_size_usdt", trading_cfg.get("max_position_size_usdt", 100.0)),
        max_daily_loss_usdt=risk_cfg.get("max_daily_loss_usdt", 50.0),
    )

    strategy_cfg = config.get("strategy", {})
    strategy_manager = StrategyManager(
        risk_manager=risk_manager,
        config=strategy_cfg,
        state_config_path=strategy_cfg.get("runtime_config_path", "state/config.json"),
    )

    reporter = DailyReporter(
        templates_dir=Path("src/reports/templates"),
        output_root=Path("reports"),
    )
    evaluator = Evaluator(state_manager)

    model_manager = ModelManager(Path("src/data/models"))
    ml_trainer = LightGBMTrainer(model_manager)
    ml_predictor: TrendModelPredictor | None = TrendModelPredictor(model_manager)
    try:
        ml_predictor.load()
    except Exception:
        logging.info("No trained ML model available yet; strategies will run without ML context.")

    rl_cfg = config.get("rl", {})
    rl_optimizer: PolicyOptimizer | None = None
    if rl_cfg.get("enabled", True):
        rl_optimizer = PolicyOptimizer(
            policy_path=Path(rl_cfg.get("policy_path", "state/rl_policy.json")),
            learning_rate=rl_cfg.get("learning_rate", 0.05),
            strategy_manager=strategy_manager,
        )

    openai_cfg = config.get("openai", {})
    ai_optimizer: AIAutoOptimizer | None = None
    if openai_cfg.get("enabled", False):
        ai_optimizer = AIAutoOptimizer(
            state_manager=state_manager,
            strategy_manager=strategy_manager,
            openai_config=openai_cfg,
        )

    backtesting_cfg = config.get("backtesting", {})
    backtesting_engine = BacktestingEngine(
        initial_cash=backtesting_cfg.get("initial_cash", 10_000.0),
        fee_rate=backtesting_cfg.get("fee_rate", trading_cfg.get("fee_rate", 0.0005)),
        strategy_configs=strategy_cfg.get("strategies"),
        risk_manager=risk_manager,
        symbol_rules=symbol_rules,
    )

    scheduler_cfg = config.get("scheduler", {})
    scheduler = Scheduler(
        state_manager=state_manager,
        exchange_client=exchange_client,
        strategy_manager=strategy_manager,
        paper_trading_engine=paper_trading_engine,
        live_trading_engine=live_trading_engine,
        evaluator=evaluator,
        reporter=reporter,
        ai_optimizer=ai_optimizer,
        config=config,
        ml_trainer=ml_trainer,
        ml_predictor=ml_predictor,
        rl_optimizer=rl_optimizer,
        backtesting_engine=backtesting_engine,
        market_data_store=market_data_store,
        trading_interval_seconds=scheduler_cfg.get("trading_interval_seconds", 30),
        daily_run_time=scheduler_cfg.get("daily_run_time", "23:55"),
    )

    frontend_thread: threading.Thread | None = None
    if config.get("frontend", {}).get("enabled", False):
        frontend_thread = _start_frontend(
            state_manager=state_manager,
            strategy_manager=strategy_manager,
            scheduler=scheduler,
            frontend_cfg=config.get("frontend", {}),
            exchange_client=exchange_client,
            ml_trainer=ml_trainer,
            ml_predictor=ml_predictor,
            backtesting_engine=backtesting_engine,
            market_data_store=market_data_store,
            config=config,
        )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received; stopping services.")
    finally:
        scheduler.stop()
        if frontend_thread:
            logging.info("Frontend server thread will exit when main process ends.")


def _load_config() -> Dict[str, Any]:
    """Load YAML configuration from config.yaml or fallback to sample."""
    config_path = Path("src/config/config.yaml")
    if not config_path.exists():
        logging.warning("config.yaml not found, falling back to sample configuration.")
        config_path = Path("src/config/config.sample.yaml")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _start_frontend(
    *,
    state_manager: StateManager,
    strategy_manager: StrategyManager,
    scheduler: Scheduler,
    frontend_cfg: Dict[str, Any],
    exchange_client: BinanceClient,
    ml_trainer: LightGBMTrainer,
    ml_predictor: TrendModelPredictor | None,
    backtesting_engine: BacktestingEngine,
    market_data_store: MarketDataStore | None,
    config: Dict[str, Any],
) -> threading.Thread:
    """Start the FastAPI dashboard server in a background thread."""
    import uvicorn

    app = create_app(
        state_manager=state_manager,
        strategy_manager=strategy_manager,
        scheduler=scheduler,
        exchange_client=exchange_client,
        ml_trainer=ml_trainer,
        ml_predictor=ml_predictor,
        backtesting_engine=backtesting_engine,
        market_data_store=market_data_store,
        config=config,
    )
    host = frontend_cfg.get("host", "127.0.0.1")
    port = frontend_cfg.get("port", 8000)

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, name="frontend-server", daemon=True)
    thread.start()
    logging.info("Frontend server running at http://%s:%s", host, port)
    return thread


if __name__ == "__main__":
    main()
