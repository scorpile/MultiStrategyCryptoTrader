# STRATEGIES — How to Add a New Strategy

Strategies are **internal plugins** (hot-swappable) that **only generate signals**. Execution (paper/live), risk, SL/TP, throttles, gates, and persistence live outside the strategy (scheduler + engines + SQLite).

Folder: `src/strategy/`

Registry: `src/strategy/registry.py`

---

## 1) Strategy contract

Every strategy implements `BaseStrategy` (`src/strategy/strategy_base.py`) and must provide:

- `name() -> str` (unique ID)
- `config_schema() -> dict` (defaults used by the UI)
- `generate_signal(indicators, context=None) -> dict`

The output must be a `dict` with these keys (use `self._build_signal(...)` to ensure a consistent format):

- `decision`: `"buy" | "sell" | "hold"`
- `price`: `float` (signal “intent” price; the engine applies slippage/spread)
- `confidence`: `0..1`
- `suggested_size`: base-asset quantity (**optional**). If you return `0`, the scheduler will do **risk sizing**.
- `reasons`: list of strings (debug-friendly)
- `metadata`: JSON-serializable dict (often persisted in DB)

> The bot is **spot long-only** by default: if there is no position, a `sell` is ignored.

---

## 2) Available inputs: `indicators`

The scheduler builds an indicator payload in `src/core/scheduler.py:_build_indicator_payload()` with common fields like:

- `ohlcv`: list of candle dicts (`open_time`, `open`, `high`, `low`, `close`, `volume`, ...)
- `closes`, `highs`, `lows`, `volumes`: aligned lists
- `rsi`: list (period depends on the active strategy)
- `macd`: dict with lists (`macd`, `signal`, `histogram`)
- `bollinger`: dict with lists (`lower`, `middle`, `upper`)
- `atr`: list (period depends on the active strategy)
- `ema_fast`, `ema_slow`: lists (period depends on the active strategy)
- `volume_sma`: list (period depends on the active strategy)
- Extras: `doji`, `engulfing`, `macd_histogram`, `bollinger_width`, `vwap`, `features_df`

Rule of thumb:
- Assume `closes` exists and use it as your primary price source.
- If you recompute indicators, do it only as a fallback (when the payload doesn’t include the series you need).

---

## 3) Available inputs: `context`

`StrategyManager` calls `generate_signal(..., context=...)` and may include:

- `capital`: approximate equity/cash for sizing (may be `None`)
- `position_qty`: current position size (if the scheduler passes it via `extra_context`)
- `ml`: ML context (probabilities/confidence/model_name)
- `rl`: RL context (policy/score; if enabled)
- `exploration`: flag for “adventurous” mode (when enabled)

Example:

```python
ctx = context or {}
position_qty = float(ctx.get("position_qty", 0.0) or 0.0)
ml = ctx.get("ml") or {}
prob_up = float(ml.get("probability_up", 0.5) or 0.5)
```

---

## 4) SL/TP and risk sizing (recommended)

There are two valid paths:

### A) Let the scheduler compute SL/TP

In your `config_schema()` define:

- `stop_atr_multiplier`
- `profit_target_multiplier`
- `risk_pct_per_trade` (optional)

Return BUY signals with `suggested_size=0.0`.

The scheduler:
- computes `stop_loss` and `take_profit` (ATR-based),
- sizes the position by risk via `RiskManager.size_position_for_risk(...)`,
- attaches everything to `SimulatedOrder.metadata` (persisted in DB).

### B) Emit absolute SL/TP from the strategy

In the BUY `metadata` include:

- `stop_loss`: absolute stop price
- `take_profit`: absolute take-profit price
- (optional) `risk_pct_per_trade`

This is useful when the strategy has explicit rules (e.g., Bollinger range: TP at middle band).

---

## 5) Steps to add a strategy

### Step 1 — Create the file

Create a new file, for example `src/strategy/my_strategy.py`:

```python
from __future__ import annotations

from typing import Any, Dict, Optional

from .strategy_base import BaseStrategy


class MyStrategy(BaseStrategy):
    description = "My strategy: describe the idea in 1 line."

    def name(self) -> str:
        return "my_strategy"

    def config_schema(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "stop_atr_multiplier": 2.0,
            "profit_target_multiplier": 3.0,
            "risk_pct_per_trade": 0.005,
        }

    def generate_signal(self, indicators: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        closes = indicators.get("closes") or []
        if not closes:
            return self._build_signal(decision="hold", price=0.0, confidence=0.0, suggested_size=0.0, reasons=["No closes"])

        price = float(closes[-1])
        ctx = context or {}
        position_qty = float(ctx.get("position_qty", 0.0) or 0.0)

        # TODO: your logic here
        if position_qty <= 0:
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=0.55,
                suggested_size=0.0,  # <- let the scheduler do risk sizing
                reasons=["Example buy"],
                metadata={"risk_pct_per_trade": float(self.config.get("risk_pct_per_trade", 0.005))},
            )
        return self._build_signal(decision="hold", price=price, confidence=0.2, suggested_size=0.0, reasons=["No signal"])
```

Best practices:
- Keep `generate_signal()` deterministic and fast.
- Do not do I/O (network, disk) inside the strategy.
- The strategy decides; it should not execute orders directly.

### Step 2 — Register the strategy

Add your class to the registry in `src/strategy/registry.py`:

```python
from .my_strategy import MyStrategy

STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

### Step 3 — Configure defaults/overrides

In `src/config/config.yaml`:

```yaml
strategy:
  strategies:
    my_strategy:
      rsi_period: 14
      stop_atr_multiplier: 2.0
      profit_target_multiplier: 3.0
      risk_pct_per_trade: 0.005
```

> The UI shows `config_schema` + overrides and supports hot switching.

### Step 4 — (Optional) Make it tuneable in the daily backtest

If you want the daily optimizer to explore parameters, add a grid in:

- `src/backtesting/walkforward_runner.py:_default_grid()`

This enables grid + walk-forward for your strategy, within caps (`backtesting.auto.grid_cap`).

---

## 6) Debugging / quick validation

- **Dashboard → Activity / Trades**: confirm it generates signals and that they execute.
- **Blocks**: inspect `decision_events` (gate, cooldown, risk_pause, manual_pause, etc.).
- Quick backtest:
  - `GET /api/backtest/run?strategy=my_strategy&interval=1m&days=30`

If you see “Backtest: n/a” in the dashboard, it usually means:
- the daily backtest has not run yet, or
- it failed and stored `backtest_last.error` under `system_parameters`.

---

## 7) Checklist before calling it “done”

- Always return a valid `price` for BUY/SELL.
- Never crash when series are missing (handle empty lists).
- Respect spot long-only (SELL only when there is a position).
- Be explicit about how SL/TP is computed (A or B).
- Produce useful `reasons` (saves hours).
