# Database / Persistence (SQLite)

This project uses **two local SQLite** databases:

1) `data/state.db` — *operational state and traceability* (trades, metrics, events, parameters, execution ledger).
2) `data/market_data.sqlite` — *OHLCV cache* for longer-horizon training/backtesting (avoids API limits).

Both databases are local, and the bot can restart and **rehydrate** its state from them.

---

## `data/state.db` (StateManager)

Created/managed by `src/core/state_manager.py`.

Notes:
- `metadata` fields are usually stored as **JSON strings** (serialized with `json.dumps`).
- `system_parameters.value` is always a **JSON string**.
- Timestamps are stored as ISO-8601 strings with a `Z` suffix (UTC).
- The connection uses `PRAGMA journal_mode = WAL;` to improve concurrency.

### Table: `system_parameters`

Persistent key/value store for flags, snapshots, and internal state.

| Column | Type | Description |
|---|---|---|
| `key` | TEXT (PK) | Parameter name |
| `value` | TEXT | Serialized JSON |
| `updated_at` | TEXT | ISO UTC |

Common keys:
- `execution_mode`, `risk_pause`, `manual_pause`, `backtest_last`, `ml_status`, `market_data_last_sync`, etc.

### Table: `simulated_trades`

**Paper** trade ledger (simulated BUY/SELL) with fees and metadata (stop/take/sizing/execution model).

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER (PK) | Auto-increment |
| `trade_time` | TEXT | ISO UTC |
| `symbol` | TEXT | e.g. `SOLUSDT` |
| `side` | TEXT | `BUY` / `SELL` |
| `quantity` | REAL | Base-asset quantity |
| `price` | REAL | Filled price (with simulated slippage/spread) |
| `fee` | REAL | Simulated fee |
| `pnl` | REAL | Per-trade PnL (net on SELL in recent versions) |
| `metadata` | TEXT | JSON string (stop_loss, take_profit, sizing, etc.) |
| `created_at` | TEXT | ISO UTC |

### Table: `daily_metrics`

Daily aggregated metrics (used by the dashboard and reports).

| Column | Type |
|---|---|
| `id` | INTEGER (PK) |
| `metric_date` | TEXT (UNIQUE) |
| `pnl` | REAL |
| `trades_count` | INTEGER |
| `win_rate` | REAL |
| `drawdown` | REAL |
| `sharpe` | REAL |
| `notes` | TEXT |
| `created_at` | TEXT |
| `expectancy` | REAL |
| `profit_factor` | REAL |
| `avg_win` | REAL |
| `avg_loss` | REAL |

> Some columns are added via `ALTER TABLE ... ADD COLUMN ...` (best-effort migrations).

### Table: `probability_flags`

System flags (e.g., “profit consistency” / readiness flags).

| Column | Type |
|---|---|
| `flag_name` | TEXT (PK) |
| `is_enabled` | INTEGER |
| `probability` | REAL |
| `updated_at` | TEXT |
| `metadata` | TEXT (JSON string) |

### Table: `decision_events`

Audit log of strategy decisions: executed or blocked (gate, cooldown, risk pause, etc.).

| Column | Type |
|---|---|
| `id` | INTEGER (PK) |
| `event_time` | TEXT |
| `symbol` | TEXT |
| `strategy` | TEXT |
| `decision` | TEXT |
| `price` | REAL |
| `suggested_size` | REAL |
| `confidence` | REAL |
| `executed` | INTEGER |
| `blocked_reason` | TEXT |
| `trade_time` | TEXT |
| `metadata` | TEXT (JSON string) |
| `created_at` | TEXT |

### Table: `equity_snapshots`

Equity time series for charts/diagnostics.

| Column | Type |
|---|---|
| `id` | INTEGER (PK) |
| `snapshot_time` | TEXT |
| `symbol` | TEXT |
| `total_value` | REAL |
| `cash_balance` | REAL |
| `realized_pnl` | REAL |
| `open_positions_count` | INTEGER |
| `metadata` | TEXT (JSON string) |
| `created_at` | TEXT |

### Table: `execution_trades`

Execution ledger for **live** (Binance Spot) and/or “paper comparable” execution when enabled.

| Column | Type |
|---|---|
| `id` | INTEGER (PK) |
| `trade_time` | TEXT |
| `mode` | TEXT (`live`/`paper`) |
| `symbol` | TEXT |
| `side` | TEXT |
| `quantity` | REAL |
| `price` | REAL |
| `fee` | REAL |
| `pnl` | REAL |
| `order_id` | TEXT |
| `client_order_id` | TEXT |
| `metadata` | TEXT (JSON string) |
| `created_at` | TEXT |

### Table: `execution_positions`

“Live” position state (and a compatibility layer to persist execution-computed positions).

Composite PK: (`symbol`, `mode`)

| Column | Type |
|---|---|
| `symbol` | TEXT |
| `mode` | TEXT |
| `quantity` | REAL |
| `avg_price` | REAL |
| `cost_basis_usdt` | REAL |
| `stop_loss` | REAL |
| `take_profit` | REAL |
| `updated_at` | TEXT |
| `metadata` | TEXT (JSON string) |

---

## `data/market_data.sqlite` (MarketDataStore)

Created/managed by `src/data/market_data_store.py`.

Main table: `ohlcv_candles`

Composite PK: (`symbol`, `interval`, `open_time_ms`)

| Column | Type |
|---|---|
| `symbol` | TEXT |
| `interval` | TEXT |
| `open_time_ms` | INTEGER |
| `open_time` | TEXT |
| `open` | REAL |
| `high` | REAL |
| `low` | REAL |
| `close` | REAL |
| `volume` | REAL |
| `close_time_ms` | INTEGER |
| `close_time` | TEXT |
| `quote_asset_volume` | REAL |
| `number_of_trades` | INTEGER |

How it’s populated:
- The bot synchronizes ranges via `ensure_range(...)` and paginates using `startTime`.
- If the cache is not enabled, it falls back to `fetch_ohlcv(limit=N)` (API-limited).

---

## Reset (wipe) & traceability

Reset wipes data but keeps the schema (useful to “start over” without breaking compatibility):

- DB wipe: `StateManager.reset_db()`
- API: `POST /api/admin/reset` with body `{"confirm":"RESET"}`

---

## Quick inspection (SQLite)

If you have `sqlite3` installed:

```bash
sqlite3 data/state.db ".tables"
sqlite3 data/state.db "select * from simulated_trades order by trade_time desc limit 5;"
sqlite3 data/state.db "select metric_date,pnl,sharpe,profit_factor from daily_metrics order by metric_date desc limit 10;"
```

For the OHLCV cache:

```bash
sqlite3 data/market_data.sqlite "select count(*) from ohlcv_candles where symbol='SOLUSDT' and interval='1m';"
```
