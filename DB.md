# DB / Persistencia (SQLite)

Este proyecto usa **dos SQLite** locales:

1) `data/state.db` — *estado operativo y trazabilidad* (trades, métricas, eventos, parámetros, live ledger).
2) `data/market_data.sqlite` — *cache OHLCV* para entrenamiento/backtesting de largo horizonte (evita límites de API).

Ambas bases son locales y el bot puede reiniciar y **rehidratar** estado desde ellas.

---

## `data/state.db` (StateManager)

Creada/gestionada por `src/core/state_manager.py`.

Notas:
- `metadata` suele ser **JSON string** (serializado con `json.dumps`).
- `system_parameters.value` siempre es **JSON string**.
- Los timestamps se guardan en ISO‑8601 con sufijo `Z` (UTC).
- La conexión usa `PRAGMA journal_mode = WAL;` para mejorar concurrencia.

### Tabla: `system_parameters`
Key/Value store persistente para flags, snapshots y estado interno.

| Columna | Tipo | Descripción |
|---|---|---|
| `key` | TEXT (PK) | Nombre del parámetro |
| `value` | TEXT | JSON serializado |
| `updated_at` | TEXT | ISO UTC |

Ejemplos de keys comunes:
- `execution_mode`, `risk_pause`, `manual_pause`, `backtest_last`, `ml_status`, `market_data_last_sync`, etc.

### Tabla: `simulated_trades`
Ledger de operaciones **paper** (BUY/SELL simulados) con fees y metadata (stop/take/sizing/execution model).

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER (PK) | Autoincrement |
| `trade_time` | TEXT | ISO UTC |
| `symbol` | TEXT | Ej: `SOLUSDT` |
| `side` | TEXT | `BUY` / `SELL` |
| `quantity` | REAL | Qty en base asset |
| `price` | REAL | Precio ejecutado (con slippage/spread simulado) |
| `fee` | REAL | Fee simulado |
| `pnl` | REAL | PnL por trade (neto en SELL en versiones recientes) |
| `metadata` | TEXT | JSON string (stop_loss, take_profit, sizing, etc.) |
| `created_at` | TEXT | ISO UTC |

### Tabla: `daily_metrics`
Métricas agregadas por día (usado por dashboard/reportes).

| Columna | Tipo |
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

> Algunas columnas se agregan vía `ALTER TABLE ... ADD COLUMN ...` (migraciones best‑effort).

### Tabla: `probability_flags`
Flags “de sistema” (ej: bandera de “profit consistency”).

| Columna | Tipo |
|---|---|
| `flag_name` | TEXT (PK) |
| `is_enabled` | INTEGER |
| `probability` | REAL |
| `updated_at` | TEXT |
| `metadata` | TEXT (JSON string) |

### Tabla: `decision_events`
Audit log de decisiones de estrategia: ejecutadas o bloqueadas (gate, cooldown, risk pause, etc.).

| Columna | Tipo |
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

### Tabla: `equity_snapshots`
Serie de equity para gráficos/diagnóstico.

| Columna | Tipo |
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

### Tabla: `execution_trades`
Ledger de operaciones **live** (Binance Spot) y/o “paper comparable” cuando se habilita.

| Columna | Tipo |
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

### Tabla: `execution_positions`
Estado de posición “live” (y compatibilidad para persistir posiciones calculadas por ejecución).

PK compuesta: (`symbol`, `mode`)

| Columna | Tipo |
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

Creada/gestionada por `src/data/market_data_store.py`.

Tabla principal: `ohlcv_candles`

PK compuesta: (`symbol`, `interval`, `open_time_ms`)

| Columna | Tipo |
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

Cómo se llena:
- El bot sincroniza rangos vía `ensure_range(...)` y usa paginación por `startTime`.
- Si el cache no está habilitado, se cae a `fetch_ohlcv(limit=N)` (limitado por API).

---

## Reset (wipe) y trazabilidad

El reset borra datos pero mantiene el schema (útil para “volver a cero” sin perder compatibilidad):

- DB wipe: `StateManager.reset_db()`
- API: `POST /api/admin/reset` con body `{"confirm":"RESET"}`

---

## Inspección rápida (SQLite)

Si tienes `sqlite3` instalado:

```bash
sqlite3 data/state.db ".tables"
sqlite3 data/state.db "select * from simulated_trades order by trade_time desc limit 5;"
sqlite3 data/state.db "select metric_date,pnl,sharpe,profit_factor from daily_metrics order by metric_date desc limit 10;"
```

Para el cache OHLCV:

```bash
sqlite3 data/market_data.sqlite "select count(*) from ohlcv_candles where symbol='SOLUSDT' and interval='1m';"
```

