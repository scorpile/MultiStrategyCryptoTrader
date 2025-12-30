
# Autonomous Crypto Trading Bot (Paper‑First) — Dashboard + Auto‑Tune + Backtesting

Un bot de trading **modular** para **SOL/USDT** (y pares similares) pensado para iterar rápido:

- **Paper trading por defecto** (simulación con spread/slippage/fills) y estado persistente en SQLite.
- **Dashboard web** (FastAPI + Jinja2) para ver señales, trades, PnL, gate/risk pauses, entrenamiento, etc.
- **Auto‑Tune** (reglas simples) y **optimización diaria** (walk‑forward + grid) para ajustar parámetros.
- **ML opcional** (LightGBM) y **sentiment opcional** (Fear & Greed) integrados al contexto de decisiones.
- Modo **Live (Binance Spot)** disponible, pero **bloqueado por seguridad** (requiere armar explícitamente + confirmación UI).

> Este repo está diseñado para aprender/iterar en simulación. El modo live existe, pero úsalo bajo tu propio riesgo.

---

## Qué incluye (high level)

- **Estrategias hot‑swappable** desde UI (RSI/MACD/Bollinger, scalping agresivo, EMA+RSI+volumen, Bollinger range, multi‑factor).
- **Risk sizing** por trade (en función de stop‑loss) + **stop_loss / take_profit** persistidos en metadata de cada trade.
- **Trading Gate** (bloquea nuevas entradas cuando las métricas están mal) + **pausas persistidas** (manual y por riesgo).
- **Backtesting**
  - On‑demand desde API.
  - Optimización diaria: walk‑forward + grid (y aplica “best config” si cumple umbrales).
- **Persistencia y trazabilidad**:
  - Ledger de trades simulados
  - Eventos de decisiones (ejecutadas/bloqueadas + razón)
  - Snapshots de equity
  - Parámetros del sistema

---

## Requisitos

- Python **3.11+** recomendado
- Windows / Linux / macOS

---

## Setup

### 1) Crear entorno e instalar dependencias

```bash
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configuración

El bot lee `src/config/config.yaml` (si existe). Si no existe, usa `src/config/config.sample.yaml`.

Recomendado para GitHub:

```bash
cp src/config/config.sample.yaml src/config/config.yaml
```

**Windows (PowerShell)**

```powershell
Copy-Item src\\config\\config.sample.yaml src\\config\\config.yaml
```

Edita `src/config/config.yaml` y ajusta:

- `general.base_symbol`: por defecto `SOLUSDT`
- `general.time_frame`: por defecto `1m` (scalping)
- `frontend.host/port`: por defecto `127.0.0.1:8000`
- (Opcional) `openai.enabled` + `OPENAI_API_KEY`
- (Opcional) `sentiment.provider` + API key si aplica

**API keys**

- Para **paper/backtest** no necesitas keys de Binance (los endpoints públicos de mercado funcionan sin autenticación).
- Si vas a usar **Live**, agrega tus keys en `exchange.api_key` / `exchange.api_secret` y usa testnet si quieres (`exchange.use_sandbox: true`).
- No commitees `src/config/config.yaml` con secretos. Usa el sample + `.gitignore` (recomendado).

---

## Ejecución

Un único entrypoint:

```bash
python run.py
```

Luego abre el dashboard:

- `http://127.0.0.1:8000/`

---

## Dashboard + API

El dashboard hace polling a `GET /api/status`.

Endpoints útiles:

- `GET /api/strategy/list` — lista estrategias disponibles
- `POST /api/strategy/set` — cambia estrategia (y opcionalmente overrides de config)
- `POST /api/training/start` — ejecuta las tareas “daily” (entrenamiento/optimización/reporte)
- `GET /api/backtest/run?strategy=ema_rsi_volume&interval=1m&days=60` — backtest rápido on‑demand
- `POST /api/trading/pause` — pausa manual de nuevas entradas
- `POST /api/risk/clear` — limpia una pausa de riesgo persistida
- `POST /api/admin/reset` — resetea estado (requiere confirmación `"RESET"`)
- `POST /api/execution/mode` — cambia `paper` / `live` (requiere confirmación `"LIVE"` para live)

Ejemplos:

```bash
curl "http://127.0.0.1:8000/api/backtest/run?strategy=ema_rsi_volume&interval=1m&days=30"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/strategy/set" -H "Content-Type: application/json" -d '{"strategy":"ema_rsi_volume","config":{"volume_spike_multiplier":1.2}}'
```

---

## Modo Live (Binance Spot) — cómo está protegido

Por defecto el bot corre en `execution.mode: paper`.

Para **habilitar live** hay 2 barreras:

1) **Arming por env var** (por defecto `BINANCE_LIVE_ARMED=1`, configurable en `execution.live.require_env_var`)
2) **Confirmación explícita** (UI/API requiere `confirm: "LIVE"`)

Además, live aplica límites adicionales:

- `execution.live.allow_symbol` (por defecto solo `SOLUSDT`)
- `execution.live.max_notional_usdt_per_trade`
- `execution.live.max_daily_loss_usdt`
- `execution.live.max_orders_per_day`

---

## Persistencia (SQLite) y artefactos

- Estado/ledger principal: `data/state.db`  
- Cache OHLCV largo (para backtesting/training): `data/market_data.sqlite`  
- Runtime config / políticas: `state/`  
- Reportes diarios: `reports/`  
- Export de backtests: `backtests/`  

Detalles: ver `DB.md`.

---

## Añadir una estrategia nueva

Guía completa en `STRATEGIES.md`.

Resumen:

1) Implementa una clase en `src/strategy/*.py` (subclase de `BaseStrategy`)
2) Regístrala en `src/strategy/registry.py`
3) Agrega defaults/overrides en `src/config/config.yaml` → `strategy.strategies.<tu_estrategia>`

---

## Tests

```bash
pytest
```

---

## Disclaimer

Este proyecto es para investigación/educación. No es asesoría financiera. Si habilitas live, eres responsable de cualquier pérdida.
