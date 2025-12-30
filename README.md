# MultiStrategyCryptoTrader

Paper-first, modular crypto trading bot with a live dashboard, auto-tuning, and walk-forward backtesting.

- ✅ **Paper trading by default** (simulated fills + fees), with full state persistence via SQLite.
- 📊 **Web dashboard** (FastAPI + Jinja2) to monitor PnL, signals, trades, gates/pauses, training, and backtests.
- 🧠 **Auto-tune** (practical heuristics) + **daily optimization** (walk-forward + grid) to evolve parameters.
- 🤖 Optional **ML context** (LightGBM) + optional **sentiment** (Fear & Greed) to enrich decisions.
- 🔒 **Live (Binance Spot)** is available, but intentionally **guarded** (explicit arming + UI confirmation).

> Designed to iterate safely in paper mode first. Live trading is optional and always at your own risk.

---

## What’s Included

- **Hot-swappable strategies** from the UI (RSI/MACD/Bollinger, aggressive scalping, EMA+RSI+volume, Bollinger range, multi-factor).
- **Risk-based position sizing** (based on stop-loss distance) + **stop_loss / take_profit** persisted in each trade’s metadata.
- **Trading Gate** (blocks new entries when metrics degrade) + **persisted pauses** (manual + risk circuit-breakers).
- **Backtesting**
  - On-demand via API.
  - Daily optimization: walk-forward + grid (optionally applies “best config” if it meets thresholds).
- **Persistence & traceability**
  - Paper trade ledger
  - Decision events (executed/blocked + reason)
  - Equity snapshots
  - System parameters

---

## Requirements

- Python **3.11+** recommended (Windows / Linux / macOS)

---

## Setup

### 1) Create a venv + install deps

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

### 2) Configure

The bot reads `src/config/config.yaml` (if present). If not present, it falls back to `src/config/config.sample.yaml`.

Recommended workflow for GitHub:

```bash
cp src/config/config.sample.yaml src/config/config.yaml
```

**Windows (PowerShell)**

```powershell
Copy-Item src\config\config.sample.yaml src\config\config.yaml
```

Edit `src/config/config.yaml` and adjust:

- `general.base_symbol` (default: `SOLUSDT`)
- `general.time_frame` (default: `1m`)
- `frontend.host` / `frontend.port` (default: `127.0.0.1:8000`)
- (Optional) `openai.enabled` + `OPENAI_API_KEY`
- (Optional) `sentiment.provider` + provider API key (if required)

**API keys**

- For **paper/backtests**, you don’t need Binance keys (public market endpoints work without auth).
- For **live**, set `exchange.api_key` / `exchange.api_secret` and consider sandbox/testnet if supported.
- Do **not** commit `src/config/config.yaml` with secrets. Use the sample + `.gitignore` (recommended).

---

## Run

Single entrypoint:

```bash
python run.py
```

Then open the dashboard:

- `http://127.0.0.1:8000/`

---

## Dashboard & API

The dashboard polls `GET /api/status`.

Useful endpoints:

- `GET /api/strategy/list` — list available strategies
- `POST /api/strategy/set` — set strategy (and optional config overrides)
- `POST /api/training/start` — run “daily” tasks (training/optimization/report)
- `GET /api/backtest/run?strategy=ema_rsi_volume&interval=1m&days=60` — quick on-demand backtest
- `POST /api/trading/pause` — pause new entries manually
- `POST /api/risk/clear` — clear a persisted risk pause
- `POST /api/admin/reset` — reset state (requires confirmation `"RESET"`)
- `POST /api/execution/mode` — switch `paper` / `live` (requires confirmation `"LIVE"` for live)

Examples:

```bash
curl "http://127.0.0.1:8000/api/backtest/run?strategy=ema_rsi_volume&interval=1m&days=30"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/strategy/set" -H "Content-Type: application/json" -d '{"strategy":"ema_rsi_volume","config":{"volume_spike_multiplier":1.2}}'
```

---

## Live Mode (Binance Spot) — Safety Guards

By default the bot runs with `execution.mode: paper`.

To enable **live**, there are two gates:

1) **Arming via env var** (default requires `BINANCE_LIVE_ARMED=1`, configurable via `execution.live.require_env_var`)
2) **Explicit confirmation** (UI/API requires `confirm: "LIVE"`)

Live mode also enforces additional limits:

- `execution.live.allow_symbol` (default: only `SOLUSDT`)
- `execution.live.max_notional_usdt_per_trade`
- `execution.live.max_daily_loss_usdt`
- `execution.live.max_orders_per_day`

---

## Persistence (SQLite) & Artifacts

- Main state/ledger: `data/state.db`
- Long-horizon OHLCV cache (training/backtests): `data/market_data.sqlite`
- Runtime config/policies: `state/`
- Daily reports: `reports/`
- Backtest exports: `backtests/`

Details: see `DB.md`.

---

## Adding a New Strategy

Full guide: `STRATEGIES.md`.

Quick summary:

1) Implement a class in `src/strategy/*.py` (subclass `BaseStrategy`)
2) Register it in `src/strategy/registry.py`
3) Add defaults/overrides in `src/config/config.yaml` → `strategy.strategies.<your_strategy>`

---

## Tests

```bash
pytest
```

---

## 🛠️ Development

### Contributing

Contributions are welcome! Whether it's reporting bugs, suggesting features, or submitting pull requests, your support is appreciated.

---

## 💖 Donations

If you find this project useful and would like to support its development, consider donating. Your contributions help in maintaining and enhancing the bot.

Binance ID: **322411022**

![Binance QR](binance.jpg)

---

## ⚠️ Disclaimer

Important:

MultiStrategyCryptoTrader is a work in progress. It does not guarantee any profits. Trading cryptocurrencies involves significant risk, and you may lose money. Use this bot at your own risk. The developer is not responsible for any financial losses incurred while using this bot.
