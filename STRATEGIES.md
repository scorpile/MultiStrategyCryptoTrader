# STRATEGIES — Cómo añadir una estrategia nueva

Las estrategias son **plugins internos** (hot‑swappable) que **solo generan señales**. La ejecución (paper/live), riesgo, SL/TP, throttles, gate y persistencia viven fuera de la estrategia (scheduler + engines + SQLite).

Carpeta: `src/strategy/`

Registro: `src/strategy/registry.py`

---

## 1) Contrato de una estrategia

Toda estrategia implementa `BaseStrategy` (`src/strategy/strategy_base.py`) y debe:

- `name() -> str` (ID único)
- `config_schema() -> dict` (defaults para UI)
- `generate_signal(indicators, context=None) -> dict`

La salida debe ser un dict con estas claves (usa `self._build_signal(...)` para asegurar formato):

- `decision`: `"buy" | "sell" | "hold"`
- `price`: `float` (precio “de intención” de la señal; el engine aplica slippage/spread)
- `confidence`: `0..1`
- `suggested_size`: cantidad (base asset) **opcional**. Si pones `0`, el scheduler hará **risk sizing**.
- `reasons`: lista de strings (debuggable)
- `metadata`: dict serializable (se persistirá en DB en muchos casos)

> El bot es **spot long‑only** por defecto: si no hay posición, un `sell` se ignora.

---

## 2) Inputs disponibles: `indicators`

El scheduler construye un payload en `src/core/scheduler.py:_build_indicator_payload()` con campos típicos:

- `ohlcv`: lista de velas dict (`open_time`, `open`, `high`, `low`, `close`, `volume`, …)
- `closes`, `highs`, `lows`, `volumes`: listas alineadas
- `rsi`: lista (periodo depende de la estrategia activa)
- `macd`: dict con listas (`macd`, `signal`, `histogram`)
- `bollinger`: dict con listas (`lower`, `middle`, `upper`)
- `atr`: lista (periodo depende de la estrategia activa)
- `ema_fast`, `ema_slow`: listas (periodos dependen de la estrategia activa)
- `volume_sma`: lista (periodo depende de la estrategia activa)
- Extras: `doji`, `engulfing`, `macd_histogram`, `bollinger_width`, `vwap`, `features_df`

Regla práctica:
- Asume que `closes` existe y es la fuente principal para precio.
- Si vas a recalcular indicadores, hazlo solo como fallback (cuando el payload no trae la serie esperada).

---

## 3) Inputs disponibles: `context`

`StrategyManager` llama a `generate_signal(..., context=...)` y agrega:

- `capital`: equity/cash aproximado para sizing (puede ser `None`)
- `position_qty`: qty actual de la posición (si el scheduler lo pasa como extra_context)
- `ml`: contexto ML (probabilidades/confianza/model_name)
- `rl`: contexto RL (policy/score; si está habilitado)
- `exploration`: flag para modo “aventurero” (cuando aplica)

Ejemplo de lectura:

```python
ctx = context or {}
position_qty = float(ctx.get("position_qty", 0.0) or 0.0)
ml = ctx.get("ml") or {}
prob_up = float(ml.get("probability_up", 0.5) or 0.5)
```

---

## 4) SL/TP y Risk Sizing (recomendado)

Hay dos caminos válidos:

### A) Dejar que el scheduler calcule SL/TP

En tu `config_schema()` define:

- `stop_atr_multiplier`
- `profit_target_multiplier`
- `risk_pct_per_trade` (opcional)

Y devuelve señales de BUY con `suggested_size=0.0`.

El scheduler:
- calcula `stop_loss` y `take_profit` (ATR‑based),
- hace sizing por riesgo usando `RiskManager.size_position_for_risk(...)`,
- adjunta todo a `SimulatedOrder.metadata` (queda persistido en DB).

### B) Emitir SL/TP absolutos desde la estrategia

En el `metadata` del BUY agrega:

- `stop_loss`: precio stop absoluto
- `take_profit`: precio TP absoluto
- (opcional) `risk_pct_per_trade`

Esto es útil si tu estrategia tiene reglas explícitas (ej: Bollinger range “TP en middle band”).

---

## 5) Pasos para añadir una estrategia

### Paso 1 — Crear el archivo

Crea un archivo nuevo, por ejemplo `src/strategy/my_strategy.py`:

```python
from __future__ import annotations

from typing import Any, Dict, Optional

from .strategy_base import BaseStrategy


class MyStrategy(BaseStrategy):
    description = "Mi estrategia: describe la idea en 1 línea."

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

        # TODO: tu lógica aquí
        if position_qty <= 0:
            return self._build_signal(
                decision="buy",
                price=price,
                confidence=0.55,
                suggested_size=0.0,  # <- deja que el scheduler haga risk sizing
                reasons=["Example buy"],
                metadata={"risk_pct_per_trade": float(self.config.get("risk_pct_per_trade", 0.005))},
            )
        return self._build_signal(decision="hold", price=price, confidence=0.2, suggested_size=0.0, reasons=["No signal"])
```

Buenas prácticas:
- Mantén `generate_signal()` determinista y rápida.
- No hagas I/O (red, disco) desde la estrategia.
- No ejecutes órdenes reales (la estrategia solo decide).

### Paso 2 — Registrar la estrategia

Agrega tu clase al registry:

- `src/strategy/registry.py`

Ejemplo:

```python
from .my_strategy import MyStrategy

STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

### Paso 3 — Configurar defaults/overrides

En `src/config/config.yaml`:

```yaml
strategy:
  strategies:
    my_strategy:
      rsi_period: 14
      stop_atr_multiplier: 2.0
      profit_target_multiplier: 3.0
      risk_pct_per_trade: 0.005
```

> La UI muestra la config_schema + overrides y permite switching en caliente.

### Paso 4 — (Opcional) Hacerla “tuneable” en el backtest diario

Si quieres que el optimizador daily explore parámetros, agrega una grilla en:

- `src/backtesting/walkforward_runner.py:_default_grid()`

Esto habilita el grid/walk-forward para tu estrategia con límites (`backtesting.auto.grid_cap`).

---

## 6) Debugging / Validación rápida

- **Dashboard → Activity / Trades**: confirma que genera señales y que se ejecutan.
- **Bloqueos**: revisa `decision_events` (gate, cooldown, risk_pause, manual_pause, etc.).
- Backtest rápido:
  - `GET /api/backtest/run?strategy=my_strategy&interval=1m&days=30`

Si ves “Backtest: n/a” en el dashboard, normalmente significa:
- aún no corrió el backtest diario, o
- falló y guardó `backtest_last.error` en `system_parameters`.

---

## 7) Checklist antes de darla por “lista”

- Devuelve siempre `price` válido cuando sea BUY/SELL.
- No crashea cuando faltan series (maneja listas vacías).
- Es consistente con spot long‑only (SELL solo cuando hay posición).
- Define claramente cómo se calcula SL/TP (A o B).
- Produce `reasons` útiles (te ahorra horas).

