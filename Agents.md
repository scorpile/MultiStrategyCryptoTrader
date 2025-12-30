# Agents – Autonomous Crypto Trading Bot (Python)

Este archivo define los agentes, perfiles, responsabilidades, flujos y límites para que Codex pueda construir y coordinar el proyecto completo.

El objetivo general es crear un bot de trading **simulado** para el par SOL/USDT (u otros pares similares), escrito en **Python**, con:
- Auto‑optimización diaria usando OpenAI.
- Estrategias combinadas (indicadores técnicos + ML).
- Persistencia completa del estado (SQLite u otra opción local).
- Reportes diarios automáticos.
- Un dashboard web para monitoreo.
- Un único punto de entrada: `python run.py`.

---

## 1. AGENT: Architect

**Rol:** Diseñar la arquitectura completa del bot.

**Responsabilidades:**
- Definir la estructura de carpetas y módulos del proyecto.
- Definir contratos entre capas (interfaces, DTOs básicos).
- Definir la capa de persistencia (SQLite simple, con tablas para: estado, métricas, trades simulados, parámetros).
- Definir los flujos principales:
  - Ciclo continuo de trading simulado.
  - Ciclo de evaluación diaria.
  - Flujo de optimización automática con OpenAI.
  - Flujo de recuperación tras apagado.
- Documentar la arquitectura en archivos bajo `/diagrams` y `/plans`.

**Entregables:**
- Estructura de carpetas bajo `/src`.
- `diagrams/architecture.mmd` y `diagrams/data_flow.mmd`.
- Descripción de módulos en `Project_Plan.md` y planes por fase.

---

## 2. AGENT: Backend Developer (Python)

**Rol:** Implementar la lógica interna del bot siguiendo la arquitectura definida.

**Responsabilidades:**
- Crear módulos en `/src/core`, `/src/exchange`, `/src/strategy`, `/src/evaluation`, etc.
- Implementar:
  - Mecanismo de scheduling interno (loop principal + tareas diarias).
  - Carga/lectura de configuración desde `config.yaml`.
  - Registro de logs estructurados (por ejemplo, en JSON o texto).
  - Módulo de paper trading (simulación de órdenes, PnL, fees).
  - Persistencia de estado (invocando a `state_manager`).


**Límites:**
- No desarrollar lógica específica de trading hasta que lo indique el Project_Plan.
- Los puntos de extensión se documentan claramente en docstrings y comentarios.

---

## 3. AGENT: Data Scientist / ML

**Rol:** Diseñar y construir los componentes de Machine Learning y estadística.

**Responsabilidades:**
- Definir y generar features a partir de datos OHLCV y volumen.
- Implementar en `/src/ml`:
  - `feature_engineering.py`
  - `trainer.py`
  - `model_manager.py`
  - `signal_generator.py`
- Explorar modelos como:
  - Modelos de clasificación (ej. RandomForest, XGBoost, regresión logística).
  - Modelos de series de tiempo simples (ej. regresión, pequeñas redes recurrentes si procede).
- Producir salidas del tipo:
  - Probabilidad de que la siguiente ventana de tiempo sea alcista/bajista.
  - Señales “buy / sell / hold” con un score de confianza.
- Guardar y cargar modelos desde `/src/data/models`.

**Límites:**
- No se requiere hiperoptimización ni arquitecturas profundas; el foco es un pipeline claro y extensible.
- Los modelos deben poder re‑entrenarse de manera incremental o diaria, usando datos locales.

---

## 4. AGENT: Evaluator / Self‑Optimizer

**Rol:** Evaluar el desempeño del bot y proponer/activar ajustes automáticos.

**Responsabilidades:**
- Analizar al final del día:
  - PnL simulado.
  - Drawdown.
  - Win rate.
  - Métricas de riesgo (Sharpe simple, etc.).
- Calcular una métrica de **probabilidad de profit positivo consistente** a partir de:
  - Métricas estadísticas.
  - Señales ML.
  - Estabilidad del sistema.
- Empaquetar un log diario y, a través del módulo `/src/evaluation/openai_optimizer.py`, enviar un payload a la API de OpenAI (el usuario configurará la clave) con:
  - Resumen del día.
  - Estadísticas clave.
  - Parámetros actuales.
- Recibir de OpenAI recomendaciones de ajuste (por ejemplo: modificar un umbral de RSI, cambiar ventana de entrenamiento, ajustar tamaño de posición) y aplicarlas de forma controlada en el estado persistente.

**Meta especial:**
- Cuando el Evaluator calcule que hay **≥ 30% de probabilidad** de obtener profit positivo de manera consistente (según reglas configurables), debe:
  - Registrar esa condición en la base de datos.
  - Exponer una bandera que el frontend pueda mostrar (ej: “Condición de probabilidad alcanzada, revisar para fondos reales”).

---

## 5. AGENT: Report Manager

**Rol:** Generar reportes diarios y almacenarlos de forma ordenada.

**Responsabilidades:**
- Construir en `/src/reports`:
  - `reporter.py`
  - Plantillas HTML/Markdown bajo `/src/reports/templates`.
- Generar cada día:
  - Un archivo Markdown con resumen del día.
  - Un archivo HTML (para el dashboard).
  - Un archivo JSON con datos de reporte estructurado (para otras integraciones).
- Guardar reportes en carpetas tipo `/reports/YYYY-MM-DD/`.

**Contenido mínimo del reporte:**
- Métricas básicas: PnL, número de trades, win rate, drawdown.
- Comentarios o notas generadas por el Evaluator/ML.
- Parámetros relevantes usados durante el día (ej. thresholds, tamaño de posición, etc.).
- Estado de la bandera de probabilidad ≥ 30%.

---

## 6. AGENT: Frontend/UI Developer

**Rol:** Crear una interfaz web ligera para visualizar el estado del bot.

**Responsabilidades:**
- Implementar un pequeño servidor web en `/src/frontend/server.py`, idealmente con:
  - FastAPI (o alternativa ligera).
  - Plantillas (Jinja2 / HTMX) bajo `/src/frontend/templates`.
  - Archivos estáticos bajo `/src/frontend/static`.
- El dashboard debe mostrar:
  - Estado de ejecución del bot.
  - PnL del día actual y de días recientes.
  - Últimas señales generadas.
  - Parámetros actuales relevantes (ej. tamaño de posición, thresholds).
  - Indicador del estado de “probabilidad ≥ 30%”.
- Incluir un mecanismo de actualización periódica (polling simple o HTMX).

**Límites:**
- El frontend **no debe** iniciar/parar el bot directamente; solo es una capa de observación.
- No se requiere autenticar usuarios en esta primera versión (aunque se debe dejar el hook preparado).

---

## 7. AGENT: Integrator / DevOps

**Rol:** Asegurar que todo funcione de forma integrada y sea fácil de ejecutar.

**Responsabilidades:**
- Crear y mantener:
  - `run.py` como punto de entrada principal.
  - `requirements.txt` con dependencias del proyecto.
- Asegurarse de que el proyecto pueda arrancar con:
  ```bash
  python run.py
  ```
- Configurar un mecanismo interno de scheduling:
  - Ciclo continuo (por ejemplo, cada X segundos para revisar condiciones de mercado).
  - Tarea diaria para evaluar y optimizar.
- (Opcional) Proveer un `Dockerfile` y/o instrucciones de despliegue en `README.md`.

---

## Flujos Principales del Sistema

### Flujo A – Ciclo continuo de trading simulado

1. Cargar configuración y estado persistente.
2. Descargar u obtener datos de mercado recientes (OHLCV).
3. Generar señales técnicas + ML.
4. Decidir operaciones simuladas (paper trading).
5. Actualizar estado (posiciones simuladas, PnL, métricas).
6. Registrar logs de cada decisión y trade simulado.

### Flujo B – Evaluación y optimización diaria

1. Al final del día (según zona horaria configurable), construir un resumen:
   - PnL del día.
   - Número de trades y win rate.
   - Evolución de PnL.
   - Métricas de riesgo (Sharpe, max drawdown simple).
2. Calcular una “probabilidad de profit positivo consistente” usando reglas y/o modelos ML.
3. Enviar a OpenAI un payload con:
   - Resumen del día.
   - Parámetros actuales.
   - Restricciones (por ejemplo: no aumentar riesgo más de X).
4. Recibir respuesta de OpenAI con recomendaciones y aplicar ajustes (dentro de límites seguros).
5. Guardar todo en la base de datos y en reportes.

### Flujo C – Recuperación tras apagado

1. Al iniciar, leer desde SQLite:
   - Parámetros vigentes.
   - Estado de posiciones simuladas.
   - Modelos ML entrenados o su metadata.
   - Última probabilidad estimada.
2. Reanudar ejecución continua sin perder coherencia.
3. Registrar cualquier inconsistencia y proceder de forma segura.

---

## Reglas de Oro

- El diseño debe ser modular y extensible: nuevas estrategias, nuevos pares, nuevos modelos ML.
- `run.py` debe ser el único punto obligatorio de entrada para el usuario final.
- Todo ajuste automático debe:
  - Estar trazado (quién lo sugirió, cuándo, por qué).
  - Persistirse antes de entrar en producción (en el siguiente ciclo).
- El sistema debe preferir la **robustez** sobre la complejidad: primero claridad, luego sofisticación.
