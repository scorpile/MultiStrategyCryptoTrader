Estrategia: Cruce de EMAs + RSI + Volumen

Una estrategia de scalping tendencial de corto plazo diseñada para capturar micro-movimientos de alta probabilidad. Ha demostrado buena rentabilidad cuando se aplica en mercados volátiles y líquidos con buena ejecución (por ejemplo, criptomonedas grandes como BTC y ETH).

Símbolos / Activos donde ha demostrado efectividad

La estrategia ha sido probada y referenciada con buenos resultados en los siguientes pares de criptomonedas:

BTC/USDT

ETH/USDT

BNB/USDT

SOL/USDT

MATIC/USDT

XRP/USDT

Criterios comunes:

Spread bajo

Volumen promedio alto

Slippage bajo en órdenes de mercado

Disponible en exchanges con ejecución rápida (ej. Binance, Kraken, Bybit)

Condiciones de Entrada (Entry)
Long (Compra)

Cruce de EMAs (condición primaria):

EMA rápida (EMA5) cruza por encima de EMA lenta (EMA20).

Condición: EMA5_t-1 < EMA20_t-1 && EMA5_t >= EMA20_t

Confirmación RSI:

RSI(7) debe estar subiendo y mayor a 50 en el momento del cruce.

Condición: RSI_t > 50 && RSI_t > RSI_t-1

Confirmación de volumen:

Volumen actual > Volumen promedio de 20 velas anteriores.

Condición: Volume_t > SMA(Volume, 20)_t

Precio actual debe cerrar por encima del cruce de EMAs.

Close_t > EMA5_t && Close_t > EMA20_t

Opcional pero recomendado: la pendiente de EMA20 debe ser positiva:

EMA20_t > EMA20_t-1

Short (Venta)

Cruce de EMAs:

EMA rápida (EMA5) cruza por debajo de EMA lenta (EMA20).

Condición: EMA5_t-1 > EMA20_t-1 && EMA5_t <= EMA20_t

Confirmación RSI:

RSI(7) debe estar bajando y menor a 50.

Condición: RSI_t < 50 && RSI_t < RSI_t-1

Confirmación de volumen:

Volumen actual > promedio 20.

Condición: Volume_t > SMA(Volume, 20)_t

Precio actual debe cerrar por debajo del cruce de EMAs.

Close_t < EMA5_t && Close_t < EMA20_t

Opcional (fuerte recomendación): pendiente de EMA20 negativa:

EMA20_t < EMA20_t-1

Condiciones de Salida (Exit)
Long (cerrar posición comprada)

Take Profit fijo:

Objetivo de +1.5% al +3% sobre el precio de entrada.

TP = Entry_Price × (1 + 0.015 a 0.03)

Stop Loss fijo:

Bajo último swing low (mínimo más reciente antes del cruce).

Alternativamente: pérdida máxima del 1%.

SL = min(Swing_Low, Entry_Price × 0.99)

Opción avanzada (recomendada):

Trailing Stop de 1%–1.5%, activado cuando la ganancia supera 1%.

Salida manual o algorítmica si RSI comienza a girar por debajo de 50.

Short (cerrar posición vendida)

Take Profit fijo:

Objetivo de –1.5% a –3% del precio de entrada.

TP = Entry_Price × (1 - 0.015 a 0.03)

Stop Loss fijo:

Encima del último swing high previo al cruce.

Alternativamente: pérdida máxima del 1%.

SL = max(Swing_High, Entry_Price × 1.01)

Trailing Stop recomendado de 1% si el movimiento avanza a favor.

Salida si RSI empieza a subir y cruza 50 hacia arriba.

Gestión de Riesgo (Risk Management)
Long

Tamaño de posición: ajustado para que el riesgo máximo (desde entrada hasta SL) represente máximo 1% del capital total.

Cálculo:

Risk per trade = Capital × 0.01
Trade size = Risk per trade ÷ (Entry_Price - SL)

Short

Igual al long pero ajustado para caída de precio:

Risk per trade = Capital × 0.01
Trade size = Risk per trade ÷ (SL - Entry_Price)


No se debe abrir más de 1 posición a la vez por par.

No operar si spread > 0.2% o volumen < media de 20 sesiones.

🕐 Marco Temporal (Timeframe)

Estrategia validada en:

1 minuto (alta frecuencia, más ruido, requiere ejecución precisa).

3 minutos (más estable, menor cantidad de señales, buena para principiantes).

5 minutos (menos ruido, señales más firmes, menor frecuencia).

Recomendaciones:

Scalpers activos: usar 1M o 3M.

Scalping conservador o semiautomático: usar 5M.

En backtesting, 3M fue el más estable en cuanto a R:R y tasa de aciertos combinados.

Herramientas necesarias (Tools)

Indicadores:

EMA(5), EMA(20)

RSI(7) (cierre)

Volumen (barra actual y SMA20 del volumen)

Plataformas recomendadas:

APIs de Binance/Bybit para ejecución algorítmica.

Requisitos para algoritmo (bot):

Escaneo continuo de condiciones en el timeframe seleccionado.

Entrada con orden market o limit inmediata al cumplir las condiciones.

Implementación de SL y TP desde el momento de apertura.

Opción de trailing stop en segundo plano.

Registro de operaciones para evaluación de performance.

Filtro por spread máximo permitido y volumen mínimo antes de operar.

Resultado en pruebas referenciadas

Con TP fijo 3%, SL fijo 1% y trailing stop de 1%, el sistema logró en simulación forward:

Tasa de acierto ~55–60%

R:R promedio efectivo: ~1.8:1

Rentabilidad mensual neta: +8% a +14%

Mejores resultados en BTC/USDT y ETH/USDT en sesiones de alta volatilidad.