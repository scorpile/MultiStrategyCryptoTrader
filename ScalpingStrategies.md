# Profitable Crypto Scalping Strategies (Research Notes)

Scalping is a very short-term trading style that aims to capture many small gains from small price movements. In crypto, it is typically executed on **1–5 minute charts** (sometimes up to 15 minutes) due to the volatility and liquidity of assets like Bitcoin and Ethereum.

This document summarizes a few commonly cited scalping approaches, including entry/exit rules, risk management, timeframe guidance, required tools, and practical notes.

> Educational notes only. Not financial advice.

---

## Strategy 1: EMA Crossover + RSI Confirmation (Trend Scalping)

**Idea:** Trade short intraday trend shifts using a fast/slow EMA crossover, filtered by RSI momentum and (optionally) volume.

**Entry**
- **Long:** EMA fast (e.g., **5**) crosses above EMA slow (e.g., **20**).
- Confirm with RSI (e.g., **RSI(7)**) supporting the move (for example, rising from oversold).
- Optional filter: a **volume spike** around the crossover to avoid weak signals.
- **Short:** the opposite crossover (fast below slow) + bearish RSI confirmation (not applicable to spot long-only bots, but common in generic scalping literature).

**Exit**
- Take profit quickly (often **~0.5%–1%** move in your favor).
- Stop-loss just below the most recent swing low (long) / above swing high (short).
- Optional: trailing stop (e.g., **~1%**) with a larger take-profit target (e.g., **~3%**) to improve risk/reward.

**Risk management**
- Keep risk per trade small (commonly **~0.5%** of capital).
- Aim for a favorable **R:R** (often **1:2** or **1:3**).
- Fees/spread matter a lot in scalping; low-fee venues and tight spreads are critical.

**Timeframe**
- Typically **1m / 3m / 5m**.

**Tools**
- Charting with EMA/RSI/volume (TradingView or exchange charts).
- Fast execution venue; optional automation via APIs/bots to reduce human latency.

---

## Strategy 2: Range Scalping with Bollinger Bands

**Idea:** When the market is sideways, buy near support and sell near resistance. Bollinger Bands help time “extremes” inside the range.

**Entry**
1) Identify a clear **range** with repeated bounces (horizontal support/resistance).
2) Use Bollinger Bands to time entries at the extremes:
   - **Long:** price near **support** and near/touching the **lower band**.
   - Confirm with a reversal candle (hammer / bullish engulfing) or an oscillator oversold reading.
   - **Short:** price near **resistance** and near/touching the **upper band** (generic; spot long-only systems typically skip shorts).

**Exit**
- Target the **opposite side** of the range (or the opposite band).
- Optional: scale out around the midline (Bollinger middle band / EMA20) and target the far side with the remainder.
- Stop-loss just **outside** the range (so a real breakout exits you quickly).

**Risk management**
- Calculate risk/reward using the range geometry before entering.
- Avoid oversized leverage in ranges; breakouts can invalidate the setup quickly.
- Many traders automate range oscillations via grid-style limit orders, but manual supervision helps avoid trading into breakouts.

**Timeframe**
- Often **5m** (balance between frequency and noise), but ranges can be traded from **1m–15m**.
- Multi-timeframe approach is common: define the range on a higher timeframe, execute on a lower one.

**Tools**
- Bollinger Bands (typical: **20 periods, ±2 std dev**).
- Drawing tools for support/resistance.
- Optional: RSI/Stoch for overbought/oversold confirmation.

---

## Strategy 3: Breakout Scalping

**Idea:** Trade the initial impulse when price breaks out of a range/level with momentum and volume.

**Entry**
- Identify a **key level**: range edge, major support/resistance, or chart pattern boundary.
- Enter on a **clean breakout** (strong candle body + elevated volume).
- Many scalpers use **stop orders** pre-placed above/below the level to enter instantly.
- Optional confirmation: price holds beyond the level for 1–2 candles, ADX/volume confirmation, etc. (trade-off: confirmation can reduce false breakouts but increases late entries).

**Exit**
- Take profit quickly (**~0.5%–1%**) to capture the initial impulse.
- Or use a trailing stop to follow a short-lived trend if it extends for a few minutes.
- Stop-loss on the **other side** of the broken level (tight stops help survive false breakouts).

**Risk management**
- Expect multiple small losses from false breakouts before a strong one pays.
- Keep per-trade risk small (commonly **≤1%**, often **~0.5%** for this style).
- Avoid “averaging down/up” against the breakout; if it fails, exit quickly.
- Consider a time-stop rule: if it doesn’t move your way fast, close early.
- Avoid major news spikes unless your execution infrastructure is designed for it.

**Timeframe**
- Typically **1m–5m** for detection and execution.
- Breakouts often cluster around high-liquidity periods.

**Tools**
- Price action + level marking.
- Volume monitoring; optional ADX/OBV for “strength” confirmation.
- Fast data feed and order types (stop-market/stop-limit) matter.

---

## Sources referenced in the original notes

The original Spanish notes included references to the following sites (not endorsed):

- `margex.com`
- `blog.binolla.com`
- `b2binpay.com`
- `tradersbusinessschool.com`
- `es.investing.com`
- `medium.com`
- `wundertrading.com`
- `cryptohopper.com`
