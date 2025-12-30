"""Technical indicator helpers used by the strategy layer."""

from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def compute_rsi(closes: Iterable[float], period: int = 14) -> List[float]:
    """Return a Relative Strength Index series for the provided closing prices."""
    closes_list = list(closes)
    if len(closes_list) < period + 1:
        return [0.0] * len(closes_list)

    gains: List[float] = []
    losses: List[float] = []

    for i in range(1, len(closes_list)):
        delta = closes_list[i] - closes_list[i - 1]
        gains.append(max(delta, 0))
        losses.append(abs(min(delta, 0)))

    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    rsis: List[float] = [0.0] * (period)

    def calc_rsi(avg_g: float, avg_l: float) -> float:
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))

    rsis.append(calc_rsi(avg_gain, avg_loss))

    for i in range(period + 1, len(closes_list)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsis.append(calc_rsi(avg_gain, avg_loss))

    return rsis


def compute_macd(
    closes: Iterable[float],
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, List[float]]:
    """Return MACD line, signal line, and histogram."""
    closes_list = list(closes)
    if not closes_list:
        return {"macd": [], "signal": [], "histogram": []}

    ema_short = _ema(closes_list, short_period)
    ema_long = _ema(closes_list, long_period)
    macd_line = [short - long for short, long in zip(ema_short, ema_long)]
    signal_line = _ema(macd_line, signal_period)
    histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def compute_bollinger_bands(
    closes: Iterable[float],
    period: int = 20,
    num_std_dev: float = 2.0,
) -> Dict[str, List[float]]:
    """Return Bollinger Bands (upper/middle/lower) for visualization and signals."""
    closes_list = list(closes)
    middle_band: List[float] = []
    upper_band: List[float] = []
    lower_band: List[float] = []
    window = deque(maxlen=period)

    for price in closes_list:
        window.append(price)
        if len(window) < period:
            middle_band.append(0.0)
            upper_band.append(0.0)
            lower_band.append(0.0)
            continue
        avg = mean(window)
        variance = sum((p - avg) ** 2 for p in window) / period
        std_dev = variance ** 0.5
        middle_band.append(avg)
        upper_band.append(avg + num_std_dev * std_dev)
        lower_band.append(avg - num_std_dev * std_dev)

    return {
        "middle": middle_band,
        "upper": upper_band,
        "lower": lower_band,
    }


def compute_atr(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    period: int = 14,
) -> List[float]:
    """Return Average True Range values for volatility-aware sizing."""
    highs_list = list(highs)
    lows_list = list(lows)
    closes_list = list(closes)
    if not (highs_list and lows_list and closes_list):
        return []

    trs: List[float] = []
    for i in range(len(highs_list)):
        if i == 0:
            tr = highs_list[i] - lows_list[i]
        else:
            tr = max(
                highs_list[i] - lows_list[i],
                abs(highs_list[i] - closes_list[i - 1]),
                abs(lows_list[i] - closes_list[i - 1]),
            )
        trs.append(tr)

    atr_values = _ema(trs, period)
    return atr_values


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def _ema(values: List[float], period: int) -> List[float]:
    """Return an exponential moving average for a list of floats."""
    if not values:
        return []
    ema_values: List[float] = []
    multiplier = 2 / (period + 1)
    ema_values.append(values[0])
    for price in values[1:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def detect_doji(ohlc: List[Tuple[float, float, float, float]], threshold: float = 0.1) -> List[bool]:
    """Detect Doji candlestick pattern. Returns list of booleans."""
    patterns = []
    for o, h, l, c in ohlc:
        body = abs(c - o)
        total_range = h - l
        if total_range == 0:
            patterns.append(False)
        else:
            ratio = body / total_range
            patterns.append(ratio <= threshold)
    return patterns


def detect_engulfing(ohlc: List[Tuple[float, float, float, float]]) -> List[int]:
    """Detect Bullish/Bearish Engulfing. Returns 1 for bullish, -1 for bearish, 0 for none."""
    patterns = [0]  # First candle can't have engulfing
    for i in range(1, len(ohlc)):
        prev_o, prev_h, prev_l, prev_c = ohlc[i-1]
        curr_o, curr_h, curr_l, curr_c = ohlc[i]
        
        prev_body = abs(prev_c - prev_o)
        curr_body = abs(curr_c - curr_o)
        
        if curr_body > prev_body:
            if curr_c > curr_o and prev_c < prev_o and curr_o < prev_l and curr_c > prev_h:
                patterns.append(1)  # Bullish engulfing
            elif curr_c < curr_o and prev_c > prev_o and curr_o > prev_h and curr_c < prev_l:
                patterns.append(-1)  # Bearish engulfing
            else:
                patterns.append(0)
        else:
            patterns.append(0)
    return patterns


def compute_volume_sma(volume: Iterable[float], period: int = 20) -> List[float]:
    """Simple Moving Average of volume."""
    vol_list = list(volume)
    if len(vol_list) < period:
        return [mean(vol_list)] * len(vol_list)
    
    sma = []
    for i in range(len(vol_list)):
        if i < period - 1:
            sma.append(mean(vol_list[:i+1]))
        else:
            sma.append(mean(vol_list[i-period+1:i+1]))
    return sma


def compute_macd_histogram(
    closes: Iterable[float],
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> List[float]:
    """Compute MACD histogram (MACD - Signal)."""
    macd_result = compute_macd(closes, short_period, long_period, signal_period)
    macd_line = macd_result["macd"]
    signal_line = macd_result["signal"]
    return [m - s for m, s in zip(macd_line, signal_line)]


def compute_bollinger_width(
    closes: Iterable[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> List[float]:
    """Compute Bollinger Bands width ((upper - lower) / middle)."""
    bands = compute_bollinger_bands(closes, period, std_dev)
    upper = bands["upper"]
    middle = bands["middle"]
    lower = bands["lower"]
    width = []
    for u, m, l in zip(upper, middle, lower):
        if m != 0:
            width.append((u - l) / m)
        else:
            width.append(0.0)
    return width


def compute_vwap(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    volumes: Iterable[float],
) -> List[float]:
    """Compute Volume Weighted Average Price (VWAP)."""
    highs_list = list(highs)
    lows_list = list(lows)
    closes_list = list(closes)
    volumes_list = list(volumes)
    n = len(closes_list)
    if n == 0:
        return []
    
    vwap = []
    cumulative_volume = 0.0
    cumulative_vwap = 0.0
    
    for i in range(n):
        typical_price = (highs_list[i] + lows_list[i] + closes_list[i]) / 3
        volume = volumes_list[i]
        cumulative_volume += volume
        cumulative_vwap += typical_price * volume
        vwap.append(cumulative_vwap / cumulative_volume if cumulative_volume > 0 else 0.0)
    
    return vwap
