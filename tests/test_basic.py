"""Basic tests for the trading bot."""

import pytest
from src.strategy.scalping_aggressive import ScalpingAggressiveStrategy
from src.strategy.ema_rsi_volume import EmaRsiVolumeStrategy
from src.strategy.bollinger_range import BollingerRangeStrategy
from src.strategy.registry import STRATEGY_REGISTRY


def test_scalping_aggressive_import():
    """Test that the new strategy can be imported and instantiated."""
    strategy = ScalpingAggressiveStrategy()
    assert strategy.name() == "scalping_aggressive"
    assert "scalping" in strategy.description.lower()


def test_strategy_registry():
    """Test that all strategies are registered."""
    assert "scalping_aggressive" in STRATEGY_REGISTRY
    assert "ema_rsi_volume" in STRATEGY_REGISTRY
    assert "bollinger_range" in STRATEGY_REGISTRY
    assert "aggressive" in STRATEGY_REGISTRY
    assert "conservative" in STRATEGY_REGISTRY


def test_ema_rsi_volume_import():
    strategy = EmaRsiVolumeStrategy()
    assert strategy.name() == "ema_rsi_volume"


def test_scalping_signal_generation():
    """Test basic signal generation for scalping strategy."""
    strategy = ScalpingAggressiveStrategy()
    # Mock minimal indicators
    indicators = {
        "closes": [100.0] * 50,
        "rsi": [50.0] * 50,
        "atr": [1.0] * 50,
    }
    signal = strategy.generate_signal(indicators)
    assert "decision" in signal
    assert signal["decision"] in ["buy", "sell", "hold"]
    assert "confidence" in signal
    assert 0 <= signal["confidence"] <= 1


def test_ema_rsi_volume_signal_generation():
    strategy = EmaRsiVolumeStrategy()
    indicators = {
        "closes": [100.0 + (i * 0.01) for i in range(60)],
        "volumes": [1000.0] * 59 + [5000.0],
    }
    signal = strategy.generate_signal(indicators, context={"position_qty": 0.0, "ml": {}})
    assert "decision" in signal
    assert signal["decision"] in ["buy", "sell", "hold"]
    assert "confidence" in signal
    assert 0 <= signal["confidence"] <= 1


def test_bollinger_range_import():
    strategy = BollingerRangeStrategy()
    assert strategy.name() == "bollinger_range"


def test_bollinger_range_signal_generation():
    strategy = BollingerRangeStrategy()
    # Create a series where the last close dips below the lower band then bounces.
    closes = [100.0] * 30 + [98.0, 97.5, 97.0, 97.2]
    ohlcv = []
    for i, c in enumerate(closes):
        ohlcv.append({"open_time": f"t{i}", "open": c + 0.2, "high": c + 0.4, "low": c - 0.4, "close": c, "volume": 1000.0})
    indicators = {
        "ohlcv": ohlcv,
        "closes": closes,
        "highs": [row["high"] for row in ohlcv],
        "lows": [row["low"] for row in ohlcv],
        "volumes": [row["volume"] for row in ohlcv],
        "bollinger": {
            "lower": [99.0] * len(closes),
            "middle": [100.0] * len(closes),
            "upper": [101.0] * len(closes),
        },
        "rsi": [25.0] * len(closes),
        "atr": [0.5] * len(closes),
        "ema_fast": [100.0] * len(closes),
        "ema_slow": [100.0] * len(closes),
    }
    signal = strategy.generate_signal(indicators, context={"position_qty": 0.0})
    assert "decision" in signal
    assert signal["decision"] in ["buy", "sell", "hold"]
    assert "confidence" in signal
    assert 0 <= signal["confidence"] <= 1
