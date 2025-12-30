"""Registry of all available strategies for runtime switching."""

from __future__ import annotations

from typing import Dict, Type

from .aggressive import AggressiveStrategy
from .conservative import ConservativeStrategy
from .moderate import ModerateStrategy
from .multifactor import MultiFactorStrategy
from .bollinger_range import BollingerRangeStrategy
from .ema_rsi_volume import EmaRsiVolumeStrategy
from .scalping_aggressive import ScalpingAggressiveStrategy
from .strategy_base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "conservative": ConservativeStrategy,
    "moderate": ModerateStrategy,
    "aggressive": AggressiveStrategy,
    "multifactor": MultiFactorStrategy,
    "bollinger_range": BollingerRangeStrategy,
    "ema_rsi_volume": EmaRsiVolumeStrategy,
    "scalping_aggressive": ScalpingAggressiveStrategy,
}

__all__ = ["STRATEGY_REGISTRY"]
