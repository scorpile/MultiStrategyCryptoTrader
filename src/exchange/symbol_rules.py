"""Helpers for parsing Binance symbol trading rules (filters)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    min_qty: float = 0.0
    step_size: float = 0.0
    min_notional: float = 0.0


def parse_symbol_rules(symbol_info: Dict[str, Any]) -> Optional[SymbolRules]:
    """Extract `min_qty`, `step_size`, and `min_notional` from `exchangeInfo` payload."""
    if not isinstance(symbol_info, dict):
        return None
    symbol = str(symbol_info.get("symbol") or "").upper()
    if not symbol:
        return None
    filters = symbol_info.get("filters") or []
    if not isinstance(filters, list):
        filters = []

    min_qty = 0.0
    step_size = 0.0
    min_notional = 0.0

    def _f(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    for flt in filters:
        if not isinstance(flt, dict):
            continue
        ftype = str(flt.get("filterType") or "")
        if ftype in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
            min_qty = max(min_qty, _f(flt.get("minQty")))
            step_size = max(step_size, _f(flt.get("stepSize")))
        if ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
            min_notional = max(min_notional, _f(flt.get("minNotional")))

    return SymbolRules(symbol=symbol, min_qty=min_qty, step_size=step_size, min_notional=min_notional)

