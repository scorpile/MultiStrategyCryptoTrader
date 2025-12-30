"""Shared exchange/order error types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OrderRejected(Exception):
    """Raised when an order is rejected due to deterministic constraints.

    Attributes
    ----------
    reason:
        Short machine-friendly reason (e.g. "min_qty", "min_notional", "insufficient_cash").
    details:
        Optional structured context for logs/UI.
    """

    reason: str
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:  # pragma: no cover
        base = str(self.reason or "rejected")
        if self.details:
            return f"{base}: {self.details}"
        return base

