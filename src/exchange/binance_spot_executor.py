"""Binance Spot REST executor (signed endpoints) for live trading.

This module is intentionally minimal and only supports MARKET orders for the
configured symbol(s). It must be gated by higher-level safety checks before use.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx


@dataclass(frozen=True)
class BinanceSpotConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://api.binance.com"
    recv_window: int = 5000
    timeout_seconds: float = 10.0


class BinanceSpotExecutor:
    """Signed Binance Spot endpoints wrapper (account + orders)."""

    def __init__(self, config: BinanceSpotConfig) -> None:
        if not config.api_key or not config.api_secret:
            raise ValueError("Binance API key/secret required for live executor.")
        self._cfg = config
        self._client = httpx.Client(
            base_url=str(config.base_url).rstrip("/"),
            timeout=float(config.timeout_seconds),
            headers={"X-MBX-APIKEY": config.api_key},
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            return

    def ping(self) -> Dict[str, Any]:
        """Public ping endpoint (unsigned)."""
        resp = self._client.get("/api/v3/ping")
        resp.raise_for_status()
        return resp.json()

    def get_account(self) -> Dict[str, Any]:
        return self._signed("GET", "/api/v3/account", params={})

    def create_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
        new_order_resp_type: str = "FULL",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a market order.

        For BUY, prefer using quote_order_qty (USDT) to keep notional stable.
        For SELL, use quantity (base asset).
        """
        sym = str(symbol or "").upper()
        side_u = str(side or "").upper()
        if side_u not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side_u,
            "type": "MARKET",
            "newOrderRespType": str(new_order_resp_type or "FULL"),
        }
        if client_order_id:
            params["newClientOrderId"] = str(client_order_id)

        if side_u == "BUY":
            if quote_order_qty is None:
                raise ValueError("quote_order_qty required for BUY market order.")
            params["quoteOrderQty"] = self._format_number(quote_order_qty)
        else:
            if quantity is None:
                raise ValueError("quantity required for SELL market order.")
            params["quantity"] = self._format_number(quantity)

        return self._signed("POST", "/api/v3/order", params=params)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_number(value: float) -> str:
        try:
            return f"{float(value):.12f}".rstrip("0").rstrip(".")
        except Exception:
            return str(value)

    def _signed(self, method: str, path: str, *, params: Dict[str, Any]) -> Dict[str, Any]:
        query = dict(params or {})
        query["timestamp"] = int(time.time() * 1000)
        query["recvWindow"] = int(self._cfg.recv_window)
        query_str = urlencode(query, doseq=True)
        signature = hmac.new(
            self._cfg.api_secret.encode("utf-8"),
            query_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        query["signature"] = signature
        method_u = method.upper()
        if method_u == "GET":
            resp = self._client.get(path, params=query)
        elif method_u == "POST":
            resp = self._client.post(path, params=query)
        elif method_u == "DELETE":
            resp = self._client.delete(path, params=query)
        else:
            raise ValueError(f"Unsupported method: {method}")
        resp.raise_for_status()
        return resp.json()

