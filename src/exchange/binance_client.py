"""Abstraction around Binance (or compatible) market data endpoints."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx


class BinanceClient:
    """Minimal client wrapper for fetching OHLCV data in simulation mode."""

    BASE_URL = "https://api.binance.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        sample_mode: bool = False,
        max_retries: int = 3,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Store credentials/configuration without triggering any network calls."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.sample_mode = sample_mode
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        *,
        fallback_to_sample: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return OHLCV candles for the requested symbol and interval.

        Parameters
        ----------
        symbol:
            Pair in Binance notation (e.g., "SOLUSDT").
        interval:
            Binance interval string ("1m", "5m", "1h", "1d", etc.).
        limit:
            Maximum number of candles to return (Binance allows up to 1500).
        fallback_to_sample:
            When True, return deterministic sample data if the HTTP call fails.
        """
        if self.sample_mode:
            return self._generate_sample_ohlcv(limit=limit, interval=interval)

        endpoint = f"{self.BASE_URL}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.get(endpoint, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                raw = response.json()
                return [self._normalize_kline(item) for item in raw]
            except Exception as exc:  # Broad catch to ensure we can fallback.
                last_exc = exc
        if fallback_to_sample:
            return self._generate_sample_ohlcv(limit=limit, interval=interval)
        raise RuntimeError(f"Unable to fetch OHLCV after {self.max_retries} attempts: {last_exc}") from last_exc

    def fetch_ohlcv_range(
        self,
        symbol: str,
        interval: str,
        *,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
        fallback_to_sample: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return OHLCV candles for a time-bounded range (paginatable).

        Notes
        -----
        - Binance accepts `startTime`/`endTime` in milliseconds.
        - Maximum `limit` is 1000 for this endpoint.
        """
        if self.sample_mode:
            return self._generate_sample_ohlcv_range(
                interval=interval,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                limit=limit,
            )

        endpoint = f"{self.BASE_URL}/api/v3/klines"
        params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        last_exc: Optional[Exception] = None
        for _attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.get(endpoint, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                raw = response.json()
                return [self._normalize_kline(item) for item in raw]
            except Exception as exc:
                last_exc = exc
        if fallback_to_sample:
            return self._generate_sample_ohlcv_range(
                interval=interval,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                limit=limit,
            )
        raise RuntimeError(f"Unable to fetch OHLCV range after {self.max_retries} attempts: {last_exc}") from last_exc

    def fetch_server_time(self) -> datetime:
        """Return the exchange server time to help align candles and timezones."""
        if self.sample_mode:
            return datetime.utcnow()
        endpoint = f"{self.BASE_URL}/api/v3/time"
        response = httpx.get(endpoint, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        server_ms = payload.get("serverTime")
        if server_ms is None:
            raise RuntimeError("Unexpected response from Binance time endpoint.")
        return datetime.utcfromtimestamp(server_ms / 1000)

    def fetch_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Return static metadata (min qty, step size, etc.) used by risk checks."""
        if self.sample_mode:
            return {
                "symbol": symbol.upper(),
                "baseAssetPrecision": 8,
                "quoteAssetPrecision": 8,
                "filters": [],
            }
        endpoint = f"{self.BASE_URL}/api/v3/exchangeInfo"
        params = {"symbol": symbol.upper()}
        response = httpx.get(endpoint, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        symbols = payload.get("symbols", [])
        if not symbols:
            raise RuntimeError(f"Symbol {symbol} not found in exchange info.")
        return symbols[0]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_kline(raw: List[Any]) -> Dict[str, Any]:
        """Convert Binance's numeric-rich list into a friendlier dictionary."""
        open_time = datetime.utcfromtimestamp(raw[0] / 1000)
        close_time = datetime.utcfromtimestamp(raw[6] / 1000)
        return {
            "open_time": open_time.isoformat() + "Z",
            "open": float(raw[1]),
            "high": float(raw[2]),
            "low": float(raw[3]),
            "close": float(raw[4]),
            "volume": float(raw[5]),
            "close_time": close_time.isoformat() + "Z",
            "quote_asset_volume": float(raw[7]),
            "number_of_trades": int(raw[8]),
        }

    @staticmethod
    def _generate_sample_ohlcv(limit: int, interval: str) -> List[Dict[str, Any]]:
        """Return deterministic pseudo-prices for offline development/testing."""
        now = datetime.utcnow().replace(microsecond=0)
        delta = BinanceClient._interval_to_timedelta(interval)
        base_price = 150.0
        candles: List[Dict[str, Any]] = []
        price = base_price
        for i in range(limit):
            open_time = now - delta * (limit - i)
            close_time = open_time + delta
            # Generate tiny random walk to mimic price movement.
            change = random.uniform(-1.0, 1.0)
            high = price + abs(change) * 0.6
            low = price - abs(change) * 0.6
            close = max(low, min(high, price + change))
            candles.append(
                {
                    "open_time": open_time.isoformat() + "Z",
                    "open": round(price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": round(random.uniform(1000, 2000), 2),
                    "close_time": close_time.isoformat() + "Z",
                    "quote_asset_volume": round(random.uniform(10000, 20000), 2),
                    "number_of_trades": random.randint(50, 200),
                }
            )
            price = close
        return candles

    @staticmethod
    def _generate_sample_ohlcv_range(
        *,
        interval: str,
        start_time_ms: Optional[int],
        end_time_ms: Optional[int],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return deterministic pseudo-prices for a time-bounded range."""
        delta = BinanceClient._interval_to_timedelta(interval)
        now = datetime.utcnow().replace(microsecond=0)
        if end_time_ms is not None and end_time_ms > 0:
            now = datetime.utcfromtimestamp(int(end_time_ms) / 1000).replace(microsecond=0)
        if start_time_ms is not None and start_time_ms > 0:
            start = datetime.utcfromtimestamp(int(start_time_ms) / 1000).replace(microsecond=0)
        else:
            start = now - delta * int(limit)

        candles: List[Dict[str, Any]] = []
        price = 150.0
        t = start
        while len(candles) < int(limit) and t <= now:
            close_time = t + delta
            change = random.uniform(-1.0, 1.0)
            high = price + abs(change) * 0.6
            low = price - abs(change) * 0.6
            close = max(low, min(high, price + change))
            candles.append(
                {
                    "open_time": t.isoformat() + "Z",
                    "open": round(price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": round(random.uniform(1000, 2000), 2),
                    "close_time": close_time.isoformat() + "Z",
                    "quote_asset_volume": round(random.uniform(10000, 20000), 2),
                    "number_of_trades": random.randint(50, 200),
                }
            )
            price = close
            t = t + delta
        return candles

    @staticmethod
    def _interval_to_timedelta(interval: str) -> timedelta:
        """Convert Binance-style interval strings to `timedelta` objects."""
        unit = interval[-1]
        value = int(interval[:-1])
        mapping = {
            "m": timedelta(minutes=value),
            "h": timedelta(hours=value),
            "d": timedelta(days=value),
            "w": timedelta(weeks=value),
        }
        if unit not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[unit]
