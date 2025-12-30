"""External Crypto Fear & Greed Index provider (Alternative.me).

This uses an external public endpoint (no auth). It's best-effort: failures should
never crash the trading loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx


@dataclass(frozen=True)
class FearGreedSnapshot:
    value: int
    classification: str
    timestamp_utc: str
    source: str = "alternative.me"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": int(self.value),
            "classification": self.classification,
            "timestamp_utc": self.timestamp_utc,
            "source": self.source,
        }


class FearGreedProvider:
    """Fetches + caches the latest Crypto Fear & Greed index."""

    DEFAULT_URL = "https://api.alternative.me/fng/?limit=1&format=json"
    CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    CMC_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical?limit=1"

    def __init__(
        self,
        *,
        provider: str = "alternative_me",
        url: str | None = None,
        coinmarketcap_api_key: str | None = None,
        timeout_seconds: float = 10.0,
        min_refresh_seconds: int = 900,
    ) -> None:
        self.provider = str(provider or "alternative_me").lower()
        self.coinmarketcap_api_key = coinmarketcap_api_key
        self.url = url or self._resolve_url(self.provider)
        self.timeout_seconds = float(timeout_seconds)
        self.min_refresh_seconds = int(min_refresh_seconds)
        self._cached: Optional[FearGreedSnapshot] = None
        self._cached_at: Optional[datetime] = None

    def get_latest(self, *, now: Optional[datetime] = None, force_refresh: bool = False) -> Optional[FearGreedSnapshot]:
        now = now or datetime.now(timezone.utc)
        if not force_refresh and self._cached is not None and self._cached_at is not None:
            if (now - self._cached_at) < timedelta(seconds=self.min_refresh_seconds):
                return self._cached
        snapshot = self._fetch()
        if snapshot is not None:
            self._cached = snapshot
            self._cached_at = now
        return snapshot

    def _resolve_url(self, provider: str) -> str:
        provider = str(provider or "").lower()
        if provider in {"cnn", "cnn_fng"}:
            return self.CNN_URL
        if provider in {"coinmarketcap", "cmc"}:
            return self.CMC_URL
        return self.DEFAULT_URL

    def _fetch(self) -> Optional[FearGreedSnapshot]:
        if self.provider in {"cnn", "cnn_fng"}:
            return self._fetch_cnn()
        if self.provider in {"coinmarketcap", "cmc"}:
            return self._fetch_coinmarketcap()
        try:
            resp = httpx.get(self.url, timeout=self.timeout_seconds)
            resp.raise_for_status()
            payload = resp.json()
            data = (payload or {}).get("data") or []
            if not data:
                return None
            item = data[0] or {}
            value = int(float(item.get("value", 0)))
            classification = str(item.get("value_classification") or item.get("classification") or "").strip()
            ts_unix = item.get("timestamp")
            if ts_unix is None:
                ts = datetime.now(timezone.utc)
            else:
                ts = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc)
            return FearGreedSnapshot(
                value=value,
                classification=classification or "unknown",
                timestamp_utc=ts.isoformat().replace("+00:00", "Z"),
            )
        except Exception:
            return None

    def _fetch_coinmarketcap(self) -> Optional[FearGreedSnapshot]:
        try:
            api_key = (self.coinmarketcap_api_key or "").strip()
            if not api_key:
                return None
            headers = {
                "Accept": "application/json",
                "X-CMC_PRO_API_KEY": api_key,
            }
            resp = httpx.get(self.url, headers=headers, timeout=self.timeout_seconds, follow_redirects=True)
            resp.raise_for_status()
            payload = resp.json() or {}

            status = payload.get("status") or {}
            try:
                if int(status.get("error_code", 0) or 0) != 0:
                    return None
            except Exception:
                pass

            data = payload.get("data") or []
            if not data:
                return None
            item = data[0] or {}
            value = int(float(item.get("value", 0) or 0))
            classification = str(item.get("value_classification") or item.get("classification") or "unknown").strip()
            ts = item.get("timestamp")
            if isinstance(ts, str) and ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
            else:
                dt = datetime.now(timezone.utc)

            return FearGreedSnapshot(
                value=value,
                classification=classification or "unknown",
                timestamp_utc=dt.isoformat().replace("+00:00", "Z"),
                source="coinmarketcap",
            )
        except Exception:
            return None

    def _fetch_cnn(self) -> Optional[FearGreedSnapshot]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
            }
            resp = httpx.get(self.url, headers=headers, timeout=self.timeout_seconds, follow_redirects=True)
            resp.raise_for_status()
            payload = resp.json() or {}
            fg = (payload.get("fear_and_greed") or {})
            score = fg.get("score")
            rating = str(fg.get("rating") or "unknown").strip()
            ts = fg.get("timestamp")
            if score is None:
                return None
            value = int(round(float(score)))
            # CNN timestamp is ISO with timezone, e.g. 2025-12-26T23:59:48+00:00
            if isinstance(ts, str) and ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    dt = datetime.now(timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
            return FearGreedSnapshot(
                value=value,
                classification=rating.replace("_", " ").title(),
                timestamp_utc=dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                source="cnn",
            )
        except Exception:
            return None
