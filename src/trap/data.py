"""Download real daily Bitcoin prices from Binance's public API (no key needed)."""

from typing import Any

import pandas as pd
import requests

# data-api.binance.vision serves public market data only; api.binance.com is the fallback.
BASE_URLS = ["https://data-api.binance.vision", "https://api.binance.com"]


def _get_klines(params: dict[str, Any]) -> list[list[Any]]:
    last_error: Exception | None = None
    for base in BASE_URLS:
        try:
            response = requests.get(f"{base}/api/v3/klines", params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Could not reach Binance: {last_error}") from last_error


def fetch_daily_closes(symbol: str = "BTCUSDT", days: int = 1500) -> pd.Series:
    """Closing prices for the last `days` completed days, oldest first."""
    rows: list[list[Any]] = []
    remaining = days + 1  # +1 because we drop today's unfinished candle
    end_time: int | None = None

    while remaining > 0:
        limit = min(1000, remaining)
        params: dict[str, Any] = {"symbol": symbol, "interval": "1d", "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get_klines(params)
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1  # next request ends just before this batch
        remaining -= len(batch)
        if len(batch) < limit:
            break  # no older history exists

    rows = rows[:-1]  # drop the incomplete current day
    index = pd.to_datetime([row[0] for row in rows], unit="ms")
    return pd.Series([float(row[4]) for row in rows], index=index, name="close")
