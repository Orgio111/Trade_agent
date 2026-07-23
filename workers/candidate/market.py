"""Public Binance best-bid/ask snapshot client."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from packages.execution import MarketSnapshot


class BinanceBookTickerClient:
    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url="https://api.binance.com",
            timeout=httpx.Timeout(5),
            trust_env=False,
        )
        self._owns_client = client is None

    async def fetch(self, symbol: str) -> MarketSnapshot:
        if symbol not in {"BTCUSDT", "ETHUSDT"}:
            raise ValueError("symbol is outside canonical allowlist")
        try:
            response = await self._client.get(
                "https://api.binance.com/api/v3/ticker/bookTicker",
                params={"symbol": symbol},
            )
            response.raise_for_status()
            body = response.json()
            if body["symbol"] != symbol:
                raise ValueError("cross-symbol response")
            return MarketSnapshot(
                venue="binance",
                market_type="spot",
                instrument=symbol,
                bid=body["bidPrice"],
                ask=body["askPrice"],
                available_bid_quantity=body["bidQty"],
                available_ask_quantity=body["askQty"],
                observed_at=datetime.now(UTC),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Binance book ticker response is invalid") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()