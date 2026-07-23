"""Authoritative Binance public best-bid/ask boundary."""

from decimal import Decimal

import httpx
import pytest

from workers.candidate.market import BinanceBookTickerClient


@pytest.mark.asyncio
async def test_book_ticker_returns_validated_market_snapshot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "BTCUSDT"
        return httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "bidPrice": "99.99",
                "bidQty": "12.5",
                "askPrice": "100.01",
                "askQty": "8.25",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await BinanceBookTickerClient(client=client).fetch("BTCUSDT")

    assert snapshot.bid == Decimal("99.99")
    assert snapshot.ask == Decimal("100.01")
    assert snapshot.available_bid_quantity == Decimal("12.5")
    assert snapshot.available_ask_quantity == Decimal("8.25")


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["BNBUSDT", "btc/usdt", ""])
async def test_noncanonical_symbol_never_reaches_exchange(symbol: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: pytest.fail("network called"))
    ) as client:
        with pytest.raises(ValueError, match="allowlist"):
            await BinanceBookTickerClient(client=client).fetch(symbol)


@pytest.mark.asyncio
async def test_malformed_or_cross_symbol_response_fails_closed() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "symbol": "ETHUSDT",
                    "bidPrice": "NaN",
                    "bidQty": "1",
                    "askPrice": "100",
                    "askQty": "1",
                },
            )
        )
    ) as client:
        with pytest.raises(RuntimeError, match="invalid"):
            await BinanceBookTickerClient(client=client).fetch("BTCUSDT")
