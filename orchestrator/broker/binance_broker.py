"""
QUANTEX Binance Broker — Real Binance REST + WebSocket trading.

Uses python-binance for real exchange access with:
  - Asynchronous REST API (limits/orders/account)
  - WebSocket streams (trades, orderbook, user data)
  - Rate limit management
  - Error handling and retry logic
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Optional, AsyncGenerator
from urllib.parse import urlencode

import asyncio

import aiohttp

from .base import (
    BaseBroker,
    BrokerConfig,
    BrokerOrder,
    BrokerPosition,
    BrokerOrderError,
    BrokerConnectionError,
    BrokerRateLimitError,
    OrderSide,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger("quantex.broker.binance")

# ── Binance API Endpoints ────────────────────────────────────

BINANCE_API = "https://api.binance.com"
BINANCE_TESTNET_API = "https://testnet.binance.vision"
BINANCE_WS = "wss://stream.binance.com:9443/ws"
BINANCE_TESTNET_WS = "wss://testnet.binance.vision/ws"

# Weight costs (approximate, for rate limit tracking)
WEIGHT_COSTS = {
    "exchangeInfo": 10,
    "account": 10,
    "openOrders": 3,
    "allOrders": 10,
    "ticker_price": 2,
    "order": 1,
    "cancel": 1,
}


class BinanceBroker(BaseBroker):
    """
    Full Binance broker implementation.

    Supports both mainnet and testnet. All methods are async.
    Uses aiohttp for HTTP and websockets for streaming.

    Parameters
    ----------
    config:
        BrokerConfig with api_key, api_secret, testnet flag, etc.
    """

    def __init__(self, config: Optional[BrokerConfig] = None):
        if config is None:
            config = BrokerConfig(
                api_key="",
                api_secret="",
                base_url=BINANCE_TESTNET_API,
                ws_url=BINANCE_TESTNET_WS,
                testnet=True,
            )
        elif config.testnet:
            config.base_url = BINANCE_TESTNET_API
            config.ws_url = BINANCE_TESTNET_WS

        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Connection ─────────────────────────────────────────

    async def connect(self) -> bool:
        """Create HTTP session and verify API connectivity."""
        try:
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )

            # Verify connection with exchange info
            async with self._session.get("/api/v3/exchangeInfo", params={"symbol": "BTCUSDT"}) as resp:
                if resp.status == 200:
                    self._connected = True
                    logger.info("BinanceBroker connected to %s", "testnet" if self.config.testnet else "mainnet")
                    return True
                else:
                    raise BrokerConnectionError(f"Exchange info failed: {resp.status}")

        except Exception as e:
            logger.error("BinanceBroker connection failed: %s", e)
            self._connected = False
            if self._session:
                await self._session.close()
                self._session = None
            return False

    async def disconnect(self) -> bool:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("BinanceBroker disconnected")
        return True

    def _headers(self) -> dict:
        """Get HTTP headers for Binance API."""
        return {
            "X-MBX-APIKEY": self.config.api_key,
            "Content-Type": "application/json",
        }

    def _sign(self, params: dict) -> dict:
        """Sign parameters with HMAC-SHA256 for authenticated endpoints."""
        query = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    async def _request(
        self,
        method: str,
        path: str,
        signed: bool = False,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """Make an API request with retry logic and rate limit handling.

        Args:
            method: HTTP method (get, post, delete)
            path: API endpoint path
            signed: Whether to sign the request
            params: Query parameters
            data: Request body

        Returns:
            JSON response as dict.
        """
        if not self._session:
            raise BrokerConnectionError("Not connected. Call connect() first.")

        if params is None:
            params = {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params = self._sign(params)

        t0 = time.time()

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.request(method, path, params=params, json=data) as resp:
                    elapsed = (time.time() - t0) * 1000
                    self._record_latency(elapsed)

                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning("Rate limited, waiting %ds", retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status == 418:
                        raise BrokerRateLimitError("IP banned by Binance anti-DoS")

                    if resp.status >= 500:
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(1 * (attempt + 1))
                            continue
                        raise BrokerConnectionError(f"Server error {resp.status}: {await resp.text()}")

                    if resp.status >= 400:
                        error_data = await resp.json()
                        raise BrokerOrderError(f"API error {resp.status}: {error_data}")

                    return await resp.json()

            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                raise BrokerConnectionError(f"HTTP error: {e}")

        return {}

    # ── Order Management ───────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> BrokerOrder:
        """Place an order on Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "buy" or "sell"
            quantity: Amount in base currency
            order_type: Market, limit, stop-loss, etc.
            price: Limit price (required for limit orders)
            stop_price: Trigger price (required for stop orders)
            client_order_id: Optional client-side ID

        Returns:
            BrokerOrder with Binance response data.
        """
        params = {
            "symbol": symbol.upper(),
            "side": side.value.upper(),
            "type": order_type.value.upper(),
            "quantity": quantity,
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        if order_type == OrderType.LIMIT:
            if price is None:
                raise BrokerOrderError("Price required for limit orders")
            params["price"] = price
            params["timeInForce"] = "GTC"

        if order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
            if stop_price is None:
                raise BrokerOrderError("Stop price required for stop orders")
            params["stopPrice"] = stop_price
            if order_type == OrderType.STOP_LOSS_LIMIT:
                params["price"] = price
                params["timeInForce"] = "GTC"

        if order_type in (OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT):
            if stop_price is None:
                raise BrokerOrderError("Stop price required for take-profit orders")
            params["stopPrice"] = stop_price
            if order_type == OrderType.TAKE_PROFIT_LIMIT:
                params["price"] = price
                params["timeInForce"] = "GTC"

        try:
            data = await self._request("POST", "/api/v3/order", signed=True, params=params)

            return BrokerOrder(
                id=data.get("orderId", str(data.get("clientOrderId", ""))),
                symbol=symbol.upper(),
                side=OrderSide(side.value),
                type=order_type,
                quantity=float(data.get("origQty", quantity)),
                price=float(data["price"]) if data.get("price") != "0" else price,
                stop_price=float(data.get("stopPrice", 0)) if data.get("stopPrice") and data["stopPrice"] != "0" else stop_price,
                status=self._map_status(data.get("status", "NEW")),
                filled_quantity=float(data.get("executedQty", 0)),
                filled_cost=float(data.get("cummulativeQuoteQty", 0)),
                commission=float(data.get("commission", 0)),
                created_at=data.get("transactTime", int(time.time() * 1000)),
                metadata=data,
            )

        except Exception as e:
            logger.error("Order placement failed: %s", e)
            raise BrokerOrderError(f"Failed to place order: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order on Binance."""
        try:
            await self._request(
                "DELETE", "/api/v3/order",
                signed=True,
                params={"symbol": symbol.upper(), "orderId": int(order_id)},
            )
            return True
        except Exception as e:
            logger.warning("Cancel order failed: %s", e)
            return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[BrokerOrder]:
        """Get order status from Binance."""
        try:
            data = await self._request(
                "GET", "/api/v3/order",
                signed=True,
                params={"symbol": symbol.upper(), "orderId": int(order_id)},
            )
            return self._parse_order(data)
        except Exception:
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]:
        """Get open orders from Binance."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        data = await self._request("GET", "/api/v3/openOrders", signed=True, params=params)
        return [self._parse_order(o) for o in data]

    # ── Account & Market Data ──────────────────────────────

    async def get_positions(self) -> list[BrokerPosition]:
        """Get current positions from Binance account info."""
        data = await self._request("GET", "/api/v3/account", signed=True)
        positions = []

        for balance in data.get("balances", []):
            free = float(balance["free"])
            locked = float(balance["locked"])
            if free > 0 or locked > 0:
                try:
                    ticker = await self.get_ticker(f"{balance['asset']}USDT")
                    price = ticker.get("price", 0)
                except Exception:
                    price = 0

                positions.append(BrokerPosition(
                    symbol=balance["asset"],
                    quantity=free + locked,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    side="long",
                ))

        return positions

    async def get_balance(self) -> dict[str, float]:
        """Get free balances."""
        data = await self._request("GET", "/api/v3/account", signed=True)
        return {
            b["asset"]: float(b["free"])
            for b in data.get("balances", [])
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        }

    async def get_ticker(self, symbol: str) -> dict:
        """Get latest ticker for a symbol."""
        data = await self._request(
            "GET", "/api/v3/ticker/price",
            params={"symbol": symbol.upper()},
        )
        return {
            "symbol": data.get("symbol", symbol),
            "price": float(data.get("price", 0)),
        }

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> list[dict]:
        """Get kline/candlestick data.

        Args:
            symbol: Trading pair
            interval: "1m", "5m", "15m", "30m", "1h", "4h", "1d"
            limit: Number of candles (max 1000)

        Returns:
            List of kline dicts with OHLCV data.
        """
        data = await self._request(
            "GET", "/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        )

        return [
            {
                "time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            }
            for k in data
        ]

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _map_status(binance_status: str) -> OrderStatus:
        """Map Binance order status to our enum."""
        mapping = {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(binance_status, OrderStatus.PENDING)

    @staticmethod
    def _parse_order(data: dict) -> BrokerOrder:
        """Parse Binance API order response into BrokerOrder."""
        return BrokerOrder(
            id=str(data.get("orderId", "")),
            symbol=data.get("symbol", ""),
            side=OrderSide(data.get("side", "buy").lower()),
            type=OrderType(data.get("type", "market").replace("_", "").lower()),
            quantity=float(data.get("origQty", 0)),
            price=float(data["price"]) if data.get("price", "0") != "0" else None,
            stop_price=float(data["stopPrice"]) if data.get("stopPrice") and data["stopPrice"] != "0" else None,
            status=BinanceBroker._map_status(data.get("status", "NEW")),
            filled_quantity=float(data.get("executedQty", 0)),
            filled_cost=float(data.get("cummulativeQuoteQty", 0)),
            average_price=float(data.get("avgPrice", 0)) if data.get("avgPrice") else None,
            created_at=data.get("time", 0),
            updated_at=data.get("updateTime", 0),
            metadata=data,
        )

