"""
Execution Service — Broker abstraction layer with Binance, FIX Simulator, and Paper trading.

Provides unified interface for:
- Binance (real trading via REST + WebSocket)
- FIX Simulator (institutional simulation)
- Paper Trading (backtesting/simulation)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
import websockets

logger = logging.getLogger(__name__)

LEGACY_PAPER_MUTATION_ENV = "QUANTEX_ENABLE_LEGACY_PAPER_MUTATIONS"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _legacy_paper_mutations_enabled() -> bool:
    """Return whether explicitly requested legacy paper-only mutation is enabled."""
    return os.getenv(LEGACY_PAPER_MUTATION_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


# ═══════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Good Till Crossing (post-only)


class ExecutionMode(str, Enum):
    LIVE = "live"      # Real Binance
    FIX_SIM = "fix_sim"  # FIX simulation
    PAPER = "paper"    # Paper trading


@dataclass
class Order:
    """Trading order."""
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str = field(default_factory=lambda: f"client_{uuid.uuid4().hex[:12]}")
    reduce_only: bool = False
    post_only: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "client_order_id": self.client_order_id,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
        }


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    price: float | None = None
    stop_price: float | None = None
    commission: float = 0.0
    commission_asset: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "status": self.status.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "avg_price": self.avg_price,
            "price": self.price,
            "stop_price": self.stop_price,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
        }


@dataclass
class Position:
    """Current position."""
    symbol: str
    side: OrderSide  # LONG = BUY, SHORT = SELL
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "leverage": self.leverage,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AccountInfo:
    """Account balance and info."""
    total_balance: float
    available_balance: float
    unrealized_pnl: float = 0.0
    positions: list[Position] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════
# ABSTRACT BROKER BASE
# ═══════════════════════════════════════════════════════════════════

class BaseBroker(ABC):
    """Abstract base class for all brokers."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self._connected = False
        self._order_callbacks: list[Callable[[OrderResult], None]] = []
        self._position_callbacks: list[Callable[[Position], None]] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place new order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel existing order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResult | None:
        """Get order status."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Get all open orders."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account info."""
        pass

    def on_order_update(self, callback: Callable[[OrderResult], None]):
        """Register order update callback."""
        self._order_callbacks.append(callback)

    def on_position_update(self, callback: Callable[[Position], None]):
        """Register position update callback."""
        self._position_callbacks.append(callback)

    def _emit_order_update(self, result: OrderResult):
        for cb in self._order_callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    def _emit_position_update(self, position: Position):
        for cb in self._position_callbacks:
            try:
                cb(position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")

    @property
    def connected(self) -> bool:
        return self._connected


# ═══════════════════════════════════════════════════════════════════
# BINANCE BROKER (REAL TRADING)
# ═══════════════════════════════════════════════════════════════════

class BinanceBroker(BaseBroker):
    """Binance Spot/Futures broker via REST + WebSocket."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.api_key = config.get("api_key", "") if config else ""
        self.api_secret = config.get("api_secret", "") if config else ""
        self.testnet = config.get("testnet", False) if config else False
        self.futures = config.get("futures", False) if config else False

        self.base_url = (
            "https://testnet.binancefuture.com" if self.testnet and self.futures
            else "https://testnet.binance.vision" if self.testnet and not self.futures
            else "https://fapi.binance.com" if self.futures
            else "https://api.binance.com"
        )
        self.ws_base = (
            "wss://stream.binancefuture.com" if self.futures
            else "wss://stream.binance.com:9443"
        )
        if self.testnet:
            self.ws_base = (
                "wss://stream.binancefuture.com" if self.futures
                else "wss://testnet.binance.vision"
            )

        self._session: aiohttp.ClientSession | None = None
        self._ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._listen_key: str | None = None
        self._orders: dict[str, OrderResult] = {}
        self._positions: dict[str, Position] = {}
        self._keep_alive_task: asyncio.Task | None = None
        self._user_stream_task: asyncio.Task | None = None

    def _sign_params(self, params: dict) -> str:
        """Sign parameters for authenticated requests."""
        query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={signature}"

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    async def connect(self) -> bool:
        """Connect to Binance."""
        try:
            self._session = aiohttp.ClientSession()
            # Test connectivity
            async with self._session.get(f"{self.base_url}/api/v3/ping") as resp:
                if resp.status != 200:
                    return False

            # Get listen key for user data stream
            await self._get_listen_key()
            self._connected = True
            logger.info(f"[BinanceBroker] Connected (testnet={self.testnet}, futures={self.futures})")

            # Start background tasks
            self._keep_alive_task = asyncio.create_task(self._keep_listen_key_alive())
            self._user_stream_task = asyncio.create_task(self._user_data_stream())
            return True
        except Exception as e:
            logger.error(f"[BinanceBroker] Connection failed: {e}")
            return False

    async def _get_listen_key(self):
        """Get listen key for user data stream."""
        endpoint = "/fapi/v1/listenKey" if self.futures else "/api/v3/userDataStream"
        async with self._session.post(
            f"{self.base_url}{endpoint}",
            headers=self._headers()
        ) as resp:
            data = await resp.json()
            self._listen_key = data.get("listenKey")
            logger.info(f"[BinanceBroker] Listen key obtained: {self._listen_key[:10]}...")

    async def _keep_listen_key_alive(self):
        """Keep listen key alive."""
        while self._connected:
            await asyncio.sleep(1800)  # 30 minutes
            if self._listen_key:
                endpoint = "/fapi/v1/listenKey" if self.futures else "/api/v3/userDataStream"
                try:
                    async with self._session.put(
                        f"{self.base_url}{endpoint}",
                        headers=self._headers(),
                        params={"listenKey": self._listen_key}
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("[BinanceBroker] Failed to keep listen key alive")
                except Exception as e:
                    logger.error(f"[BinanceBroker] Keep alive error: {e}")

    async def _user_data_stream(self):
        """WebSocket user data stream for order updates."""
        ws_url = f"{self.ws_base}/ws/{self._listen_key}"
        while self._connected:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info("[BinanceBroker] User data stream connected")
                    async for msg in ws:
                        if not self._connected:
                            break
                        data = json.loads(msg)
                        await self._handle_user_data(data)
            except Exception as e:
                logger.error(f"[BinanceBroker] User stream error: {e}")
                await asyncio.sleep(5)

    async def _handle_user_data(self, data: dict):
        """Handle user data stream events."""
        event_type = data.get("e", "")
        if event_type == "ORDER_TRADE_UPDATE":
            order_data = data.get("o", {})
            await self._process_order_update(order_data)
        elif event_type == "ACCOUNT_UPDATE":
            await self._process_account_update(data.get("a", {}))

    async def _process_order_update(self, order_data: dict):
        """Process order update from user stream."""
        result = OrderResult(
            order_id=str(order_data.get("i", "")),
            client_order_id=order_data.get("c", ""),
            symbol=order_data.get("s", ""),
            side=OrderSide(order_data.get("S", "BUY")),
            type=OrderType(order_data.get("o", "MARKET")),
            status=OrderStatus(order_data.get("X", "NEW")),
            quantity=float(order_data.get("q", 0)),
            filled_quantity=float(order_data.get("z", 0)),
            avg_price=float(order_data.get("ap", 0)),
            price=float(order_data.get("p", 0)) if order_data.get("p") else None,
            stop_price=float(order_data.get("sp", 0)) if order_data.get("sp") else None,
            commission=float(order_data.get("n", 0)),
            commission_asset=order_data.get("N", ""),
            timestamp=datetime.fromtimestamp(order_data.get("T", 0) / 1000),
        )
        self._orders[result.order_id] = result
        self._emit_order_update(result)

    async def _process_account_update(self, account_data: dict):
        """Process account/position update."""
        for pos_data in account_data.get("P", []):
            pos = Position(
                symbol=pos_data.get("s", ""),
                side=OrderSide.BUY if float(pos_data.get("pa", 0)) > 0 else OrderSide.SELL,
                size=abs(float(pos_data.get("pa", 0))),
                entry_price=float(pos_data.get("ep", 0)),
                mark_price=float(pos_data.get("mp", 0)),
                unrealized_pnl=float(pos_data.get("up", 0)),
                leverage=int(pos_data.get("le", 1)),
            )
            self._positions[pos.symbol] = pos
            self._emit_position_update(pos)

    async def disconnect(self) -> None:
        self._connected = False
        if self._keep_alive_task:
            self._keep_alive_task.cancel()
        if self._user_stream_task:
            self._user_stream_task.cancel()
        for ws in self._ws_connections.values():
            await ws.close()
        if self._session:
            await self._session.close()
        logger.info("[BinanceBroker] Disconnected")

    async def place_order(self, order: Order) -> OrderResult:
        """Place order on Binance."""
        endpoint = "/fapi/v1/order" if self.futures else "/api/v3/order"

        params = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "quantity": order.quantity,
            "timestamp": int(time.time() * 1000),
        }

        if order.type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            params["price"] = order.price
            params["timeInForce"] = order.time_in_force.value

        if order.type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT):
            params["stopPrice"] = order.stop_price

        if order.client_order_id:
            params["newClientOrderId"] = order.client_order_id

        if order.reduce_only:
            params["reduceOnly"] = "true"

        if order.post_only:
            params["priceProtect"] = "true"  # GTX equivalent

        signed_query = self._sign_params(params)

        async with self._session.post(
            f"{self.base_url}{endpoint}?{signed_query}",
            headers=self._headers()
        ) as resp:
            data = await resp.json()

            if resp.status != 200:
                return OrderResult(
                    order_id="",
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    type=order.type,
                    status=OrderStatus.REJECTED,
                    quantity=order.quantity,
                    error_message=data.get("msg", "Unknown error"),
                )

            result = OrderResult(
                order_id=str(data.get("orderId", "")),
                client_order_id=data.get("clientOrderId", ""),
                symbol=data.get("symbol", ""),
                side=OrderSide(data.get("side", "BUY")),
                type=OrderType(data.get("type", "MARKET")),
                status=OrderStatus(data.get("status", "NEW")),
                quantity=float(data.get("origQty", 0)),
                filled_quantity=float(data.get("executedQty", 0)),
                avg_price=float(data.get("avgPrice", 0)) if data.get("avgPrice") else 0.0,
                price=float(data.get("price", 0)) if data.get("price") else None,
                commission=0.0,  # Filled via user stream
            )
            self._orders[result.order_id] = result
            return result

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        endpoint = "/fapi/v1/order" if self.futures else "/api/v3/order"
        params = {"symbol": symbol, "orderId": order_id, "timestamp": int(time.time() * 1000)}
        signed = self._sign_params(params)

        async with self._session.delete(
            f"{self.base_url}{endpoint}?{signed}",
            headers=self._headers()
        ) as resp:
            return resp.status == 200

    async def get_order(self, order_id: str, symbol: str) -> OrderResult | None:
        endpoint = "/fapi/v1/order" if self.futures else "/api/v3/order"
        params = {"symbol": symbol, "orderId": order_id, "timestamp": int(time.time() * 1000)}
        signed = self._sign_params(params)

        async with self._session.get(
            f"{self.base_url}{endpoint}?{signed}",
            headers=self._headers()
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return None
            return OrderResult(
                order_id=str(data.get("orderId", "")),
                client_order_id=data.get("clientOrderId", ""),
                symbol=data.get("symbol", ""),
                side=OrderSide(data.get("side", "BUY")),
                type=OrderType(data.get("type", "MARKET")),
                status=OrderStatus(data.get("status", "NEW")),
                quantity=float(data.get("origQty", 0)),
                filled_quantity=float(data.get("executedQty", 0)),
                avg_price=float(data.get("avgPrice", 0)) if data.get("avgPrice") else 0.0,
                price=float(data.get("price", 0)) if data.get("price") else None,
            )

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        endpoint = "/fapi/v1/openOrders" if self.futures else "/api/v3/openOrders"
        params = {"timestamp": int(time.time() * 1000)}
        if symbol:
            params["symbol"] = symbol
        signed = self._sign_params(params)

        async with self._session.get(
            f"{self.base_url}{endpoint}?{signed}",
            headers=self._headers()
        ) as resp:
            data = await resp.json()
            return [
                OrderResult(
                    order_id=str(o.get("orderId", "")),
                    client_order_id=o.get("clientOrderId", ""),
                    symbol=o.get("symbol", ""),
                    side=OrderSide(o.get("side", "BUY")),
                    type=OrderType(o.get("type", "MARKET")),
                    status=OrderStatus(o.get("status", "NEW")),
                    quantity=float(o.get("origQty", 0)),
                    filled_quantity=float(o.get("executedQty", 0)),
                    avg_price=float(o.get("avgPrice", 0)) if o.get("avgPrice") else 0.0,
                    price=float(o.get("price", 0)) if o.get("price") else None,
                )
                for o in data
            ]

    async def get_positions(self) -> list[Position]:
        if not self.futures:
            return []

        endpoint = "/fapi/v2/positionRisk"
        params = {"timestamp": int(time.time() * 1000)}
        signed = self._sign_params(params)

        async with self._session.get(
            f"{self.base_url}{endpoint}?{signed}",
            headers=self._headers()
        ) as resp:
            data = await resp.json()
            positions = []
            for p in data:
                size = float(p.get("positionAmt", 0))
                if size != 0:
                    positions.append(Position(
                        symbol=p.get("symbol", ""),
                        side=OrderSide.BUY if size > 0 else OrderSide.SELL,
                        size=abs(size),
                        entry_price=float(p.get("entryPrice", 0)),
                        mark_price=float(p.get("markPrice", 0)),
                        unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                        leverage=int(p.get("leverage", 1)),
                    ))
            return positions

    async def get_account(self) -> AccountInfo:
        endpoint = "/fapi/v2/account" if self.futures else "/api/v3/account"
        params = {"timestamp": int(time.time() * 1000)}
        signed = self._sign_params(params)

        async with self._session.get(
            f"{self.base_url}{endpoint}?{signed}",
            headers=self._headers()
        ) as resp:
            data = await resp.json()

            balances = data.get("assets" if self.futures else "balances", [])
            total = sum(float(b.get("walletBalance" if self.futures else "free", 0)) for b in balances)
            available = sum(float(b.get("availableBalance" if self.futures else "free", 0)) for b in balances)

            positions = await self.get_positions()

            return AccountInfo(
                total_balance=total,
                available_balance=available,
                unrealized_pnl=sum(p.unrealized_pnl for p in positions),
                positions=positions,
            )


# ═══════════════════════════════════════════════════════════════════
# FIX SIMULATOR (INSTITUTIONAL)
# ═══════════════════════════════════════════════════════════════════

class FIXSimulator(BaseBroker):
    """
    FIX Protocol Simulator.
    Simulates institutional FIX 4.4 behavior with realistic latency.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.latency_ms = config.get("latency_ms", 2) if config else 2
        self.fill_probability = config.get("fill_probability", 1.0) if config else 1.0
        self.slippage_bps = config.get("slippage_bps", 1) if config else 1

        self._orders: dict[str, OrderResult] = {}
        self._positions: dict[str, Position] = {}
        self._balances: dict[str, float] = {"USDT": 100000.0}
        self._sequence = 1
        self._price_feed: dict[str, float] = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}

    async def connect(self) -> bool:
        self._connected = True
        logger.info(f"[FIXSimulator] Connected (latency={self.latency_ms}ms)")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("[FIXSimulator] Disconnected")

    async def _simulate_latency(self):
        await asyncio.sleep(self.latency_ms / 1000)

    def _generate_order_id(self) -> str:
        self._sequence += 1
        return f"FIX_{self._sequence:08d}"

    async def place_order(self, order: Order) -> OrderResult:
        await self._simulate_latency()

        order_id = self._generate_order_id()
        client_oid = order.client_order_id

        # Check balance for BUY
        if order.side == OrderSide.BUY:
            required = order.quantity * (order.price or self._price_feed.get(order.symbol, 50000))
            available = self._balances.get("USDT", 0)
            if available < required:
                return OrderResult(
                    order_id="",
                    client_order_id=client_oid,
                    symbol=order.symbol,
                    side=order.side,
                    type=order.type,
                    status=OrderStatus.REJECTED,
                    quantity=order.quantity,
                    error_message=f"Insufficient balance: {available} < {required}",
                )

        # Simulate fill
        fill_price = order.price
        if order.type == OrderType.MARKET:
            base_price = self._price_feed.get(order.symbol, 50000)
            slippage = base_price * self.slippage_bps / 10000
            fill_price = base_price + (slippage if order.side == OrderSide.BUY else -slippage)

        filled_qty = order.quantity = order.quantity
        if self.fill_probability < 1.0:
            if random.random() > self.fill_probability:
                filled_qty = 0
                status = OrderStatus.NEW
            else:
                status = OrderStatus.FILLED
        else:
            status = OrderStatus.FILLED

        commission = filled_qty * fill_price * 0.0004  # 4 bps
        self._balances["USDT"] = self._balances.get("USDT", 0) - commission

        if order.side == OrderSide.BUY:
            self._balances["USDT"] -= filled_qty * fill_price
        else:
            self._balances["USDT"] += filled_qty * fill_price

        result = OrderResult(
            order_id=order_id,
            client_order_id=client_oid,
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            status=status,
            quantity=order.quantity,
            filled_quantity=filled_qty,
            avg_price=fill_price if filled_qty > 0 else 0,
            price=order.price,
            commission=commission,
            commission_asset="USDT",
        )

        self._orders[order_id] = result

        # Update position
        if filled_qty > 0:
            self._update_position(order.symbol, order.side, filled_qty, fill_price)

        self._emit_order_update(result)
        return result

    def _update_position(self, symbol: str, side: OrderSide, qty: float, price: float):
        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                size=qty,
                entry_price=price,
                mark_price=price,
            )
        else:
            pos = self._positions[symbol]
            if pos.side == side:
                # Add to position
                total_cost = pos.size * pos.entry_price + qty * price
                pos.size += qty
                pos.entry_price = total_cost / pos.size
            else:
                # Reduce or flip
                if qty >= pos.size:
                    # Flip
                    pos.side = side
                    pos.size = qty - pos.size
                    pos.entry_price = price
                else:
                    # Reduce
                    pos.size -= qty

            pos.mark_price = price
            self._emit_position_update(pos)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self._simulate_latency()
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELED
            self._emit_order_update(self._orders[order_id])
            return True
        return False

    async def get_order(self, order_id: str, symbol: str) -> OrderResult | None:
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        orders = [o for o in self._orders.values()
                  if o.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_account(self) -> AccountInfo:
        positions = await self.get_positions()
        total = sum(self._balances.values())
        return AccountInfo(
            total_balance=total,
            available_balance=self._balances.get("USDT", 0),
            positions=positions,
        )

    def set_balance(self, asset: str, amount: float):
        """Set balance for testing."""
        self._balances[asset] = amount

    def update_price(self, symbol: str, price: float):
        """Update mock price feed."""
        self._price_feed[symbol] = price
        # Update mark prices
        if symbol in self._positions:
            self._positions[symbol].mark_price = price


# ═══════════════════════════════════════════════════════════════════
# PAPER BROKER (BACKTESTING/SIMULATION)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PaperConfig:
    initial_balance: float = 100000.0
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0004


class PaperBroker(BaseBroker):
    """Paper trading broker for backtesting and simulation."""

    def __init__(self, config: PaperConfig | None = None):
        super().__init__()
        self.config = config or PaperConfig()
        self._balances: dict[str, float] = {"USDT": self.config.initial_balance}
        self._orders: dict[str, OrderResult] = {}
        self._positions: dict[str, Position] = {}
        self._price_feed: dict[str, float] = {}
        self._order_history: list[OrderResult] = []
        self._trade_history: list[dict] = []
        self._equity_curve: list[tuple[datetime, float]] = []

    async def connect(self) -> bool:
        self._connected = True
        logger.info(f"[PaperBroker] Connected (balance={self.config.initial_balance} USDT)")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("[PaperBroker] Disconnected")

    def update_price(self, symbol: str, price: float):
        """Update price feed and mark prices."""
        self._price_feed[symbol] = price
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.mark_price = price
            if pos.side == OrderSide.BUY:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.size
            self._emit_position_update(pos)

    def update_prices(self, prices: dict[str, float]):
        """Batch update prices."""
        for sym, price in prices.items():
            self.update_price(sym, price)

    async def place_order(self, order: Order) -> OrderResult:
        price = self._price_feed.get(order.symbol)
        if price is None:
            return OrderResult(
                order_id="",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                status=OrderStatus.REJECTED,
                quantity=order.quantity,
                error_message=f"No price feed for {order.symbol}",
            )

        # Calculate fill price
        if order.type == OrderType.MARKET:
            slippage = price * self.config.slippage_bps / 10000
            fill_price = price + (slippage if order.side == OrderSide.BUY else -slippage)
            fee_rate = self.config.taker_fee_rate
        else:
            fill_price = order.price or price
            fee_rate = self.config.maker_fee_rate

        filled_qty = order.quantity
        status = OrderStatus.FILLED

        commission = filled_qty * fill_price * fee_rate
        self._balances["USDT"] -= commission

        if order.side == OrderSide.BUY:
            cost = filled_qty * fill_price + commission
            if self._balances["USDT"] < cost:
                return OrderResult(
                    order_id="",
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    type=order.type,
                    status=OrderStatus.REJECTED,
                    quantity=order.quantity,
                    error_message=f"Insufficient balance: {self._balances['USDT']} < {cost}",
                )
            self._balances["USDT"] -= cost
        else:
            self._balances["USDT"] += filled_qty * fill_price - commission

        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        result = OrderResult(
            order_id=order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            status=status,
            quantity=order.quantity,
            filled_quantity=filled_qty,
            avg_price=fill_price,
            price=order.price,
            commission=commission,
            commission_asset="USDT",
        )

        self._orders[order_id] = result
        self._order_history.append(result)
        self._update_position(order.symbol, order.side, filled_qty, fill_price)
        self._record_trade(order, result)

        self._emit_order_update(result)
        return result

    def _update_position(self, symbol: str, side: OrderSide, qty: float, price: float):
        if symbol not in self._positions:
            pos = Position(
                symbol=symbol,
                side=side,
                size=qty,
                entry_price=price,
                mark_price=price,
            )
            self._positions[symbol] = pos
        else:
            pos = self._positions[symbol]
            if pos.side == side:
                total_cost = pos.size * pos.entry_price + qty * price
                pos.size += qty
                pos.entry_price = total_cost / pos.size
            else:
                if qty >= pos.size:
                    pos.side = side
                    pos.size = qty - pos.size
                    pos.entry_price = price
                else:
                    pos.size -= qty
                    pos.realized_pnl += (pos.entry_price - price) * qty if pos.side == OrderSide.BUY else (price - pos.entry_price) * qty

            pos.mark_price = price

        self._emit_position_update(pos)
        self._update_equity()

    def _record_trade(self, order: Order, result: OrderResult):
        self._trade_history.append({
            "timestamp": datetime.now(),
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "quantity": result.filled_quantity,
            "price": result.avg_price,
            "commission": result.commission,
        })

    def _update_equity(self):
        total = self._balances["USDT"]
        for pos in self._positions.values():
            total += pos.unrealized_pnl
        self._equity_curve.append((datetime.now(), total))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self._orders and self._orders[order_id].status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
            self._orders[order_id].status = OrderStatus.CANCELED
            self._emit_order_update(self._orders[order_id])
            return True
        return False

    async def get_order(self, order_id: str, symbol: str) -> OrderResult | None:
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        orders = [o for o in self._orders.values()
                  if o.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_account(self) -> AccountInfo:
        positions = await self.get_positions()
        unrealized = sum(p.unrealized_pnl for p in positions)
        total = self._balances["USDT"] + unrealized
        return AccountInfo(
            total_balance=total,
            available_balance=self._balances["USDT"],
            unrealized_pnl=unrealized,
            positions=positions,
        )

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        return self._equity_curve

    def get_trade_history(self) -> list[dict]:
        return self._trade_history


# ═══════════════════════════════════════════════════════════════════
# EXECUTION ORCHESTRATOR (SMART ROUTER)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.PAPER
    binance_config: dict | None = None
    fix_config: dict | None = None
    paper_config: PaperConfig | None = None


class ExecutionOrchestrator:
    """
    Smart order router that manages multiple brokers.
    Routes orders based on execution mode.
    """

    def __init__(self, config: ExecutionConfig | dict | None = None):
        if isinstance(config, dict):
            config = ExecutionConfig(
                mode=ExecutionMode(config.get("mode", "paper")),
                binance_config=config.get("binance"),
                fix_config=config.get("fix"),
                paper_config=PaperConfig(**config.get("paper", {})) if config.get("paper") else PaperConfig(),
            )
        self.config = config or ExecutionConfig()
        if self.config.mode == ExecutionMode.LIVE:
            raise RuntimeError(
                "Legacy live execution is quarantined. Use the canonical execution "
                "worker with a persisted deterministic risk decision."
            )
        self.brokers: dict[ExecutionMode, BaseBroker] = {}
        self._init_brokers()

    def _init_brokers(self):
        """Initialize brokers based on config."""
        # Always have paper broker for fallback
        self.brokers[ExecutionMode.PAPER] = PaperBroker(self.config.paper_config or PaperConfig())

        if self.config.binance_config:
            logger.warning(
                "Ignoring Binance configuration: legacy live execution is quarantined"
            )

        if self.config.mode == ExecutionMode.FIX_SIM or self.config.fix_config:
            self.brokers[ExecutionMode.FIX_SIM] = FIXSimulator(self.config.fix_config)

    def get_broker(self, mode: ExecutionMode | None = None) -> BaseBroker:
        """Get broker for specific mode."""
        mode = mode or self.config.mode
        if mode == ExecutionMode.LIVE:
            raise PermissionError(
                "Legacy live execution is disabled; canonical risk authorization is required"
            )
        return self.brokers[mode]

    async def connect(self) -> bool:
        """Connect all brokers."""
        results = await asyncio.gather(*[b.connect() for b in self.brokers.values()])
        return all(results)

    async def disconnect(self):
        """Disconnect all brokers."""
        await asyncio.gather(*[b.disconnect() for b in self.brokers.values()])

    def set_mode(self, mode: ExecutionMode):
        """Change execution mode."""
        if mode == ExecutionMode.LIVE:
            raise PermissionError("Legacy live execution cannot be enabled")
        if mode in self.brokers:
            self.config.mode = mode
        else:
            raise ValueError(f"Mode {mode} not available")

    async def place_order(self, order: Order, mode: ExecutionMode | None = None) -> OrderResult:
        """Place order using specified or default mode."""
        mode = mode or self.config.mode
        if mode != ExecutionMode.PAPER or not _legacy_paper_mutations_enabled():
            raise PermissionError(
                "Legacy order mutation is disabled. The canonical execution worker "
                "requires a persisted deterministic risk decision. For isolated paper "
                f"tests only, explicitly set {LEGACY_PAPER_MUTATION_ENV}=true."
            )
        broker = self.get_broker(mode)
        return await broker.place_order(order)

    async def cancel_order(self, order_id: str, symbol: str, mode: ExecutionMode | None = None) -> bool:
        mode = mode or self.config.mode
        if mode != ExecutionMode.PAPER or not _legacy_paper_mutations_enabled():
            raise PermissionError(
                "Legacy order mutation is disabled; canonical execution authorization is required"
            )
        broker = self.get_broker(mode)
        return await broker.cancel_order(order_id, symbol)

    async def get_order(self, order_id: str, symbol: str, mode: ExecutionMode | None = None) -> OrderResult | None:
        mode = mode or self.config.mode
        broker = self.get_broker(mode)
        return await broker.get_order(order_id, symbol)

    async def get_open_orders(self, symbol: str | None = None, mode: ExecutionMode | None = None) -> list[OrderResult]:
        mode = mode or self.config.mode
        broker = self.get_broker(mode)
        return await broker.get_open_orders(symbol)

    async def get_positions(self, mode: ExecutionMode | None = None) -> list[Position]:
        mode = mode or self.config.mode
        broker = self.get_broker(mode)
        return await broker.get_positions()

    async def get_account(self, mode: ExecutionMode | None = None) -> AccountInfo:
        mode = mode or self.config.mode
        broker = self.get_broker(mode)
        return await broker.get_account()

    def get_paper_broker(self) -> PaperBroker:
        """Get paper broker for direct access (price updates, etc.)."""
        return self.brokers[ExecutionMode.PAPER]

    def get_fix_broker(self) -> FIXSimulator:
        """Get FIX simulator for direct access."""
        return self.brokers.get(ExecutionMode.FIX_SIM)


# ═══════════════════════════════════════════════════════════════════
# FACTORIES
# ════════════════════════════════════════════════════════════════════

def create_execution_orchestrator(config: ExecutionConfig | dict | None = None) -> ExecutionOrchestrator:
    """Factory for ExecutionOrchestrator."""
    return ExecutionOrchestrator(config)


if __name__ == "__main__":
    import random

    async def test():
        # Test paper broker
        orchestrator = create_execution_orchestrator({
            "mode": "paper",
            "paper": {"initial_balance": 100000, "fee_rate": 0.0004}
        })

        # Update price feed
        paper_broker = orchestrator.get_broker(ExecutionMode.PAPER)
        paper_broker.update_price("BTCUSDT", 50000)

        await orchestrator.connect()

        # Place buy order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1,
        )
        result = await orchestrator.place_order(order)
        print(f"Order result: {result.to_dict()}")

        # Check position
        positions = await orchestrator.get_positions()
        print(f"Positions: {[p.to_dict() for p in positions]}")

        # Check account
        account = await orchestrator.get_account()
        print(f"Account: {account.total_balance:.2f} USDT")

        await orchestrator.disconnect()

    asyncio.run(test())
