"""
QUANTEX Broker Abstraction — Base interface for all exchange brokers.

Every broker (Binance, FIX, Paper) must implement BaseBroker.
This enables drop-in switching between real and simulated execution.

Architecture:
  ┌──────────────────────────┐
  │      Trading System      │
  └──────────┬───────────────┘
             │ place_order() / get_positions() / ...
             ▼
  ┌──────────────────────────┐
  │      BaseBroker ABC      │  ← Abstract interface
  └──────┬──────┬──────┬─────┘
         │      │      │
    Binance  FIX    Paper
    Broker  Sim    Broker
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING = "pending"


# ── Data Classes ─────────────────────────────────────────────

@dataclass
class BrokerConfig:
    """Configuration for a broker connection."""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.binance.com"
    ws_url: str = "wss://stream.binance.com:9443/ws"
    testnet: bool = True
    timeout: float = 30.0
    rate_limit_rps: float = 10.0
    max_retries: int = 3
    extra: dict = field(default_factory=dict)


@dataclass
class BrokerOrder:
    """Normalized order representation across all brokers."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    filled_cost: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BrokerPosition:
    """Normalized position representation."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # "long" or "short"
    leverage: int = 1
    liquidation_price: Optional[float] = None
    margin: float = 0.0
    updated_at: float = 0.0


# ── Abstract Base Class ──────────────────────────────────────

class BaseBroker(ABC):
    """
    Abstract base class for all exchange brokers.

    Subclasses must implement:
      - connect(): Establish connection to the exchange
      - disconnect(): Close connection
      - place_order(): Submit an order
      - cancel_order(): Cancel an existing order
      - get_order(): Get order status
      - get_open_orders(): List open orders
      - get_positions(): Current positions
      - get_balance(): Account balance
      - get_ticker(): Latest price ticker

    Optional (override for real-time data):
      - subscribe_trades(): Real-time trade stream
      - subscribe_orderbook(): Real-time orderbook stream
    """

    def __init__(self, config: BrokerConfig):
        self.config = config
        self._connected = False
        self._order_count = 0
        self._latency_history: list[float] = []

    # ── Lifecycle ──────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the exchange.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the exchange.

        Returns:
            True if disconnection successful.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected

    # ── Order Management ───────────────────────────────────

    @abstractmethod
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
        """Place an order on the exchange.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: Buy or sell
            quantity: Amount in base currency
            order_type: Market, limit, stop-loss, etc.
            price: Limit price (required for limit orders)
            stop_price: Trigger price (required for stop orders)
            client_order_id: Optional client-side ID

        Returns:
            BrokerOrder with the order details.

        Raises:
            BrokerError: If order placement fails
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            True if cancellation successful.
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[BrokerOrder]:
        """Get order status.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            BrokerOrder if found, None otherwise.
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]:
        """List all open orders.

        Args:
            symbol: Optional filter by trading pair

        Returns:
            List of open BrokerOrder instances.
        """
        pass

    # ── Account & Market Data ──────────────────────────────

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Get all current positions.

        Returns:
            List of BrokerPosition instances.
        """
        pass

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Get account balances.

        Returns:
            Dict mapping asset symbol to free balance.
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict:
        """Get latest price ticker for a symbol.

        Returns:
            Dict with keys: symbol, price, volume, change_24h, etc.
        """
        pass

    # ── Real-time Data (optional) ─────────────────────────

    async def subscribe_trades(self, symbol: str, callback):
        """Subscribe to real-time trade stream (optional)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support trade subscriptions")

    async def subscribe_orderbook(self, symbol: str, callback):
        """Subscribe to real-time orderbook updates (optional)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support orderbook subscriptions")

    # ── Utility ────────────────────────────────────────────

    def _record_latency(self, latency_ms: float):
        """Record request latency."""
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 1000:
            self._latency_history = self._latency_history[-500:]

    @property
    def avg_latency_ms(self) -> float:
        """Average request latency."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self._order_count += 1
        return f"{self.__class__.__name__.lower()}_{int(time.time() * 1000)}_{self._order_count}"

    async def health(self) -> dict:
        """Get broker health status."""
        return {
            "broker": self.__class__.__name__,
            "connected": self._connected,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "config": {
                "testnet": self.config.testnet,
                "base_url": self.config.base_url,
            },
        }


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class BrokerConnectionError(BrokerError):
    """Connection-related errors."""
    pass


class BrokerOrderError(BrokerError):
    """Order placement/management errors."""
    pass


class BrokerRateLimitError(BrokerError):
    """Rate limit exceeded."""
    pass
