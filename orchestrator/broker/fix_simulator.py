"""
QUANTEX FIX Simulator — Institutional FIX protocol simulation for paper trading.

Simulates the FIX (Financial Information eXchange) protocol for institutional
trading without requiring a real FIX gateway connection.

Features:
  - FIX message encoding/decoding (Tag=Value format)
  - Order lifecycle management (New → Ack → Fill → Done)
  - Simulated fill engine with configurable latency
  - Market data via simulated orderbook
  - Session-level sequence numbers and heartbeat

Usage:
    broker = FIXSimulator()
    await broker.connect()
    order = await broker.place_order("BTCUSDT", "buy", 0.01)
    await broker.disconnect()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .base import (
    BaseBroker,
    BrokerConfig,
    BrokerOrder,
    BrokerPosition,
    BrokerError,
    OrderSide,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger("quantex.broker.fix")

# ── FIX Tag Constants ────────────────────────────────────────

FIX_TAGS = {
    "BeginString": 8,
    "BodyLength": 9,
    "MsgType": 35,
    "SenderCompID": 49,
    "TargetCompID": 56,
    "MsgSeqNum": 34,
    "SendingTime": 52,
    "ClOrdID": 11,
    "Symbol": 55,
    "Side": 54,
    "OrderQty": 38,
    "OrdType": 40,
    "Price": 44,
    "StopPx": 99,
    "TimeInForce": 59,
    "ExecID": 17,
    "ExecType": 150,
    "OrdStatus": 39,
    "LeavesQty": 151,
    "CumQty": 14,
    "AvgPx": 6,
    "LastQty": 32,
    "LastPx": 31,
    "Commission": 12,
    "Text": 58,
}

FIX_MSG_TYPES = {
    "NEW_ORDER_SINGLE": "D",
    "EXECUTION_REPORT": "8",
    "ORDER_CANCEL_REQUEST": "F",
    "ORDER_CANCEL_REJECT": "9",
    "HEARTBEAT": "0",
    "TEST_REQUEST": "1",
}

FIX_SIDE = {"buy": "1", "sell": "2", "short": "5"}
FIX_ORD_TYPE = {"market": "1", "limit": "2", "stop_loss": "3", "stop_loss_limit": "4"}
FIX_ORD_STATUS = {
    OrderStatus.NEW: "0",
    OrderStatus.PARTIALLY_FILLED: "1",
    OrderStatus.FILLED: "2",
    OrderStatus.CANCELED: "4",
    OrderStatus.REJECTED: "8",
    OrderStatus.EXPIRED: "C",
    OrderStatus.PENDING: "A",
}

# Reverse mapping
FIX_ORD_STATUS_REV = {v: k for k, v in FIX_ORD_STATUS.items()}


@dataclass
class FIXSession:
    """FIX session state."""
    sender_comp_id: str = "QUANTEX"
    target_comp_id: str = "SIMULATOR"
    begin_string: str = "FIX.4.4"
    msg_seq_num: int = 1
    heartbt_int: int = 30
    is_logged_on: bool = False


@dataclass
class SimulatedFill:
    """Simulated fill event."""
    order_id: str
    fill_quantity: float
    fill_price: float
    fill_time: float
    is_complete: bool


class FIXSimulator(BaseBroker):
    """
    Institutional FIX protocol simulation.

    Simulates a FIX gateway with realistic message flow:
      NewOrderSingle → ExecutionReport(Ack) → ExecutionReport(Fill) → Done

    Parameters
    ----------
    config:
        BrokerConfig with optional latency simulation settings:
        - extra.fix_latency_min_ms: Minimum fill latency (default 50)
        - extra.fix_latency_max_ms: Maximum fill latency (default 500)
        - extra.fix_fill_probability: Probability of immediate fill (default 0.7)
        - extra.slippage_bps: Simulated slippage for market orders (default 0.5)
    """

    def __init__(self, config: Optional[BrokerConfig] = None):
        if config is None:
            config = BrokerConfig(
                base_url="fix://simulator",
                testnet=True,
                extra={
                    "fix_latency_min_ms": 50,
                    "fix_latency_max_ms": 500,
                    "fix_fill_probability": 0.7,
                    "slippage_bps": 0.5,
                },
            )

        super().__init__(config)

        self._session = FIXSession()
        self._pending_orders: dict[str, dict] = {}  # cl_ord_id -> order data
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[SimulatedFill] = []
        self._positions: dict[str, BrokerPosition] = {}

        # Simulated market data
        self._current_prices: dict[str, float] = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
        }

    # ── FIX Message Protocol ───────────────────────────────

    def _encode_fix(self, msg_type: str, fields: dict) -> str:
        """Encode a FIX message in Tag=Value| format.

        Args:
            msg_type: FIX message type (e.g., "D" for NewOrderSingle)
            fields: Tag=Value pairs to include

        Returns:
            FIX message string.
        """
        body = [f"8={self._session.begin_string}"]
        body.append(f"35={msg_type}")
        body.append(f"49={self._session.sender_comp_id}")
        body.append(f"56={self._session.target_comp_id}")
        body.append(f"34={self._session.msg_seq_num}")
        body.append(f"52={time.strftime('%Y%m%d-%H:%M:%S')}")

        for tag, value in fields.items():
            body.append(f"{tag}={value}")

        body_str = "\x01".join(body) + "\x01"
        # BodyLength after tag 8, 9
        body_length = len(body_str.encode("utf-8"))
        body.insert(1, f"9={body_length}")
        body_str = "\x01".join(body) + "\x01"

        # Simple checksum
        checksum = sum(ord(c) for c in body_str) % 256
        body_str += f"10={checksum:03d}\x01"

        self._session.msg_seq_num += 1
        return body_str

    def _decode_fix(self, message: str) -> dict:
        """Decode a FIX message string to tag-value dict.

        Args:
            message: Raw FIX message string

        Returns:
            Dict of tag number -> value
        """
        result = {}
        pairs = message.strip().split("\x01")
        for pair in pairs:
            if "=" in pair:
                tag, value = pair.split("=", 1)
                try:
                    result[int(tag)] = value
                except ValueError:
                    result[tag] = value
        return result

    # ── Connection Lifecycle ───────────────────────────────

    async def connect(self) -> bool:
        """Establish simulated FIX session."""
        self._session = FIXSession()
        self._connected = True
        logger.info("FIXSimulator connected (sender=%s, target=%s)",
                     self._session.sender_comp_id,
                     self._session.target_comp_id)

        # Send logon message
        logon = self._encode_fix("A", {"98": "0", "108": str(self._session.heartbt_int)})
        logger.debug("FIX Logon: %s", logon[:80])

        return True

    async def disconnect(self) -> bool:
        """Close FIX session."""
        if self._connected:
            # Send logout
            logout = self._encode_fix("5", {})
            logger.debug("FIX Logout: %s", logout[:80])

        self._connected = False
        logger.info("FIXSimulator disconnected")
        return True

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
        """Place an order via simulated FIX protocol.

        Simulates the full lifecycle:
          1. Send NewOrderSingle message
          2. Receive ExecutionReport (Ack)
          3. Simulated fill with latency
          4. Receive ExecutionReport (Fill)

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Amount in base currency
            order_type: Market, limit, etc.
            price: Limit price
            stop_price: Stop price
            client_order_id: Optional client ID

        Returns:
            BrokerOrder with final fill status.
        """
        if not self._connected:
            raise BrokerError("Not connected")

        cl_ord_id = client_order_id or self._generate_order_id()
        symbol = symbol.upper()

        # ── 1. Build NewOrderSingle ────────────────────────
        fields = {
            FIX_TAGS["ClOrdID"]: cl_ord_id,
            FIX_TAGS["Symbol"]: symbol,
            FIX_TAGS["Side"]: FIX_SIDE.get(side.value, "1"),
            FIX_TAGS["OrderQty"]: str(quantity),
            FIX_TAGS["OrdType"]: FIX_ORD_TYPE.get(order_type.value, "1"),
        }

        if price is not None:
            fields[FIX_TAGS["Price"]] = str(price)
        if stop_price is not None:
            fields[FIX_TAGS["StopPx"]] = str(stop_price)

        # Time in Force: Day for limit, IOC for market
        if order_type == OrderType.LIMIT:
            fields[FIX_TAGS["TimeInForce"]] = "0"  # Day

        # Send NewOrderSingle
        new_order_msg = self._encode_fix(FIX_MSG_TYPES["NEW_ORDER_SINGLE"], fields)
        logger.debug("FIX NewOrderSingle: %s", new_order_msg[:100])

        # ── 2. Simulate latency ────────────────────────────
        latency_ms = random.uniform(
            self.config.extra.get("fix_latency_min_ms", 50),
            self.config.extra.get("fix_latency_max_ms", 500),
        )
        await asyncio.sleep(latency_ms / 1000)

        # ── 3. Generate ExecutionReport (Ack + Fill) ──────
        current_price = self._current_prices.get(symbol, 50000.0)
        slippage_bps = self.config.extra.get("slippage_bps", 0.5)

        if order_type == OrderType.MARKET or random.random() < self.config.extra.get("fix_fill_probability", 0.7):
            # Immediate fill with slippage
            slippage = slippage_bps / 10000
            fill_price = current_price * (1 + slippage) if side == OrderSide.BUY else current_price * (1 - slippage)
            fill_price = round(fill_price, 2)

            # Create BrokerOrder (filled)
            broker_order = BrokerOrder(
                id=cl_ord_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price or fill_price,
                stop_price=stop_price,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                filled_cost=round(quantity * fill_price, 2),
                average_price=fill_price,
                created_at=int(time.time() * 1000),
                updated_at=int(time.time() * 1000),
                metadata={
                    "fill_price": fill_price,
                    "fix_msg_type": "EXECUTION_REPORT",
                    "exec_type": "FILL",
                    "latency_ms": latency_ms,
                },
            )

            # Record fill
            self._fills.append(SimulatedFill(
                order_id=cl_ord_id,
                fill_quantity=quantity,
                fill_price=fill_price,
                fill_time=time.time(),
                is_complete=True,
            ))

        else:
            # Partial fill or pending
            partial_qty = quantity * random.uniform(0.3, 0.8)
            fill_price = current_price

            broker_order = BrokerOrder(
                id=cl_ord_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price or fill_price,
                stop_price=stop_price,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_quantity=round(partial_qty, 8),
                filled_cost=round(partial_qty * fill_price, 2),
                average_price=fill_price,
                created_at=int(time.time() * 1000),
                updated_at=int(time.time() * 1000),
                metadata={"latency_ms": latency_ms, "partial": True, "remaining": round(quantity - partial_qty, 8)},
            )

        # Update position
        self._update_position(broker_order)
        self._orders[cl_ord_id] = broker_order

        logger.info("FIX order %s: %s %s %f @ %s (status=%s)",
                     cl_ord_id[:12], side.value.upper(), symbol, quantity,
                     broker_order.average_price or "MKT", broker_order.status.value)

        return broker_order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order via FIX CancelRequest."""
        if order_id in self._orders and self._orders[order_id].status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
            self._orders[order_id].status = OrderStatus.CANCELED

            # Send CancelReject / Ack
            fields = {
                FIX_TAGS["ClOrdID"]: order_id,
                FIX_TAGS["Symbol"]: symbol.upper(),
                FIX_TAGS["OrdStatus"]: "4",  # Canceled
                FIX_TAGS["Text"]: "CANCELED_BY_USER",
            }
            cancel_ack = self._encode_fix(FIX_MSG_TYPES["ORDER_CANCEL_REJECT"], fields)
            logger.debug("FIX Cancel Ack: %s", cancel_ack[:80])

            return True
        return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[BrokerOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]:
        """Get all open (non-final) orders."""
        result = []
        for order in self._orders.values():
            if order.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING):
                if symbol is None or order.symbol == symbol.upper():
                    result.append(order)
        return result

    # ── Account & Market Data ──────────────────────────────

    async def get_positions(self) -> list[BrokerPosition]:
        """Get all simulated positions."""
        return list(self._positions.values())

    async def get_balance(self) -> dict[str, float]:
        """Get simulated account balances."""
        balance = {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0}

        # Adjust for open positions
        for pos in self._positions.values():
            if pos.symbol == "BTCUSDT":
                balance["BTC"] -= pos.quantity
                balance["USDT"] += pos.quantity * pos.entry_price

        return balance

    async def get_ticker(self, symbol: str) -> dict:
        """Get simulated ticker with slight random variation."""
        symbol = symbol.upper()
        base = self._current_prices.get(symbol, 50000.0)

        # Add small random walk
        change = base * random.uniform(-0.001, 0.001)
        price = base + change
        self._current_prices[symbol] = round(price, 2)

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "volume": round(random.uniform(1000, 10000), 2),
            "change_24h": round(random.uniform(-5, 5), 2),
        }

    # ── Helpers ────────────────────────────────────────────

    def _update_position(self, order: BrokerOrder):
        """Update simulated position after a fill."""
        if order.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            return

        qty = order.filled_quantity
        cost = order.filled_cost
        side_mult = 1 if order.side == OrderSide.BUY else -1

        symbol = order.symbol
        current_price = self._current_prices.get(symbol, 50000.0)

        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos.quantity

            # Update position
            new_qty = old_qty + (qty * side_mult)
            if abs(new_qty) < 1e-8:
                del self._positions[symbol]
            else:
                pos.quantity = new_qty
                pos.current_price = current_price
                pos.unrealized_pnl = round((current_price - pos.entry_price) * new_qty, 2)
                pos.side = "long" if new_qty > 0 else "short"
        else:
            if abs(qty * side_mult) > 0:
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol,
                    quantity=qty * side_mult,
                    entry_price=cost / qty,
                    current_price=current_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    side="long" if side_mult > 0 else "short",
                )

    def get_fix_session_state(self) -> dict:
        """Get current FIX session state (for debugging)."""
        return {
            "sender_comp_id": self._session.sender_comp_id,
            "target_comp_id": self._session.target_comp_id,
            "begin_string": self._session.begin_string,
            "msg_seq_num": self._session.msg_seq_num,
            "is_logged_on": self._session.is_logged_on,
            "pending_orders": len(self._pending_orders),
            "total_orders": len(self._orders),
        }
