"""
QUANTEX Paper Broker — Simulated paper trading for backtesting and dry-run.

Provides a complete simulated exchange environment for testing trading
strategies without real capital. Features:
  - Configurable initial balance
  - Slippage and fee simulation
  - Position tracking
  - PnL calculation
  - Order book simulation
  - Full trade history

Usage:
    broker = PaperBroker(initial_balance=10000.0)
    await broker.connect()
    order = await broker.place_order("BTCUSDT", "buy", 0.01)
    positions = await broker.get_positions()
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
    BrokerOrderError,
    OrderSide,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger("quantex.broker.paper")


@dataclass
class PaperOrderBook:
    """Simulated L2 order book."""
    bids: list[tuple[float, float]] = field(default_factory=list)  # (price, qty)
    asks: list[tuple[float, float]] = field(default_factory=list)
    spread_bps: float = 3.0
    depth_pct: float = 0.1  # Each level is this % away from mid


class PaperBroker(BaseBroker):
    """
    Paper trading broker — simulated exchange for backtesting.

    All trades are simulated with realistic fees and slippage.
    No real API calls are made. Supports all order types.

    Parameters
    ----------
    config:
        BrokerConfig with:
        - extra.initial_balance: Starting balance in USDT (default 10000)
        - extra.maker_fee: Maker fee rate (default 0.0002)
        - extra.taker_fee: Taker fee rate (default 0.0004)
        - extra.slippage_bps: Market order slippage (default 1.0)
        - extra.fill_probability: Limit order fill chance (0-1, default 0.5)
    """

    def __init__(self, config: Optional[BrokerConfig] = None):
        if config is None:
            config = BrokerConfig(
                base_url="paper://simulated",
                testnet=True,
                extra={
                    "initial_balance": 10000.0,
                    "maker_fee": 0.0002,
                    "taker_fee": 0.0004,
                    "slippage_bps": 1.0,
                    "fill_probability": 0.5,
                },
            )

        super().__init__(config)

        self._balance: dict[str, float] = {
            "USDT": config.extra.get("initial_balance", 10000.0),
        }
        self._initial_balance = config.extra.get("initial_balance", 10000.0)
        self._orders: dict[str, BrokerOrder] = {}
        self._positions: dict[str, BrokerPosition] = {}
        self._trade_history: list[dict] = []

        # Simulated market prices (updated on each ticker call)
        self._prices: dict[str, float] = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "SOLUSDT": 120.0,
        }

        # Order books for limit order simulation
        self._orderbooks: dict[str, PaperOrderBook] = {
            "BTCUSDT": PaperOrderBook(spread_bps=2.0),
            "ETHUSDT": PaperOrderBook(spread_bps=3.0),
            "SOLUSDT": PaperOrderBook(spread_bps=5.0),
        }

        # Maker/taker fee rates
        self._maker_fee = config.extra.get("maker_fee", 0.0002)
        self._taker_fee = config.extra.get("taker_fee", 0.0004)
        self._slippage_bps = config.extra.get("slippage_bps", 1.0)
        self._fill_probability = config.extra.get("fill_probability", 0.5)

        # Performance tracking
        self._peak_equity = config.extra.get("initial_balance", 10000.0)
        self._total_pnl = 0.0

    # ── Connection ─────────────────────────────────────────

    async def connect(self) -> bool:
        """Initialize paper broker (always succeeds)."""
        self._connected = True
        logger.info("PaperBroker connected (balance=%.2f USDT)", self._balance.get("USDT", 0))
        return True

    async def disconnect(self) -> bool:
        """Close paper broker."""
        self._connected = False
        logger.info("PaperBroker disconnected")
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
        """Place a simulated order.

        Market orders are filled immediately with slippage.
        Limit orders have a configurable fill probability.
        Stop orders trigger when the simulated price crosses the stop price.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Amount in base currency
            order_type: Market, limit, stop-loss, take-profit
            price: Limit price
            stop_price: Trigger price
            client_order_id: Optional client ID

        Returns:
            BrokerOrder with simulated fill details.
        """
        order_id = client_order_id or self._generate_order_id()
        symbol = symbol.upper()
        current_price = self._prices.get(symbol, 50000.0)

        # Set price from market if not specified
        if price is None:
            price = current_price

        # ── Market Orders (immediate fill) ────────────────
        if order_type == OrderType.MARKET:
            return self._fill_market_order(order_id, symbol, side, quantity, current_price)

        # ── Limit Orders (probabilistic fill) ─────────────
        elif order_type == OrderType.LIMIT:
            if price is None:
                raise BrokerOrderError("Price required for limit orders")

            # Check if limit order should fill based on probability
            will_fill = random.random() < self._fill_probability

            if will_fill:
                return self._fill_limit_order(order_id, symbol, side, quantity, price, current_price)
            else:
                # Place as pending
                broker_order = BrokerOrder(
                    id=order_id,
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.NEW,
                    created_at=int(time.time() * 1000),
                    metadata={"pending": True, "fill_later": True},
                )
                self._orders[order_id] = broker_order
                return broker_order

        # ── Stop Orders (trigger-based) ──────────────────
        elif order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT,
                            OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT):
            if stop_price is None:
                raise BrokerOrderError("Stop price required for stop orders")

            # Check if stop is triggered
            triggered = False
            if order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
                # Stop-loss triggers when price crosses below stop
                if side == OrderSide.SELL and current_price <= stop_price:
                    triggered = True
                elif side == OrderSide.BUY and current_price >= stop_price:
                    triggered = True
            else:
                # Take-profit triggers when price crosses above
                if side == OrderSide.SELL and current_price >= stop_price:
                    triggered = True
                elif side == OrderSide.BUY and current_price <= stop_price:
                    triggered = True

            if triggered:
                if order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
                    # Market after trigger
                    return self._fill_market_order(order_id, symbol, side, quantity, current_price)
                else:
                    # Limit after trigger
                    return self._fill_limit_order(order_id, symbol, side, quantity, price or current_price, current_price)
            else:
                # Not triggered — place as pending
                broker_order = BrokerOrder(
                    id=order_id,
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity,
                    price=price or current_price,
                    stop_price=stop_price,
                    status=OrderStatus.PENDING,
                    created_at=int(time.time() * 1000),
                    metadata={"stop_not_triggered": True, "current_price": current_price},
                )
                self._orders[order_id] = broker_order
                return broker_order

        return BrokerOrder(
            id=order_id, symbol=symbol, side=side, type=order_type,
            quantity=quantity, price=price,
            status=OrderStatus.REJECTED, created_at=int(time.time() * 1000),
            metadata={"error": f"Unsupported order type: {order_type}"},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a simulated order."""
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status in (OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED):
                order.status = OrderStatus.CANCELED
                order.updated_at = int(time.time() * 1000)
                return True
        return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[BrokerOrder]:
        """Get a simulated order by ID."""
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]:
        """Get all open simulated orders."""
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
        return dict(self._balance)

    async def get_ticker(self, symbol: str) -> dict:
        """Get simulated ticker with random walk."""
        symbol = symbol.upper()
        base = self._prices.get(symbol, 50000.0)

        # Add random walk
        change = base * random.uniform(-0.002, 0.002)
        price = round(base + change, 2)
        self._prices[symbol] = price

        return {
            "symbol": symbol,
            "bid": round(price * (1 - 0.00015), 2),
            "ask": round(price * (1 + 0.00015), 2),
            "price": price,
            "volume": round(random.uniform(500, 5000), 2),
            "change_24h": round(random.uniform(-3, 3), 2),
        }

    # ── Fill Logic ─────────────────────────────────────────

    def _fill_market_order(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
    ) -> BrokerOrder:
        """Fill a market order with slippage.

        Args:
            order_id: Generated order ID
            symbol: Trading pair
            side: Buy or sell
            quantity: Amount to fill
            current_price: Current market price

        Returns:
            Filled BrokerOrder.
        """
        # Apply slippage
        slippage = self._slippage_bps / 10000
        if side == OrderSide.BUY:
            fill_price = current_price * (1 + slippage)
        else:
            fill_price = current_price * (1 - slippage)

        fill_price = round(fill_price, 2)
        fee_rate = self._taker_fee
        fee = round(quantity * fill_price * fee_rate, 2)

        # Check if we have enough balance for buy
        if side == OrderSide.BUY:
            cost = quantity * fill_price + fee
            usdt_balance = self._balance.get("USDT", 0)
            if cost > usdt_balance:
                # Partial fill with available balance
                max_qty = (usdt_balance - fee) / fill_price
                quantity = max(0, max_qty)
                if quantity <= 0:
                    return BrokerOrder(
                        id=order_id, symbol=symbol, side=side, type=OrderType.MARKET,
                        quantity=0, price=fill_price, status=OrderStatus.REJECTED,
                        created_at=int(time.time() * 1000),
                        metadata={"error": "Insufficient balance"},
                    )

        # ── Update Balance ────────────────────────────────
        if side == OrderSide.BUY:
            cost = round(quantity * fill_price, 2)
            self._balance["USDT"] = round(self._balance.get("USDT", 0) - cost - fee, 2)
            self._balance[symbol.replace("USDT", "")] = round(
                self._balance.get(symbol.replace("USDT", ""), 0) + quantity, 8
            )
        else:
            revenue = round(quantity * fill_price, 2)
            asset = symbol.replace("USDT", "")
            self._balance[asset] = round(self._balance.get(asset, 0) - quantity, 8)
            self._balance["USDT"] = round(self._balance.get("USDT", 0) + revenue - fee, 2)

        # ── Update Position ───────────────────────────────
        self._update_position(symbol, side, quantity, fill_price)

        # ── Create Order ──────────────────────────────────
        broker_order = BrokerOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            price=fill_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_cost=round(quantity * fill_price, 2),
            average_price=fill_price,
            commission=fee,
            created_at=int(time.time() * 1000),
            updated_at=int(time.time() * 1000),
            metadata={"fill_price": fill_price, "slippage_bps": self._slippage_bps, "fee": fee},
        )

        self._orders[order_id] = broker_order

        # ── Record Trade ──────────────────────────────────
        self._trade_history.append({
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": fill_price,
            "fee": fee,
            "pnl": 0.0,
            "time": time.time(),
        })

        logger.info("Paper %s %s %f @ %.2f (fee=%.4f)", side.value.upper(), symbol, quantity, fill_price, fee)

        return broker_order

    def _fill_limit_order(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        current_price: float,
    ) -> BrokerOrder:
        """Fill a limit order at the specified price.

        Args:
            order_id: Generated order ID
            symbol: Trading pair
            side: Buy or sell
            quantity: Amount to fill
            limit_price: Limit order price
            current_price: Current market price

        Returns:
            Filled BrokerOrder.
        """
        fill_price = limit_price
        fee_rate = self._maker_fee  # Limit orders get maker fee
        fee = round(quantity * fill_price * fee_rate, 2)

        # Check balance
        if side == OrderSide.BUY:
            cost = quantity * fill_price + fee
            if cost > self._balance.get("USDT", 0):
                return BrokerOrder(
                    id=order_id, symbol=symbol, side=side, type=OrderType.LIMIT,
                    quantity=0, price=limit_price, status=OrderStatus.REJECTED,
                    created_at=int(time.time() * 1000),
                    metadata={"error": "Insufficient balance"},
                )

        # Update balances
        if side == OrderSide.BUY:
            cost = round(quantity * fill_price, 2)
            self._balance["USDT"] = round(self._balance.get("USDT", 0) - cost - fee, 2)
            self._balance[symbol.replace("USDT", "")] = round(
                self._balance.get(symbol.replace("USDT", ""), 0) + quantity, 8
            )
        else:
            revenue = round(quantity * fill_price, 2)
            asset = symbol.replace("USDT", "")
            self._balance[asset] = round(self._balance.get(asset, 0) - quantity, 8)
            self._balance["USDT"] = round(self._balance.get("USDT", 0) + revenue - fee, 2)

        self._update_position(symbol, side, quantity, fill_price)

        broker_order = BrokerOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            price=limit_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_cost=round(quantity * fill_price, 2),
            average_price=fill_price,
            commission=fee,
            created_at=int(time.time() * 1000),
            updated_at=int(time.time() * 1000),
            metadata={"fill_price": fill_price, "fee": fee, "maker": True},
        )

        self._orders[order_id] = broker_order
        self._trade_history.append({
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": fill_price,
            "fee": fee,
            "pnl": 0.0,
            "time": time.time(),
        })

        return broker_order

    # ── Position Management ───────────────────────────────

    def _update_position(self, symbol: str, side: OrderSide, quantity: float, price: float):
        """Update position after a fill."""
        side_mult = 1 if side == OrderSide.BUY else -1
        qty_delta = quantity * side_mult

        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + qty_delta

            if abs(new_qty) < 1e-10:
                # Position closed — compute correct PnL
                if old_qty > 0:  # was long: PnL = (exit - entry) * qty
                    pnl = round((price - pos.entry_price) * abs(qty_delta), 2)
                else:  # was short: PnL = (entry - exit) * |qty|
                    pnl = round((pos.entry_price - price) * abs(qty_delta), 2)
                self._total_pnl += pnl
                self._total_pnl = round(self._total_pnl, 2)
                del self._positions[symbol]
            else:
                # Update avg entry for position increases
                if (old_qty > 0 and qty_delta > 0) or (old_qty < 0 and qty_delta < 0):
                    # Adding to position
                    total_cost = (old_qty * pos.entry_price) + (qty_delta * price)
                    pos.entry_price = round(total_cost / new_qty, 2)
                pos.quantity = round(new_qty, 8)
                pos.current_price = price
                pos.side = "long" if new_qty > 0 else "short"
                pos.unrealized_pnl = round((price - pos.entry_price) * new_qty, 2)
        else:
            if abs(qty_delta) > 0:
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol,
                    quantity=round(qty_delta, 8),
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    side="long" if qty_delta > 0 else "short",
                )

        # Update equity tracking
        self._update_equity()

    def _update_equity(self):
        """Update equity and peak equity tracking."""
        total_equity = self._balance.get("USDT", 0)
        for pos in self._positions.values():
            total_equity += pos.quantity * pos.current_price

        self._peak_equity = max(self._peak_equity, total_equity)

    # ── Performance ───────────────────────────────────────

    @property
    def equity(self) -> float:
        """Current total equity (balance + position value)."""
        eq = self._balance.get("USDT", 0)
        for pos in self._positions.values():
            eq += pos.quantity * pos.current_price
        return round(eq, 2)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak equity."""
        current = self.equity
        if self._peak_equity > 0:
            return round((self._peak_equity - current) / self._peak_equity, 4)
        return 0.0

    @property
    def total_pnl(self) -> float:
        """Total realized PnL."""
        return self._total_pnl

    def get_performance(self) -> dict:
        """Get complete performance summary."""
        return {
            "initial_balance": self._initial_balance,
            "current_balance": self._balance.get("USDT", 0),
            "equity": self.equity,
            "peak_equity": self._peak_equity,
            "drawdown": self.drawdown,
            "total_pnl": self._total_pnl,
            "open_positions": len(self._positions),
            "total_trades": len(self._trade_history),
            "total_orders": len(self._orders),
        }

    def reset(self):
        """Reset paper broker to initial state."""
        self._balance = {"USDT": self._initial_balance}
        self._orders.clear()
        self._positions.clear()
        self._trade_history.clear()
        self._peak_equity = self._initial_balance
        self._total_pnl = 0.0
        logger.info("PaperBroker reset")
