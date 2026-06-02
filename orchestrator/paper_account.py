"""
QUANTEX Paper Trading Engine — Simulates exchange behavior for safe strategy testing.

Features:
- Realistic fee structure (maker/taker)
- Configurable slippage and latency
- Order fill simulation with partial fills
- Balance, equity, and PnL tracking
- Full trade history
- Stop loss and take profit auto-management
"""
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable


# ── Data Structures ──────────────────────────────────────

@dataclass
class OrderRequest:
    symbol: str
    side: str  # 'buy' | 'sell'
    order_type: str  # 'market' | 'limit'
    quantity: float
    price: Optional[float] = None
    reduce_only: bool = False
    post_only: bool = False
    stop_loss: Optional[float] = None
    take_profits: Optional[list[dict]] = None


@dataclass
class Fill:
    price: float
    quantity: float
    fee: float
    timestamp: int
    maker: bool


@dataclass
class PaperOrder:
    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    price: Optional[float]
    avg_fill_price: Optional[float]
    status: str  # 'open', 'filled', 'cancelled', 'rejected'
    fees: float
    fills: list[Fill]
    created_at: int
    reduce_only: bool
    post_only: bool
    stop_loss: Optional[float]
    take_profits: Optional[list[dict]]


@dataclass
class PaperPosition:
    id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    leverage: int
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float
    stop_loss: Optional[float]
    take_profits: Optional[list[dict]]
    opened_at: int
    order_id: str


@dataclass
class PaperTrade:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    pnl: float
    pnl_pct: float
    fees: float
    entry_time: int
    exit_time: int
    exit_reason: str  # 'tp', 'sl', 'manual', 'signal'
    entry_order_id: str
    exit_order_id: Optional[str] = None


# ── Paper Account ────────────────────────────────────────

class PaperAccount:
    """
    Simulated exchange account for paper trading.
    
    Models realistic behavior:
    - Maker/taker fee structure
    - Configurable slippage
    - Order book fill simulation
    - SL/TP auto-management
    - Balance and PnL tracking
    """

    def __init__(
        self,
        initial_balance: float = 10.0,
        maker_fee: float = 0.0002,    # 0.02%
        taker_fee: float = 0.0004,    # 0.04%
        slippage_bps: float = 0.3,    # 0.3 bps (very tight for paper)
        max_leverage: int = 10,
    ):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.equity = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.max_leverage = max_leverage

        self.positions: dict[str, PaperPosition] = {}  # symbol -> position
        self.orders: dict[str, PaperOrder] = {}
        self.trades: list[PaperTrade] = []
        self.closed_positions: list[PaperPosition] = []

        self._consecutive_losses = 0
        self._total_trades = 0
        self._winning_trades = 0
        self._peak_balance = initial_balance

        # Price feed (must be updated externally)
        self._price_feed: dict[str, dict] = {}

    # ── Price Feed ────────────────────────────────────

    def update_price(self, symbol: str, bid: float, ask: float, last: float):
        """Update current market price for a symbol."""
        self._price_feed[symbol] = {
            "bid": bid,
            "ask": ask,
            "last": last,
            "timestamp": int(time.time() * 1000),
        }
        self._check_positions(symbol)

    # ── Order Execution ───────────────────────────────

    def place_order(self, request: OrderRequest) -> PaperOrder:
        """Place an order and attempt to fill it."""
        if request.quantity <= 0:
            return self._reject_order(request, "Invalid quantity")

        if request.order_type == "market":
            return self._execute_market(request)
        else:
            return self._place_limit(request)

    def _execute_market(self, request: OrderRequest) -> PaperOrder:
        """Execute a market order with slippage simulation."""
        symbol = request.symbol
        feed = self._price_feed.get(symbol)
        if not feed:
            return self._reject_order(request, f"No price feed for {symbol}")

        # Determine fill price with slippage
        if request.side == "buy":
            base_price = feed["ask"]
            fill_price = base_price * (1 + self.slippage_bps / 10000)
        else:
            base_price = feed["bid"]
            fill_price = base_price * (1 - self.slippage_bps / 10000)

        # Calculate fee
        fee = fill_price * request.quantity * self.taker_fee
        notional = fill_price * request.quantity

        # Check if we have enough balance
        if request.side == "buy" and notional > self.balance:
            return self._reject_order(request, f"Insufficient balance: ${notional:.2f} > ${self.balance:.2f}")

        # Create order
        order = PaperOrder(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=request.side,
            order_type="market",
            quantity=request.quantity,
            filled_quantity=request.quantity,
            price=fill_price,
            avg_fill_price=fill_price,
            status="filled",
            fees=fee,
            fills=[Fill(price=fill_price, quantity=request.quantity, fee=fee, timestamp=int(time.time() * 1000), maker=False)],
            created_at=int(time.time() * 1000),
            reduce_only=request.reduce_only,
            post_only=False,
            stop_loss=request.stop_loss,
            take_profits=request.take_profits,
        )
        self.orders[order.id] = order

        # Update balance
        if request.side == "buy":
            self.balance -= notional + fee
        else:
            self.balance += notional - fee

        # Track position
        self._update_position(order)

        return order

    def _place_limit(self, request: OrderRequest) -> PaperOrder:
        """Place a limit order (may fill immediately if price crosses)."""
        feed = self._price_feed.get(request.symbol)
        fill_immediately = False
        fill_price = request.price or 0

        if feed and request.price:
            if request.side == "buy" and request.price >= feed["ask"]:
                fill_immediately = True
                fill_price = feed["ask"]
            elif request.side == "sell" and request.price <= feed["bid"]:
                fill_immediately = True
                fill_price = feed["bid"]

        if fill_immediately and fill_price > 0:
            fee = fill_price * request.quantity * self.maker_fee
            order = PaperOrder(
                id=str(uuid.uuid4()),
                symbol=request.symbol,
                side=request.side,
                order_type="limit",
                quantity=request.quantity,
                filled_quantity=request.quantity,
                price=request.price,
                avg_fill_price=fill_price,
                status="filled",
                fees=fee,
                fills=[Fill(price=fill_price, quantity=request.quantity, fee=fee, timestamp=int(time.time() * 1000), maker=True)],
                created_at=int(time.time() * 1000),
                reduce_only=request.reduce_only,
                post_only=request.post_only,
                stop_loss=request.stop_loss,
                take_profits=request.take_profits,
            )
            self.orders[order.id] = order
            self._update_position(order)
            return order

        # Order stays open
        order = PaperOrder(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            side=request.side,
            order_type="limit",
            quantity=request.quantity,
            filled_quantity=0,
            price=request.price,
            avg_fill_price=None,
            status="open",
            fees=0,
            fills=[],
            created_at=int(time.time() * 1000),
            reduce_only=request.reduce_only,
            post_only=request.post_only,
            stop_loss=request.stop_loss,
            take_profits=request.take_profits,
        )
        self.orders[order.id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id in self.orders and self.orders[order_id].status == "open":
            self.orders[order_id].status = "cancelled"
            return True
        return False

    def _reject_order(self, request: OrderRequest, reason: str) -> PaperOrder:
        order = PaperOrder(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            filled_quantity=0,
            price=request.price,
            avg_fill_price=None,
            status="rejected",
            fees=0,
            fills=[],
            created_at=int(time.time() * 1000),
            reduce_only=request.reduce_only,
            post_only=request.post_only,
            stop_loss=request.stop_loss,
            take_profits=request.take_profits,
        )
        self.orders[order.id] = order
        return order

    # ── Position Management ──────────────────────────

    def _update_position(self, order: PaperOrder):
        """Update position after an order fill."""
        current = self.positions.get(order.symbol)
        qty = order.filled_quantity if order.side == "buy" else -order.filled_quantity

        if current:
            # Merge with existing position
            total_qty = (current.quantity if current.side == "long" else -current.quantity) + qty
            if abs(total_qty) < 0.0001:
                # Position closed
                self._close_position(current, order)
                return

            new_side = "long" if total_qty > 0 else "short"
            new_entry = ((current.entry_price * current.quantity) +
                        (order.avg_fill_price * abs(qty))) / abs(total_qty)
            
            current.quantity = abs(total_qty)
            current.side = new_side
            current.entry_price = new_entry
            current.current_price = order.avg_fill_price or current.current_price
            current.stop_loss = order.stop_loss or current.stop_loss
            current.take_profits = order.take_profits or current.take_profits
        else:
            # New position
            position = PaperPosition(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                side="long" if qty > 0 else "short",
                entry_price=order.avg_fill_price or 0,
                current_price=order.avg_fill_price or 0,
                quantity=abs(qty),
                leverage=self.max_leverage,
                unrealized_pnl=0,
                realized_pnl=0,
                fees_paid=order.fees,
                stop_loss=order.stop_loss,
                take_profits=order.take_profits,
                opened_at=order.created_at,
                order_id=order.id,
            )
            self.positions[order.symbol] = position

        self._total_trades += 1

    def _close_position(self, position: PaperPosition, order: PaperOrder):
        """Close a position and record the trade."""
        exit_price = order.avg_fill_price or position.current_price
        direction_mult = 1 if position.side == "long" else -1
        pnl = (exit_price - position.entry_price) * position.quantity * position.leverage * direction_mult
        pnl -= order.fees

        trade = PaperTrade(
            id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl / (position.entry_price * position.quantity) if position.entry_price * position.quantity > 0 else 0,
            fees=position.fees_paid + order.fees,
            entry_time=position.opened_at,
            exit_time=order.created_at,
            exit_reason="signal",
            entry_order_id=position.order_id,
            exit_order_id=order.id,
        )
        self.trades.append(trade)

        if pnl > 0:
            self._winning_trades += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self.balance += pnl
        if self.balance > self._peak_balance:
            self._peak_balance = self.balance

        position.realized_pnl = pnl
        position.current_price = exit_price
        self.closed_positions.append(position)
        del self.positions[position.symbol]

    # ── SL/TP Auto-Management ────────────────────────

    def _check_positions(self, symbol: str):
        """Check SL/TP levels for all positions on price update."""
        position = self.positions.get(symbol)
        if not position:
            return

        feed = self._price_feed.get(symbol)
        if not feed:
            return

        price = feed["last"]
        position.current_price = price

        # Update unrealized PnL
        direction_mult = 1 if position.side == "long" else -1
        position.unrealized_pnl = (
            (price - position.entry_price) * position.quantity * position.leverage * direction_mult
        )

        # Check stop loss
        if position.stop_loss is not None:
            hit_sl = (
                (position.side == "long" and price <= position.stop_loss) or
                (position.side == "short" and price >= position.stop_loss)
            )
            if hit_sl:
                self._execute_close(position, "sl")
                return

        # Check take profits
        if position.take_profits:
            for tp in position.take_profits:
                if not tp.get("filled", False):
                    hit_tp = (
                        (position.side == "long" and price >= tp["price"]) or
                        (position.side == "short" and price <= tp["price"])
                    )
                    if hit_tp:
                        tp["filled"] = True
                        qty_to_close = position.quantity * tp.get("qty_pct", 0.5)

                        if tp.get("level") == 1:
                            # Move SL to breakeven after first TP
                            position.stop_loss = position.entry_price * 1.001
                        elif tp.get("level", 0) >= 2:
                            # Close remaining on final TP
                            self._execute_close(position, "tp")
                            return

    def _execute_close(self, position: PaperPosition, reason: str):
        """Execute a forced close (SL/TP hit)."""
        feed = self._price_feed.get(position.symbol, {})
        exit_price = feed.get("last", position.current_price)
        direction_mult = 1 if position.side == "long" else -1
        pnl = (exit_price - position.entry_price) * position.quantity * position.leverage * direction_mult

        trade = PaperTrade(
            id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            pnl=pnl,
            pnl_pct=pnl / (position.entry_price * position.quantity) if position.entry_price * position.quantity > 0 else 0,
            fees=position.fees_paid,
            entry_time=position.opened_at,
            exit_time=int(time.time() * 1000),
            exit_reason=reason,
            entry_order_id=position.order_id,
        )
        self.trades.append(trade)

        if pnl > 0:
            self._winning_trades += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self.balance += pnl
        if self.balance > self._peak_balance:
            self._peak_balance = self.balance

        position.realized_pnl = pnl
        self.closed_positions.append(position)
        del self.positions[position.symbol]

    # ── Portfolio Metrics ────────────────────────────

    def get_portfolio(self) -> dict:
        """Get full portfolio snapshot."""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.equity = self.balance + total_unrealized

        # Calculate drawdown
        dd = (self._peak_balance - self.equity) / self._peak_balance if self._peak_balance > 0 else 0

        # Win rate
        win_rate = self._winning_trades / self._total_trades if self._total_trades > 0 else 0

        return {
            "balance": round(self.balance, 4),
            "equity": round(self.equity, 4),
            "unrealized_pnl": round(total_unrealized, 4),
            "open_positions": len(self.positions),
            "drawdown": round(dd, 6),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": round(win_rate, 4),
            "consecutive_losses": self._consecutive_losses,
            "total_pnl": round(self.balance - self.initial_balance, 4),
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        return [
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": round(p.entry_price, 2),
                "current_price": round(p.current_price, 2),
                "quantity": round(p.quantity, 6),
                "leverage": p.leverage,
                "unrealized_pnl": round(p.unrealized_pnl, 4),
                "stop_loss": round(p.stop_loss, 2) if p.stop_loss else None,
                "take_profits": p.take_profits,
                "opened_at": p.opened_at,
            }
            for p in self.positions.values()
        ]

    def get_trades(self, limit: int = 50) -> list[dict]:
        """Get recent trade history."""
        recent = sorted(self.trades, key=lambda t: t.exit_time, reverse=True)[:limit]
        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "quantity": round(t.quantity, 6),
                "pnl": round(t.pnl, 4),
                "pnl_pct": round(t.pnl_pct, 4),
                "fees": round(t.fees, 4),
                "exit_reason": t.exit_reason,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            }
            for t in recent
        ]

    def reset(self, balance: Optional[float] = None):
        """Reset the account to initial state."""
        self.balance = balance or self.initial_balance
        self.equity = self.balance
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.closed_positions.clear()
        self._consecutive_losses = 0
        self._total_trades = 0
        self._winning_trades = 0
        self._peak_balance = self.balance
