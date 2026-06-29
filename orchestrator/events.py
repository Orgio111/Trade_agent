"""
QUANTEX Standardized Event Types — Event-driven architecture backbone.

Defines the canonical event types used across the entire system.
All services (orchestrator, realtime, execution, frontend) publish
and subscribe to these event types through NATS JetStream.

Architecture:
  ┌────────────┐    signals.raw     ┌─────────────┐   signals.aggregated   ┌───────────┐
  │ BrainRunners│ ────────────────→ │   NATS      │ ────────────────────→ │ Execution │
  │ (Layer A)  │   (MarketData,    │  JetStream  │                       │ (Layer C) │
  │            │    TradeSignal,    │             │                       │           │
  │            │    PortfolioEvent) │             │                       │           │
  └────────────┘                   └──────┬──────┘                       └───────────┘
                                          │
                                          │ ws.updates
                                          ▼
                                    ┌────────────┐
                                    │  Frontend   │
                                    │ (Layer C)   │
                                    └────────────┘

Event Types (subject-based routing):
  - signals.raw.>         : Raw brain signals (Layer A → NATS)
  - signals.aggregated    : Aggregated signals (NATS → Layer B)
  - signals.executed      : Execution confirmations (Layer C)
  - market.>              : Market data events
  - portfolio.>           : Portfolio/account events
  - system.>              : System/infrastructure events
  - rl.>                  : Reinforcement learning events
  - ws.>                  : WebSocket push events (to frontend)

Usage:
    from orchestrator.events import (
        EventType, quantex_event,
        TradeSignalEvent, MarketDataEvent, PortfolioEvent
    )

    # Create an event
    event = TradeSignalEvent(
        source="custom_nn",
        symbol="BTCUSDT",
        signal="long",
        confidence=0.85,
    )

    # Publish to NATS
    await nc.publish(event.subject, event.to_json().encode())

    # Subscribe
    sub = await nc.subscribe("signals.raw.>")
    async for msg in sub.messages:
        event = quantex_event(json.loads(msg.data))
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ── Event Type Enums ────────────────────────────────────────

class EventCategory(str, Enum):
    """Top-level event category (maps to NATS subject prefix)."""
    SIGNALS = "signals"
    MARKET = "market"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    RL = "rl"
    WS = "ws"


class SignalEventType(str, Enum):
    """Types of trading signal events."""
    RAW = "raw"
    AGGREGATED = "aggregated"
    EXECUTED = "executed"


class MarketEventType(str, Enum):
    """Types of market data events."""
    CANDLE = "candle"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    TRADE = "trade"


class PortfolioEventType(str, Enum):
    """Types of portfolio/account events."""
    PNL = "pnl"
    BALANCE = "balance"
    POSITION = "position"
    ORDER = "order"
    RISK = "risk"


class SystemEventType(str, Enum):
    """Types of system events."""
    HEALTH = "health"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEPLOY = "deploy"
    CONFIG = "config"


class RLEventType(str, Enum):
    """Types of reinforcement learning events."""
    REWARD = "reward"
    WEIGHT = "weight"
    TRAINING = "training"
    EVAL = "evaluation"


class WSEventType(str, Enum):
    """Types of WebSocket push events."""
    UPDATE = "update"
    SIGNAL = "signal"
    PORTFOLIO = "portfolio"


# ── Base Event ──────────────────────────────────────────────

@dataclass
class QuantexEvent:
    """
    Base event for all QUANTEX events.

    Every event has:
      - id:         Unique event ID (UUID)
      - timestamp:  Unix timestamp of event creation
      - source:     Component that created the event (e.g., "custom_nn", "orchestrator")
      - category:   Event category (determines NATS subject prefix)
      - subject:    Full NATS subject for publishing
      - metadata:   Arbitrary key-value metadata
    """

    id: str = ""
    timestamp: float = 0.0
    source: str = "unknown"
    category: str = ""
    subject: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.subject and self.category:
            self._build_subject()

    def _build_subject(self):
        """Build the NATS subject from category and type."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary (for serialization)."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert event to JSON string (for NATS publishing)."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "QuantexEvent":
        """Create event from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Signal Events ───────────────────────────────────────────

@dataclass
class TradeSignalEvent(QuantexEvent):
    """
    A trading signal from a brain runner.

    NATS Subject: signals.raw.<source>.<symbol>

    Attributes:
        symbol:     Trading pair (e.g., "BTCUSDT")
        signal:     "long", "short", or "hold"
        confidence: Confidence level (0-1)
        price:      Current price when signal was generated
        entry_price:  Optional entry price
        stop_loss:    Optional stop-loss price
        take_profits: Optional take-profit levels
        reason:       Human-readable reasoning
        brain_version: Brain model version
    """
    symbol: str = ""
    signal: str = "hold"  # "long", "short", "hold"
    confidence: float = 0.0
    price: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: Optional[list[dict]] = None  # [{"level": 1, "price": ..., "qty_pct": ...}]
    reason: str = ""
    brain_version: str = ""
    event_type: str = SignalEventType.RAW.value

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SIGNALS.value
        self.subject = f"signals.{self.event_type}.{self.source}.{self.symbol}"
        if self.take_profits is None:
            self.take_profits = []


@dataclass
class AggregatedSignalEvent(QuantexEvent):
    """
    Aggregated signal from the Go orchestrator (Trinity Layer B).

    NATS Subject: signals.aggregated

    Attributes:
        symbol:         Trading pair
        consensus:      "long", "short", "hold"
        confidence:     Aggregated confidence (0-1)
        weights:        Dict of brain -> weight used in aggregation
        brain_signals:  Dict of brain -> individual signal
        volatility:     Current volatility estimate
        regime:         Market regime label
    """
    symbol: str = ""
    consensus: str = "hold"
    confidence: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    brain_signals: dict[str, dict] = field(default_factory=dict)
    volatility: float = 0.0
    regime: str = "unknown"

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SIGNALS.value
        self.event_type = SignalEventType.AGGREGATED.value
        self.subject = "signals.aggregated"


@dataclass
class ExecutedSignalEvent(QuantexEvent):
    """
    Execution confirmation after a signal is traded.

    NATS Subject: signals.executed

    Attributes:
        symbol:         Trading pair
        signal:         Original signal direction
        order_id:       Exchange order ID
        filled_qty:     Filled quantity
        avg_price:      Average fill price
        status:         Order status ("filled", "partial", "rejected")
        execution_ms:   Execution latency in milliseconds
    """
    symbol: str = ""
    signal: str = "hold"
    order_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    status: str = "filled"
    execution_ms: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SIGNALS.value
        self.event_type = SignalEventType.EXECUTED.value
        self.subject = "signals.executed"


# ── Market Data Events ──────────────────────────────────────

@dataclass
class MarketDataEvent(QuantexEvent):
    """
    Market data update (candle, ticker, trade).

    NATS Subject: market.<type>.<symbol>

    Attributes:
        symbol: Trading pair
        event_type: "candle", "ticker", "trade", "orderbook"
        data: Market data payload (dict)
    """
    symbol: str = ""
    event_type: str = "ticker"
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.MARKET.value
        self.subject = f"market.{self.event_type}.{self.symbol}"


@dataclass
class OrderbookEvent(QuantexEvent):
    """
    Order book snapshot or update.

    NATS Subject: market.orderbook.<symbol>

    Attributes:
        symbol:    Trading pair
        bids:      List of [price, quantity] tuples
        asks:      List of [price, quantity] tuples
        imbalance: Order book imbalance ratio (-1 to 1)
        spread:    Current spread in quote currency
        mid_price: Mid price
    """
    symbol: str = ""
    bids: list[list[float]] = field(default_factory=list)
    asks: list[list[float]] = field(default_factory=list)
    imbalance: float = 0.0
    spread: float = 0.0
    mid_price: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.MARKET.value
        self.event_type = MarketEventType.ORDERBOOK.value
        self.subject = f"market.orderbook.{self.symbol}"

    def compute_imbalance(self) -> float:
        """Compute order book imbalance from bids/asks.

        Returns:
            Imbalance ratio: positive = bid-heavy (bullish), negative = ask-heavy (bearish)
        """
        total_bid_qty = sum(b[1] for b in self.bids if len(b) > 1)
        total_ask_qty = sum(a[1] for a in self.asks if len(a) > 1)
        total = total_bid_qty + total_ask_qty
        if total == 0:
            return 0.0
        return (total_bid_qty - total_ask_qty) / total


# ── Portfolio Events ────────────────────────────────────────

@dataclass
class PortfolioEvent(QuantexEvent):
    """
    Portfolio/account update.

    NATS Subject: portfolio.<type>

    Attributes:
        event_type:    "pnl", "balance", "position", "order", "risk"
        balance:       Current balance
        equity:        Current equity
        total_pnl:     Total realized PnL
        drawdown:      Current drawdown
        open_positions: Number of open positions
    """
    event_type: str = "pnl"
    balance: float = 0.0
    equity: float = 0.0
    total_pnl: float = 0.0
    drawdown: float = 0.0
    open_positions: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.PORTFOLIO.value
        self.subject = f"portfolio.{self.event_type}"


# ── RL Events ───────────────────────────────────────────────

@dataclass
class RLEvent(QuantexEvent):
    """
    Reinforcement learning event.

    NATS Subject: rl.<type>

    Attributes:
        event_type: "reward", "weight", "training", "evaluation"
        agent_id:   RL agent identifier
        reward:     Current reward value (for reward events)
        weights:    Updated weights (for weight update events)
        metrics:    Training metrics (for training/eval events)
    """
    event_type: str = "reward"
    agent_id: str = "ppo_portfolio"
    reward: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.RL.value
        self.subject = f"rl.{self.event_type}.{self.agent_id}"


# ── WebSocket Push Events ───────────────────────────────────

@dataclass
class WSEvent(QuantexEvent):
    """
    WebSocket push event (sent to frontend clients).

    NATS Subject: ws.<type>

    The Go orchestrator subscribes to ws.* and forwards to connected
    WebSocket clients when NATS is used as the backend transport.

    Attributes:
        event_type: Type of WS update
        payload:    Data to push to frontend
    """
    event_type: str = "update"
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.WS.value
        self.subject = f"ws.{self.event_type}"


# ── System Events ───────────────────────────────────────────

@dataclass
class SystemEvent(QuantexEvent):
    """
    System/infrastructure event.

    NATS Subject: system.<type>

    Attributes:
        event_type: "health", "error", "warning", "info", "deploy", "config"
        level:      Severity ("info", "warning", "error", "critical")
        message:    Human-readable message
        component:  Component name
    """
    event_type: str = "info"
    level: str = "info"
    message: str = ""
    component: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SYSTEM.value
        self.subject = f"system.{self.event_type}"


# ── Factory ──────────────────────────────────────────────────

def quantex_event(data: dict) -> QuantexEvent:
    """Create the appropriate event type from a dict.

    Inspects the 'category' and 'event_type' fields to determine
    the correct dataclass.

    Args:
        data: Event dictionary (from JSON deserialization)

    Returns:
        Typed QuantexEvent subclass instance.
    """
    category = data.get("category", "")
    event_type = data.get("event_type", "")

    if category == EventCategory.SIGNALS.value:
        if event_type == SignalEventType.RAW.value:
            return TradeSignalEvent.from_dict(data)
        elif event_type == SignalEventType.AGGREGATED.value:
            return AggregatedSignalEvent.from_dict(data)
        elif event_type == SignalEventType.EXECUTED.value:
            return ExecutedSignalEvent.from_dict(data)

    elif category == EventCategory.MARKET.value:
        if event_type == MarketEventType.ORDERBOOK.value:
            return OrderbookEvent.from_dict(data)
        return MarketDataEvent.from_dict(data)

    elif category == EventCategory.PORTFOLIO.value:
        return PortfolioEvent.from_dict(data)

    elif category == EventCategory.RL.value:
        return RLEvent.from_dict(data)

    elif category == EventCategory.WS.value:
        return WSEvent.from_dict(data)

    elif category == EventCategory.SYSTEM.value:
        return SystemEvent.from_dict(data)

    # Fallback
    return QuantexEvent.from_dict(data)


# ── Subject Helpers ─────────────────────────────────────────

def raw_signal_subject(source: str, symbol: str) -> str:
    """Build NATS subject for raw brain signals."""
    return f"signals.raw.{source}.{symbol}"


def market_data_subject(event_type: str, symbol: str) -> str:
    """Build NATS subject for market data events."""
    return f"market.{event_type}.{symbol}"


def portfolio_subject(event_type: str) -> str:
    """Build NATS subject for portfolio events."""
    return f"portfolio.{event_type}"


def rl_subject(event_type: str, agent_id: str = "ppo_portfolio") -> str:
    """Build NATS subject for RL events."""
    return f"rl.{event_type}.{agent_id}"


def ws_subject(event_type: str) -> str:
    """Build NATS subject for WebSocket push events."""
    return f"ws.{event_type}"


def system_subject(event_type: str) -> str:
    """Build NATS subject for system events."""
    return f"system.{event_type}"


# ── NATS Stream Configuration ───────────────────────────────

# Default stream configurations for NATS JetStream
# Used by the orchestrator's NATS setup to ensure streams exist

NATS_STREAMS: dict[str, dict] = {
    "signals": {
        "subjects": ["signals.raw.>", "signals.aggregated", "signals.executed"],
        "storage": "file",
        "max_age_days": 7,
        "max_size_gb": 10,
        "description": "All brain trading signals (raw → aggregated → executed)",
    },
    "market": {
        "subjects": ["market.>"],
        "storage": "file",
        "max_age_days": 3,
        "max_size_gb": 5,
        "description": "Market data events (candles, orderbooks, trades, tickers)",
    },
    "portfolio": {
        "subjects": ["portfolio.>"],
        "storage": "file",
        "max_age_days": 30,
        "max_size_gb": 1,
        "description": "Portfolio and account events",
    },
    "rl": {
        "subjects": ["rl.>"],
        "storage": "file",
        "max_age_days": 14,
        "max_size_gb": 1,
        "description": "Reinforcement learning events (rewards, weights, metrics)",
    },
    "ws": {
        "subjects": ["ws.>"],
        "storage": "memory",
        "max_age_days": 0,  # Don't persist WebSocket events
        "max_size_gb": 0,
        "description": "WebSocket push events (not persisted)",
    },
    "system": {
        "subjects": ["system.>"],
        "storage": "memory",
        "max_age_days": 7,
        "max_size_gb": 1,
        "description": "System events (health, errors, warnings)",
    },
}
