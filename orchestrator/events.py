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
      - priority:   Event priority ("low", "medium", "high", "critical")
      - trace_id:   Session/correlation ID for end-to-end tracing
      - context:    Shared context (session_state, market_regime, last_signal, etc.)
      - metadata:   Arbitrary key-value metadata
    """

    id: str = ""
    timestamp: float = 0.0
    source: str = "unknown"
    category: str = ""
    subject: str = ""
    priority: str = "medium"  # low, medium, high, critical
    trace_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)
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
        """Create event from dictionary.

        Uses dataclasses.fields() which properly walks the class hierarchy
        to collect all fields (parent + child), so base fields like id,
        timestamp, source, etc. are preserved during deserialization.
        """
        from dataclasses import fields as dc_fields
        all_field_names = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in all_field_names})


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


# ── VLM (Vision Language Model) Events ─────────────────────

class VLMEventType(str, Enum):
    """Types of VLM (vision) events."""
    CHART_SNAPSHOT = "chart_snapshot"
    VLM_ANALYSIS = "vlm_analysis"


@dataclass
class ChartSnapshotEvent(QuantexEvent):
    """
    Chart image captured for VLM analysis.

    NATS Subject: agent.vlm.chart_snapshot.<symbol>

    Attributes:
        symbol:    Trading pair
        image_url: Path/URL to chart image
        timeframe: Candle interval used for chart
        indicators: List of indicators overlaid on chart
    """
    symbol: str = ""
    image_url: str = ""
    timeframe: str = "1m"
    indicators: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.category = "agent"
        self.event_type = VLMEventType.CHART_SNAPSHOT.value
        self.priority = "medium"
        self.subject = f"agent.vlm.chart_snapshot.{self.symbol}"


@dataclass
class VLMAnalysisEvent(QuantexEvent):
    """
    VLM analysis result — chart pattern recognition output.

    NATS Subject: agent.vlm.analysis.<symbol>

    Attributes:
        symbol:      Trading pair
        trend:       "bullish", "bearish", "neutral"
        pattern:     Detected chart pattern (e.g., "ascending_triangle")
        support:     Key support level
        resistance:  Key resistance level
        confidence:  Analysis confidence (0-1)
        reasoning:   Human-readable reasoning
    """
    symbol: str = ""
    trend: str = "neutral"
    pattern: str = ""
    support: float = 0.0
    resistance: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.category = "agent"
        self.event_type = VLMEventType.VLM_ANALYSIS.value
        self.priority = "medium"
        self.subject = f"agent.vlm.analysis.{self.symbol}"


# ── RAG (Retrieval Augmented Generation) Events ─────────────

class RAGEventType(str, Enum):
    """Types of RAG events."""
    QUERY = "rag_query"
    RESULT = "rag_result"


@dataclass
class RAGQueryEvent(QuantexEvent):
    """
    RAG query request — search vector store for similar patterns.

    NATS Subject: agent.rag.query.<symbol>

    Attributes:
        symbol: Trading pair
        query:  Natural language query string
        top_k:  Number of results to return
        filters: Optional metadata filters
    """
    symbol: str = ""
    query: str = ""
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.category = "agent"
        self.event_type = RAGEventType.QUERY.value
        self.priority = "low"
        self.subject = f"agent.rag.query.{self.symbol}"


@dataclass
class RAGResultEvent(QuantexEvent):
    """
    RAG retrieval result — documents found from vector search.

    NATS Subject: agent.rag.result.<symbol>

    Attributes:
        symbol:    Trading pair
        documents: List of retrieved documents with scores
        metadata:  Pattern metadata (return, duration, etc.)
        query_time_ms: Retrieval latency
    """
    symbol: str = ""
    documents: list[dict] = field(default_factory=list)
    query_time_ms: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.category = "agent"
        self.event_type = RAGEventType.RESULT.value
        self.priority = "low"
        self.subject = f"agent.rag.result.{self.symbol}"


# ── Risk Decision Events ────────────────────────────────────

class RiskEventType(str, Enum):
    """Types of risk decision events."""
    CHECK = "risk_check"
    APPROVED = "risk_approved"
    REJECTED = "risk_rejected"


@dataclass
class RiskCheckEvent(QuantexEvent):
    """
    Risk check request — proposed trade submitted for evaluation.

    NATS Subject: system.risk.check

    Attributes:
        symbol:            Trading pair
        signal:            Proposed signal direction
        confidence:        Signal confidence
        entry_price:       Proposed entry price
        stop_loss:         Proposed stop loss
        position_size_pct: Proposed position size as % of equity
        max_position_size: Max allowed position size
        current_exposure:  Current portfolio exposure
        drawdown:          Current drawdown
        volatility:        Current volatility regime
    """
    symbol: str = ""
    signal: str = "hold"
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    position_size_pct: float = 0.0
    max_position_size: float = 0.0
    current_exposure: float = 0.0
    drawdown: float = 0.0
    volatility: str = "normal"

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SYSTEM.value
        self.event_type = RiskEventType.CHECK.value
        self.priority = "high"
        self.subject = "system.risk.check"


@dataclass
class RiskApprovedEvent(QuantexEvent):
    """
    Risk approved — trade cleared for execution.

    NATS Subject: system.risk.approved

    Attributes:
        symbol:        Trading pair
        signal:        Approved signal direction
        approved_size: Adjusted/approved position size
        max_leverage:  Max allowed leverage
        risk_score:    Composite risk score (0-1)
        reasoning:     Risk evaluation reasoning
    """
    symbol: str = ""
    signal: str = "hold"
    approved_size: float = 0.0
    max_leverage: int = 1
    risk_score: float = 0.0
    reasoning: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SYSTEM.value
        self.event_type = RiskEventType.APPROVED.value
        self.priority = "high"
        self.subject = "system.risk.approved"


@dataclass
class RiskRejectedEvent(QuantexEvent):
    """
    Risk rejected — trade blocked by risk engine.

    NATS Subject: system.risk.rejected

    Attributes:
        symbol:  Trading pair
        signal:  Rejected signal direction
        reason:  Rejection reason
        severity: "hard" (absolute block) or "soft" (size reduction)
        cooldown_minutes: How long to wait before retry
    """
    symbol: str = ""
    signal: str = "hold"
    reason: str = ""
    severity: str = "hard"
    cooldown_minutes: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SYSTEM.value
        self.event_type = RiskEventType.REJECTED.value
        self.priority = "critical"
        self.subject = "system.risk.rejected"


# ── System Events ───────────────────────────────────────────

@dataclass
# ── Scalping Events ──────────────────────────────────────────

class ScalpingEventType(str, Enum):
    """Types of scalping decision events."""
    DECISION = "scalp_decision"
    EXECUTED = "scalp_executed"


@dataclass
class ScalpDecisionEvent(QuantexEvent):
    """
    Scalping engine decision — ultra-fast CPU-only signal.

    NATS Subject: signals.scalp.<symbol>

    Attributes:
        symbol:     Trading pair
        action:     "BUY", "SELL", or "HOLD"
        confidence: 0-100
        size_pct:   0-5 (position size as % of equity)
        reason:     Short keyword reason
        volatility: Current volatility level
        latency_ms: Decision latency in milliseconds
    """
    symbol: str = ""
    action: str = "HOLD"
    confidence: int = 0
    size_pct: float = 0.0
    reason: str = ""
    volatility: str = "medium"
    latency_ms: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SIGNALS.value
        self.event_type = ScalpingEventType.DECISION.value
        self.priority = "high"
        self.subject = f"signals.scalp.{self.symbol}"


@dataclass
class ScalpExecutedEvent(QuantexEvent):
    """
    Scalping execution confirmation.

    NATS Subject: signals.scalp.executed.<symbol>

    Attributes:
        symbol:     Trading pair
        action:     Executed action
        price:      Fill price
        size_pct:   Executed size
        fill_ms:    Execution latency
    """
    symbol: str = ""
    action: str = "HOLD"
    price: float = 0.0
    size_pct: float = 0.0
    fill_ms: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.category = EventCategory.SIGNALS.value
        self.event_type = ScalpingEventType.EXECUTED.value
        self.priority = "high"
        self.subject = f"signals.scalp.executed.{self.symbol}"


# ── System Events ───────────────────────────────────────────

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
        elif event_type == ScalpingEventType.DECISION.value:
            return ScalpDecisionEvent.from_dict(data)
        elif event_type == ScalpingEventType.EXECUTED.value:
            return ScalpExecutedEvent.from_dict(data)

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
        if event_type == RiskEventType.CHECK.value:
            return RiskCheckEvent.from_dict(data)
        elif event_type == RiskEventType.APPROVED.value:
            return RiskApprovedEvent.from_dict(data)
        elif event_type == RiskEventType.REJECTED.value:
            return RiskRejectedEvent.from_dict(data)
        return SystemEvent.from_dict(data)

    elif category == "agent":
        if event_type == VLMEventType.CHART_SNAPSHOT.value:
            return ChartSnapshotEvent.from_dict(data)
        elif event_type == VLMEventType.VLM_ANALYSIS.value:
            return VLMAnalysisEvent.from_dict(data)
        elif event_type == RAGEventType.QUERY.value:
            return RAGQueryEvent.from_dict(data)
        elif event_type == RAGEventType.RESULT.value:
            return RAGResultEvent.from_dict(data)

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
    "agent": {
        "subjects": ["agent.>"],
        "storage": "memory",
        "max_age_days": 3,
        "max_size_gb": 2,
        "description": "Agent events (VLM analysis, RAG queries, inter-agent communication)",
    },
}
