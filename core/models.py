"""Canonical data models shared across all agents."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TickData(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    spread_bps: float | None = None


class TechnicalSignal(BaseModel):
    symbol: str
    timestamp: datetime
    rsi_14: float
    ema_fast: float
    ema_slow: float
    atr_14: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    volume_ratio: float
    trend: Side
    confidence: float = Field(ge=0.0, le=1.0)


class FundamentalSignal(BaseModel):
    symbol: str
    timestamp: datetime
    on_chain_score: float = Field(ge=-1.0, le=1.0)
    nvt_ratio: float | None = None
    mvrv_zscore: float | None = None
    earnings_surprise_pct: float | None = None
    revenue_growth_yoy: float | None = None
    trend: Side
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class SentimentSignal(BaseModel):
    symbol: str
    timestamp: datetime
    news_score: float = Field(ge=-1.0, le=1.0)
    social_score: float = Field(ge=-1.0, le=1.0)
    fear_greed_index: float = Field(ge=0.0, le=100.0)
    trending_topics: list[str] = Field(default_factory=list)
    trend: Side
    confidence: float = Field(ge=0.0, le=1.0)
    raw_headlines: list[str] = Field(default_factory=list)


class DebateArgument(BaseModel):
    agent_name: str
    position: Side
    argument: str
    quantitative_score: float = Field(ge=0.0, le=1.0)
    supporting_factors: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)


class CouncilDecision(BaseModel):
    symbol: str
    timestamp: datetime
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bull_score: float = Field(ge=0.0, le=1.0)
    bear_score: float = Field(ge=0.0, le=1.0)
    consensus_score: float = Field(ge=0.0, le=1.0)
    final_side: Side
    debate_log: list[DebateArgument] = Field(default_factory=list)
    rationale: str = ""
    technical: TechnicalSignal | None = None
    fundamental: FundamentalSignal | None = None
    sentiment: SentimentSignal | None = None
    features: FeatureSignal | None = None


class RiskReport(BaseModel):
    symbol: str
    timestamp: datetime
    session_id: str
    var_95: float
    var_99: float
    cvar_99: float
    kelly_raw: float
    kelly_fractional: float
    position_size_usd: float
    position_size_units: float
    stop_loss_price: float
    take_profit_price: float
    max_drawdown_pct: float
    approved: bool
    rejection_reason: str | None = None


class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    avg_fill_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    slippage_bps: float | None = None
    exchange_order_id: str | None = None
    supervisor_rationale: str = ""


class TradeOutcome(BaseModel):
    order_id: str
    session_id: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float | None = None
    quantity: float
    pnl: float | None = None
    pnl_pct: float | None = None
    holding_period_s: float | None = None
    win: bool | None = None
    failure_analysis: str | None = None
    updated_prompts: dict[str, str] = Field(default_factory=dict)


class PortfolioState(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    equity: float
    cash: float
    positions: dict[str, float] = Field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    kill_switch_active: bool = False


class FeatureSignal(BaseModel):
    """Order-flow, funding, and open-interest derived features."""
    symbol: str
    timestamp: datetime
    ofi: float | None = None
    """Order Flow Imbalance over the current bar: (buyVol - sellVol) / (buyVol + sellVol).
    Range [-1, 1]; positive = aggressive buying pressure."""
    cvd: float | None = None
    """Cumulative Volume Delta (rolling) — cumulative sum of (buy_taker_vol - sell_taker_vol)."""
    cvd_delta: float | None = None
    """CVD change over the last bar."""
    funding_rate: float | None = None
    """Latest perpetual funding rate from the primary exchange (annualized %)."""
    funding_rate_delta: float | None = None
    """Funding rate change vs previous value."""
    open_interest: float | None = None
    """Latest open interest value."""
    open_interest_delta: float | None = None
    """Open interest change over the last poll interval."""
    oi_price_delta_corr: float | None = None
    """Short-term rolling correlation between OI delta and price delta.
    Positive → OI rising with price (trend strength).
    Negative → OI rising while price falls (potential reversal)."""
    trade_strength: float | None = None
    """Normalised trade intensity over the bar — (trade_count / avg_trade_count) in [0, 2]."""
    trend: Side = Side.HOLD
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AlphaFactors(BaseModel):
    symbol: str
    timestamp: datetime
    session_id: str
    momentum_score: float = Field(ge=-1.0, le=1.0)
    value_score: float = Field(ge=-1.0, le=1.0)
    quality_score: float = Field(ge=-1.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    combined_alpha: float = Field(ge=-1.0, le=1.0)
    raw_signals: dict[str, Any] = Field(default_factory=dict)
