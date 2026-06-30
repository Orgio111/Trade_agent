"""Pydantic models for the trading system."""

from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
from datetime import datetime
from enum import Enum


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class RegimeState(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOLATILE = "volatile"
    TRANSITION = "transition"


class Candle(BaseModel):
    """1-minute OHLCV candle."""
    timestamp: int  # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = "BTCUSDT"
    
    def to_dict(self) -> dict:
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: dict) -> "Candle":
        return cls(**data)
    
    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3
    
    @property
    def vwap_component(self) -> float:
        return self.typical_price * self.volume


class CandleFeatures(BaseModel):
    """Technical features extracted from candles."""
    # Returns
    ret_1: float = 0.0
    ret_5: float = 0.0
    ret_15: float = 0.0
    ret_60: float = 0.0
    
    # Volatility
    atr_14: float = 0.0
    atr_pct: float = 0.0
    bb_width: float = 0.0
    bb_pct: float = 0.0
    
    # Momentum
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    
    # Trend
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    
    # Volume
    volume_ratio: float = 1.0
    obv: float = 0.0
    vwap_dev: float = 0.0
    
    # Market structure
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False
    
    stoch_k: float = 50.0
    willr: float = -50.0
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for model input."""
        return [
            self.ret_1, self.ret_5, self.ret_15, self.ret_60,
            self.atr_14, self.atr_pct, self.bb_width,
            self.rsi_14, self.macd, self.macd_signal, self.macd_hist,
            self.ema_9, self.ema_21, self.ema_50, self.adx,
            self.volume_ratio, self.obv, self.vwap_dev,
            float(self.higher_high), float(self.higher_low),
            float(self.lower_high), float(self.lower_low),
            self.stoch_k, self.willr,
        ]


class Signal(BaseModel):
    """Trading signal from model."""
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    size_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    regime: Optional[RegimeState] = None
    model_used: str = ""
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))


class RiskDecision(BaseModel):
    """Risk engine decision."""
    allow: bool
    reason: str = ""
    adjusted_signal: Optional[Signal] = None
    max_size: float = 0.0


class ExecutionResult(BaseModel):
    """Order execution result."""
    success: bool
    reason: str = ""
    order_id: str = ""
    symbol: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0


class Trade(BaseModel):
    """Completed trade for memory."""
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    entry_time: int
    exit_time: int
    regime: Optional[RegimeState] = None
    signal_confidence: float = 0.0
    model_used: str = ""
    reasoning: str = ""
    
    def to_meta(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "regime": self.regime.value if self.regime else None,
            "signal_confidence": self.signal_confidence,
            "model_used": self.model_used,
        }


class AccountState(BaseModel):
    """Current account state for risk checks."""
    equity: float
    balance: float
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    open_positions: int = 0
    loss_streak: int = 0
    max_drawdown_pct: float = 0.0


class OrderRequest(BaseModel):
    """Order request parameters."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    post_only: bool = True


class SystemState(BaseModel):
    """Complete system state summary."""
    active_candle: Optional[Candle] = None
    regime: RegimeState = RegimeState.TRANSITION
    regime_confidence: float = 0.0
    key_levels: Dict[str, List[float]] = Field(default_factory=dict)
    session_vwap: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    session_bias: str = "neutral"
    strategy_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    buffer_len: int = 0
    last_signal: Optional[Signal] = None
    last_execution: Optional[Dict] = None