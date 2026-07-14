"""
Feature Engine Service — Technical indicator computation from candle streams.

Provides:
- Real-time incremental indicator calculation
- Multiple timeframe support
- Vectorized batch computation for historical data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from services.market_data import Candle, create_market_data_service


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FeatureVector:
    """Computed feature vector for a single candle."""
    symbol: str
    timestamp: datetime
    timeframe: str

    # Price features
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Trend
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0

    # Momentum
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # Volatility
    atr_14: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.5

    # Volume
    vol_sma_20: float = 0.0
    vol_ratio: float = 1.0
    obv: float = 0.0
    vwap: float = 0.0

    # Custom
    price_change_pct: float = 0.0
    high_low_range: float = 0.0
    close_open_pct: float = 0.0

    # Regime
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    # Microstructure
    bid_ask_spread: float = 0.0
    order_flow_imbalance: float = 0.0

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "open": self.open, "high": self.high, "low": self.low, "close": self.close, "volume": self.volume,
            "ema_9": self.ema_9, "ema_21": self.ema_21, "ema_50": self.ema_50, "ema_200": self.ema_200,
            "sma_20": self.sma_20, "sma_50": self.sma_50,
            "rsi_14": self.rsi_14, "rsi_7": self.rsi_7,
            "macd": self.macd, "macd_signal": self.macd_signal, "macd_hist": self.macd_hist,
            "stoch_k": self.stoch_k, "stoch_d": self.stoch_d,
            "atr_14": self.atr_14,
            "bb_upper": self.bb_upper, "bb_middle": self.bb_middle, "bb_lower": self.bb_lower,
            "bb_width": self.bb_width, "bb_position": self.bb_position,
            "vol_sma_20": self.vol_sma_20, "vol_ratio": self.vol_ratio,
            "obv": self.obv, "vwap": self.vwap,
            "price_change_pct": self.price_change_pct,
            "high_low_range": self.high_low_range,
            "close_open_pct": self.close_open_pct,
            "adx": self.adx, "plus_di": self.plus_di, "minus_di": self.minus_di,
            "bid_ask_spread": self.bid_ask_spread,
            "order_flow_imbalance": self.order_flow_imbalance,
            "metadata": self.metadata,
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML inference."""
        return np.array([
            self.close, self.volume,
            self.ema_9, self.ema_21, self.ema_50, self.ema_200,
            self.sma_20, self.sma_50,
            self.rsi_14, self.rsi_7,
            self.macd, self.macd_signal, self.macd_hist,
            self.stoch_k, self.stoch_d,
            self.atr_14,
            self.bb_upper, self.bb_middle, self.bb_lower, self.bb_width, self.bb_position,
            self.vol_sma_20, self.vol_ratio, self.obv, self.vwap,
            self.price_change_pct, self.high_low_range, self.close_open_pct,
            self.adx, self.plus_di, self.minus_di,
            self.bid_ask_spread, self.order_flow_imbalance,
        ], dtype=np.float32)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # EMA periods
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50, 200])
    # SMA periods
    sma_periods: list[int] = field(default_factory=lambda: [20, 50])
    # RSI periods
    rsi_periods: list[int] = field(default_factory=lambda: [7, 14])
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # ATR
    atr_period: int = 14
    # Volume
    vol_sma_period: int = 20
    # ADX
    adx_period: int = 14


# ═══════════════════════════════════════════════════════════════════
# INCREMENTAL CALCULATOR (O(1) per tick)
# ═══════════════════════════════════════════════════════════════════

class IncrementalCalculator:
    """
    O(1) incremental indicator updates.
    Maintains state for each symbol/timeframe.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.state: dict[str, dict] = {}  # key: f"{symbol}:{timeframe}"

    def _get_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def _init_state(self, key: str):
        """Initialize state for new symbol/timeframe."""
        if key not in self.state:
            self.state[key] = {
                "candles": deque(maxlen=500),
                "closes": deque(maxlen=500),
                "highs": deque(maxlen=500),
                "lows": deque(maxlen=500),
                "volumes": deque(maxlen=500),
                # EMA state
                "ema": {p: None for p in self.config.ema_periods},
                # RSI state
                "rsi_gains": {p: deque(maxlen=p) for p in self.config.rsi_periods},
                "rsi_losses": {p: deque(maxlen=p) for p in self.config.rsi_periods},
                # MACD state
                "macd_ema_fast": None,
                "macd_ema_slow": None,
                "macd_signal_ema": None,
                # Stochastic
                "stoch_highs": deque(maxlen=self.config.stoch_k),
                "stoch_lows": deque(maxlen=self.config.stoch_k),
                # Bollinger Bands
                "bb_closes": deque(maxlen=self.config.bb_period),
                # ATR
                "atr_tr": deque(maxlen=self.config.atr_period),
                "prev_close": None,
                # Volume
                "vol_volumes": deque(maxlen=self.config.vol_sma_period),
                # ADX
                "adx_plus_dm": deque(maxlen=self.config.adx_period),
                "adx_minus_dm": deque(maxlen=self.config.adx_period),
                "adx_tr": deque(maxlen=self.config.adx_period),
                "adx_dx": deque(maxlen=self.config.adx_period),
                # OBV
                "obv": 0.0,
                # VWAP
                "vwap_pv_sum": 0.0,
                "vwap_vol_sum": 0.0,
            }

    def update(self, candle: Candle) -> FeatureVector:
        """Update indicators with new candle, return feature vector."""
        key = self._get_key(candle.symbol, candle.timeframe)
        self._init_state(key)
        s = self.state[key]

        # Store candle data
        s["candles"].append(candle)
        s["closes"].append(candle.close)
        s["highs"].append(candle.high)
        s["lows"].append(candle.low)
        s["volumes"].append(candle.volume)

        # Calculate all indicators
        self._update_emas(s, candle.close)
        self._update_rsi(s, candle.close)
        self._update_macd(s, candle.close)
        self._update_stochastic(s, candle.high, candle.low, candle.close)
        self._update_bollinger(s, candle.close)
        self._update_atr(s, candle.high, candle.low, candle.close)
        self._update_volume(s, candle.volume)
        self._update_adx(s, candle.high, candle.low, candle.close)
        self._update_obv(s, candle.close, candle.volume)
        self._update_vwap(s, candle.high, candle.low, candle.close, candle.volume)

        return self._build_feature_vector(candle, s)

    def _update_emas(self, s: dict, close: float):
        for period in self.config.ema_periods:
            alpha = 2.0 / (period + 1)
            if s["ema"][period] is None:
                s["ema"][period] = close
            else:
                s["ema"][period] = alpha * close + (1 - alpha) * s["ema"][period]

    def _update_rsi(self, s: dict, close: float):
        if s["prev_close"] is not None:
            change = close - s["prev_close"]
            for period in self.config.rsi_periods:
                if change > 0:
                    s["rsi_gains"][period].append(change)
                    s["rsi_losses"][period].append(0.0)
                else:
                    s["rsi_gains"][period].append(0.0)
                    s["rsi_losses"][period].append(abs(change))
        s["prev_close"] = close

    def _update_macd(self, s: dict, close: float):
        alpha_fast = 2.0 / (self.config.macd_fast + 1)
        alpha_slow = 2.0 / (self.config.macd_slow + 1)
        alpha_signal = 2.0 / (self.config.macd_signal + 1)

        if s["macd_ema_fast"] is None:
            s["macd_ema_fast"] = close
            s["macd_ema_slow"] = close
            s["macd_signal_ema"] = 0.0
        else:
            s["macd_ema_fast"] = alpha_fast * close + (1 - alpha_fast) * s["macd_ema_fast"]
            s["macd_ema_slow"] = alpha_slow * close + (1 - alpha_slow) * s["macd_ema_slow"]

        macd = s["macd_ema_fast"] - s["macd_ema_slow"]
        s["macd_signal_ema"] = alpha_signal * macd + (1 - alpha_signal) * s["macd_signal_ema"]

        s["macd"] = macd
        s["macd_signal"] = s["macd_signal_ema"]
        s["macd_hist"] = macd - s["macd_signal_ema"]

    def _update_stochastic(self, s: dict, high: float, low: float, close: float):
        s["stoch_highs"].append(high)
        s["stoch_lows"].append(low)

        if len(s["stoch_highs"]) >= self.config.stoch_k:
            highest = max(s["stoch_highs"])
            lowest = min(s["stoch_lows"])
            if highest != lowest:
                s["stoch_k"] = 100 * (close - lowest) / (highest - lowest)
            else:
                s["stoch_k"] = 50.0

    def _update_bollinger(self, s: dict, close: float):
        s["bb_closes"].append(close)
        if len(s["bb_closes"]) >= self.config.bb_period:
            mean = np.mean(s["bb_closes"])
            std = np.std(s["bb_closes"])
            s["bb_middle"] = mean
            s["bb_upper"] = mean + self.config.bb_std * std
            s["bb_lower"] = mean - self.config.bb_std * std
            if s["bb_upper"] != s["bb_lower"]:
                s["bb_position"] = (close - s["bb_lower"]) / (s["bb_upper"] - s["bb_lower"])
                s["bb_width"] = (s["bb_upper"] - s["bb_lower"]) / mean
            else:
                s["bb_position"] = 0.5
                s["bb_width"] = 0.0

    def _update_atr(self, s: dict, high: float, low: float, close: float):
        if s["prev_close"] is not None:
            tr = max(
                high - low,
                abs(high - s["prev_close"]),
                abs(low - s["prev_close"]),
            )
            s["atr_tr"].append(tr)
            if len(s["atr_tr"]) >= self.config.atr_period:
                s["atr_14"] = np.mean(s["atr_tr"])
            else:
                s["atr_14"] = np.mean(s["atr_tr"])

    def _update_volume(self, s: dict, volume: float):
        s["vol_volumes"].append(volume)
        if len(s["vol_volumes"]) >= self.config.vol_sma_period:
            s["vol_sma"] = np.mean(s["vol_volumes"])
            s["vol_ratio"] = volume / s["vol_sma"] if s["vol_sma"] > 0 else 1.0
        else:
            s["vol_sma"] = np.mean(s["vol_volumes"])
            s["vol_ratio"] = 1.0

    def _update_adx(self, s: dict, high: float, low: float, close: float):
        if s["prev_close"] is not None:
            up_move = high - s.get("prev_high", high)
            down_move = s.get("prev_low", low) - low

            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0

            tr = max(
                high - low,
                abs(high - s["prev_close"]),
                abs(low - s["prev_close"]),
            )

            s["adx_plus_dm"].append(plus_dm)
            s["adx_minus_dm"].append(minus_dm)
            s["adx_tr"].append(tr)

            if len(s["adx_tr"]) >= self.config.adx_period:
                # Wilder's smoothing
                atr = np.mean(s["adx_tr"])
                plus_di = 100 * np.mean(s["adx_plus_dm"]) / atr if atr > 0 else 0
                minus_di = 100 * np.mean(s["adx_minus_dm"]) / atr if atr > 0 else 0

                s["plus_di"] = plus_di
                s["minus_di"] = minus_di

                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
                s["adx_dx"].append(dx)

                if len(s["adx_dx"]) >= self.config.adx_period:
                    s["adx"] = np.mean(s["adx_dx"])

        s["prev_high"] = high
        s["prev_low"] = low

    def _update_obv(self, s: dict, close: float, volume: float):
        if s["prev_close"] is not None:
            if close > s["prev_close"]:
                s["obv"] += volume
            elif close < s["prev_close"]:
                s["obv"] -= volume
        # else: first candle, obv stays 0

    def _update_vwap(self, s: dict, high: float, low: float, close: float, volume: float):
        typical_price = (high + low + close) / 3
        s["vwap_pv_sum"] += typical_price * volume
        s["vwap_vol_sum"] += volume
        s["vwap"] = s["vwap_pv_sum"] / s["vwap_vol_sum"] if s["vwap_vol_sum"] > 0 else close

    def _build_feature_vector(self, candle: Candle, s: dict) -> FeatureVector:
        fv = FeatureVector(
            symbol=candle.symbol,
            timestamp=candle.timestamp,
            timeframe=candle.timeframe,
            open=candle.open, high=candle.high, low=candle.low, close=candle.close, volume=candle.volume,
        )

        # EMA
        fv.ema_9 = s["ema"].get(9, 0)
        fv.ema_21 = s["ema"].get(21, 0)
        fv.ema_50 = s["ema"].get(50, 0)
        fv.ema_200 = s["ema"].get(200, 0)

        # SMA (compute from closes deque)
        if len(s["closes"]) >= 20:
            fv.sma_20 = np.mean(list(s["closes"])[-20:])
        if len(s["closes"]) >= 50:
            fv.sma_50 = np.mean(list(s["closes"])[-50:])

        # RSI
        for period in self.config.rsi_periods:
            gains = list(s["rsi_gains"][period])
            losses = list(s["rsi_losses"][period])
            if gains and losses:
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 1e-10
                rs = avg_gain / avg_loss
                rsi = 100 - 100 / (1 + rs)
                if period == 14:
                    fv.rsi_14 = rsi
                elif period == 7:
                    fv.rsi_7 = rsi

        # MACD
        fv.macd = s.get("macd", 0)
        fv.macd_signal = s.get("macd_signal", 0)
        fv.macd_hist = s.get("macd_hist", 0)

        # Stochastic
        fv.stoch_k = s.get("stoch_k", 50)
        fv.stoch_d = s.get("stoch_d", 50)  # Would need another EMA

        # ATR
        fv.atr_14 = s.get("atr_14", 0)

        # Bollinger
        fv.bb_upper = s.get("bb_upper", 0)
        fv.bb_middle = s.get("bb_middle", 0)
        fv.bb_lower = s.get("bb_lower", 0)
        fv.bb_width = s.get("bb_width", 0)
        fv.bb_position = s.get("bb_position", 0.5)

        # Volume
        fv.vol_sma_20 = s.get("vol_sma", 0)
        fv.vol_ratio = s.get("vol_ratio", 1.0)
        fv.obv = s.get("obv", 0)
        fv.vwap = s.get("vwap", candle.close)

        # Price action
        fv.price_change_pct = (candle.close - candle.open) / candle.open if candle.open > 0 else 0
        fv.high_low_range = (candle.high - candle.low) / candle.close if candle.close > 0 else 0
        fv.close_open_pct = (candle.close - candle.open) / candle.open if candle.open > 0 else 0

        # ADX
        fv.adx = s.get("adx", 0)
        fv.plus_di = s.get("plus_di", 0)
        fv.minus_di = s.get("minus_di", 0)

        return fv

    def compute_batch(self, symbol: str, timeframe: str, candles: list[Candle]) -> list[FeatureVector]:
        """Compute features for historical batch (vectorized)."""
        results = []
        for candle in candles:
            results.append(self.update(candle))
        return results


# ═══════════════════════════════════════════════════════════════════
# VECTORIZED BATCH CALCULATOR (for historical data)
# ═══════════════════════════════════════════════════════════════════

def compute_features_vectorized(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Vectorized feature computation on pandas DataFrame.
    Much faster for historical batch processing.
    """
    cfg = config or FeatureConfig()
    df = df.copy()

    # Ensure required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # EMA
    for period in cfg.ema_periods:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # SMA
    for period in cfg.sma_periods:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()

    # RSI
    for period in cfg.rsi_periods:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # MACD
    ema_fast = df["close"].ewm(span=cfg.macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=cfg.macd_slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=cfg.macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic
    low_min = df["low"].rolling(cfg.stoch_k).min()
    high_max = df["high"].rolling(cfg.stoch_k).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(cfg.stoch_d).mean()

    # Bollinger Bands
    bb_mid = df["close"].rolling(cfg.bb_period).mean()
    bb_std = df["close"].rolling(cfg.bb_period).std()
    df["bb_upper"] = bb_mid + cfg.bb_std * bb_std
    df["bb_lower"] = bb_mid - cfg.bb_std * bb_std
    df["bb_middle"] = bb_mid
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(cfg.atr_period).mean()

    # Volume
    df["vol_sma_20"] = df["volume"].rolling(cfg.vol_sma_period).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, 1)

    # OBV
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv

    # VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()

    # Price action
    df["price_change_pct"] = df["close"].pct_change()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]

    # ADX
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(cfg.adx_period).mean()

    plus_di = 100 * pd.Series(plus_dm).rolling(cfg.adx_period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(cfg.adx_period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.rolling(cfg.adx_period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    return df


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINE SERVICE
# ═══════════════════════════════════════════════════════════════════

class FeatureEngineService:
    """
    Service that connects market data stream to feature computation.
    Publishes FeatureVector events.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.calculator = IncrementalCalculator(
            FeatureConfig() if "features" not in self.config else FeatureConfig(**self.config["features"])
        )
        self.market_service = create_market_data_service(self.config.get("market_data", {}))
        self._feature_callbacks: list[Callable] = []

        # Wire up market data callbacks
        self.market_service.on_candle(self._on_candle)

    def on_features(self, callback: Callable[[FeatureVector], None]):
        """Register callback for feature vectors."""
        self._feature_callbacks.append(callback)

    async def _on_candle(self, candle: Candle):
        """Process incoming candle and emit features."""
        fv = self.calculator.update(candle)
        for cb in self._feature_callbacks:
            try:
                await cb(fv)
            except Exception as e:
                print(f"[FeatureEngine] Callback error: {e}")

    async def start(self, symbols: list[str], timeframes: list[str] = None):
        """Start the service."""
        await self.market_service.start(symbols, timeframes)

    async def stop(self):
        await self.market_service.stop()

    def get_features(self, symbol: str, timeframe: str) -> FeatureVector | None:
        """Get latest computed features for symbol/timeframe."""
        # Would need to store last computed vector per symbol/timeframe
        # For now, returns None - implement caching if needed
        return None


def create_feature_engine_service(config: dict | None = None) -> FeatureEngineService:
    """Factory for FeatureEngineService."""
    return FeatureEngineService(config)


if __name__ == "__main__":
    import asyncio

    async def test():
        config = {
            "market_data": {"exchanges": ["binance"], "binance": {"testnet": True}},
            "features": {},
        }
        service = create_feature_engine_service(config)

        async def on_fv(fv: FeatureVector):
            print(f"{fv.symbol} {fv.timeframe} close={fv.close:.2f} rsi={fv.rsi_14:.1f} macd={fv.macd:.4f} bb_pos={fv.bb_position:.2f}")

        service.on_features(on_fv)
        await service.start(["BTCUSDT", "ETHUSDT"], ["1m"])
        await asyncio.sleep(60)
        await service.stop()

    asyncio.run(test())