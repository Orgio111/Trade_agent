"""Incremental Candle State Management - No full history recomputation."""

from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np
from models import Candle, CandleFeatures, RegimeState
import yaml
from pathlib import Path


class IncrementalState:
    """
    4-Level Incremental State Architecture:
    
    Level 0: Active Candle (updated every 60s)
    Level 1: Rolling Buffer (200 candles, fixed size)
    Level 2: Key Levels (persistent, event-driven updates)
    Level 3: Session Context (daily reset)
    Level 4: Strategy Performance (per-trade update)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.buffer_size = self.config.get("memory", {}).get("buffer_size", 200)
        
        # Level 0: Active candle
        self.active_candle: Optional[Candle] = None
        self.active_features: Optional[CandleFeatures] = None
        
        # Level 1: Rolling buffer (compressed features)
        self.candle_buffer: deque = deque(maxlen=self.buffer_size)
        self.feature_buffer: deque = deque(maxlen=self.buffer_size)
        
        # Regime detection (streaming HMM)
        self.regime_state: RegimeState = RegimeState.TRANSITION
        self.regime_history: deque = deque(maxlen=50)
        self.regime_confidence: float = 0.0
        
        # Level 2: Key levels (persistent, updated on break/retest)
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.volume_nodes: List[Dict[str, Any]] = []  # {price, volume, timestamp}
        
        # Level 3: Session context (daily reset)
        self.session_vwap: float = 0.0
        self.session_high: float = 0.0
        self.session_low: float = float('inf')
        self.session_open: float = 0.0
        self.session_volume: float = 0.0
        self.session_bias: str = "neutral"
        self.day_start_ts: int = 0
        
        # Level 4: Strategy performance (per-trade update)
        self.strategy_stats: Dict[str, Dict[str, float]] = {
            "scalp": {"wins": 0, "losses": 0, "avg_r": 0.0, "total_r": 0.0},
            "swing": {"wins": 0, "losses": 0, "avg_r": 0.0, "total_r": 0.0},
        }
        
        # Technical indicator state (for incremental computation)
        self._prev_closes: deque = deque(maxlen=60)
        self._prev_highs: deque = deque(maxlen=60)
        self._prev_lows: deque = deque(maxlen=60)
        self._prev_volumes: deque = deque(maxlen=60)
        
        # EMA state
        self._ema_9: Optional[float] = None
        self._ema_21: Optional[float] = None
        self._ema_50: Optional[float] = None
        
        # OBV state
        self._obv: float = 0.0
        self._prev_close: Optional[float] = None
        
        # RSI state
        self._gains: deque = deque(maxlen=14)
        self._losses: deque = deque(maxlen=14)
        
    def on_new_candle(self, candle: Candle) -> CandleFeatures:
        """
        Process new 1-minute candle.
        Updates all 4 levels incrementally.
        Returns computed features for model inference.
        """
        # Check for new day
        self._check_new_day(candle)
        
        # Update active candle
        self.active_candle = candle
        
        # Update rolling buffers
        self.candle_buffer.append(candle)
        self._prev_closes.append(candle.close)
        self._prev_highs.append(candle.high)
        self._prev_lows.append(candle.low)
        self._prev_volumes.append(candle.volume)
        
        # Update session stats
        self._update_session(candle)
        
        # Compute incremental features
        features = self._compute_features_incremental(candle)
        self.active_features = features
        self.feature_buffer.append(features)
        
        # Update regime (streaming HMM)
        if len(self.feature_buffer) >= 20:
            self._update_regime()
        
        # Update key levels
        self._update_key_levels(candle, features)
        
        # Update EMAs
        self._update_emas(candle.close)
        
        return features
    
    def _check_new_day(self, candle: Candle):
        """Reset session state on new day."""
        candle_date = candle.timestamp // 86400000  # days since epoch
        if self.day_start_ts != candle_date:
            self.session_vwap = 0.0
            self.session_high = 0.0
            self.session_low = float('inf')
            self.session_open = candle.open
            self.session_volume = 0.0
            self.session_bias = "neutral"
            self.day_start_ts = candle_date
    
    def _update_session(self, candle: Candle):
        """Update session-level statistics."""
        # VWAP update
        if self.session_volume == 0:
            self.session_vwap = candle.typical_price
        else:
            self.session_vwap = (
                (self.session_vwap * self.session_volume) + candle.vwap
            ) / (self.session_volume + candle.volume)
        
        self.session_volume += candle.volume
        self.session_high = max(self.session_high, candle.high)
        self.session_low = min(self.session_low, candle.low)
        
        # Session bias
        if self.session_vwap > 0:
            dev = (candle.close - self.session_vwap) / self.session_vwap
            if dev > 0.01:
                self.session_bias = "bullish"
            elif dev < -0.01:
                self.session_bias = "bearish"
            else:
                self.session_bias = "neutral"
    
    def _compute_features_incremental(self, candle: Candle) -> CandleFeatures:
        """Compute features using only incremental state."""
        features = CandleFeatures()
        
        # Need minimum history
        if len(self._prev_closes) < 2:
            return features
        
        closes = list(self._prev_closes)
        highs = list(self._prev_highs)
        lows = list(self._prev_lows)
        volumes = list(self._prev_volumes)
        
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_volume = volumes[-1]
        
        # Returns
        if len(closes) >= 2:
            features.returns_1 = (closes[-1] - closes[-2]) / closes[-2]
        if len(closes) >= 6:
            features.returns_5 = (closes[-1] - closes[-6]) / closes[-6]
        if len(closes) >= 16:
            features.returns_15 = (closes[-1] - closes[-16]) / closes[-16]
        if len(closes) >= 61:
            features.returns_60 = (closes[-1] - closes[-61]) / closes[-61]
        
        # ATR (14-period)
        if len(highs) >= 15:
            tr_values = []
            for i in range(-14, 0):
                h = highs[i]
                l = lows[i]
                pc = closes[i-1]
                tr = max(h - l, abs(h - pc), abs(l - pc))
                tr_values.append(tr)
            features.atr_14 = np.mean(tr_values)
            features.atr_pct = features.atr_14 / current_close
        
        # Bollinger Bands width (20-period)
        if len(closes) >= 20:
            recent = closes[-20:]
            sma = np.mean(recent)
            std = np.std(recent)
            features.bb_width = (2 * std) / sma if sma > 0 else 0
        
        # RSI (14-period) - incremental
        if len(closes) >= 2:
            change = closes[-1] - closes[-2]
            self._gains.append(max(change, 0))
            self._losses.append(max(-change, 0))
            if len(self._gains) == 14:
                avg_gain = np.mean(self._gains)
                avg_loss = np.mean(self._losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features.rsi_14 = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9) - using EMA state
        if self._ema_12 is not None and self._ema_26 is not None:
            features.macd = self._ema_12 - self._ema_26
            # MACD signal line (9-period EMA of MACD)
            if not hasattr(self, '_macd_signal'):
                self._macd_signal = features.macd
            else:
                self._macd_signal = 0.2 * features.macd + 0.8 * self._macd_signal
            features.macd_signal = self._macd_signal
            features.macd_hist = features.macd - features.macd_signal
        
        # Volume ratio
        if len(volumes) >= 21:
            recent_vol = np.mean(volumes[-5:])
            prev_vol = np.mean(volumes[-20:-5])
            features.volume_ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0
        
        # OBV
        if self._prev_close is not None:
            if current_close > self._prev_close:
                self._obv += current_volume
            elif current_close < self._prev_close:
                self._obv -= current_volume
        features.obv = self._obv
        self._prev_close = current_close
        
        # VWAP deviation
        if self.session_vwap > 0:
            features.vwap_dev = (current_close - self.session_vwap) / self.session_vwap
        
        # EMAs (already computed)
        features.ema_9 = self._ema_9 or 0.0
        features.ema_21 = self._ema_21 or 0.0
        features.ema_50 = self._ema_50 or 0.0
        
        # ADX (simplified)
        if len(highs) >= 14 and len(lows) >= 14:
            features.adx = self._compute_adx(highs[-14:], lows[-14:], closes[-14:])
        
        # Market structure
        if len(highs) >= 3 and len(lows) >= 3:
            features.higher_high = highs[-1] > highs[-2] and highs[-2] > highs[-3]
            features.higher_low = lows[-1] > lows[-2] and lows[-2] > lows[-3]
            features.lower_high = highs[-1] < highs[-2] and highs[-2] < highs[-3]
            features.lower_low = lows[-1] < lows[-2] and lows[-2] < lows[-3]
        
        return features
    
    def _update_emas(self, close: float):
        """Update exponential moving averages incrementally."""
        alpha_9 = 2 / (9 + 1)
        alpha_21 = 2 / (21 + 1)
        alpha_50 = 2 / (50 + 1)
        
        if self._ema_9 is None:
            self._ema_9 = close
            self._ema_21 = close
            self._ema_50 = close
        else:
            self._ema_9 = alpha_9 * close + (1 - alpha_9) * self._ema_9
            self._ema_21 = alpha_21 * close + (1 - alpha_21) * self._ema_21
            self._ema_50 = alpha_50 * close + (1 - alpha_50) * self._ema_50
        
        # MACD EMAs
        alpha_12 = 2 / (12 + 1)
        alpha_26 = 2 / (26 + 1)
        if not hasattr(self, '_ema_12'):
            self._ema_12 = close
            self._ema_26 = close
        else:
            self._ema_12 = alpha_12 * close + (1 - alpha_12) * self._ema_12
            self._ema_26 = alpha_26 * close + (1 - alpha_26) * self._ema_26
    
    def _compute_adx(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """Simplified ADX computation."""
        if len(highs) < 2:
            return 0.0
        
        dm_plus = []
        dm_minus = []
        tr_values = []
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            dm_plus.append(max(up_move, 0) if up_move > down_move else 0)
            dm_minus.append(max(down_move, 0) if down_move > up_move else 0)
            
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        if not tr_values:
            return 0.0
        
        atr = np.mean(tr_values)
        if atr == 0:
            return 0.0
        
        di_plus = 100 * np.mean(dm_plus) / atr
        di_minus = 100 * np.mean(dm_minus) / atr
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
        return dx
    
    def _update_regime(self):
        """
        Streaming HMM-style regime detection.
        5 states: trend_up, trend_down, range, volatile, transition
        """
        # Simple rule-based regime for now
        # Can be replaced with actual HMM later
        
        if len(self.feature_buffer) < 20:
            return
        
        recent = list(self.feature_buffer)[-20:]
        
        # Trend strength
        returns = [f.returns_15 for f in recent if f.returns_15 != 0]
        avg_return = np.mean(returns) if returns else 0
        return_std = np.std(returns) if len(returns) > 1 else 0
        
        # Volatility
        atr_pcts = [f.atr_pct for f in recent if f.atr_pct > 0]
        avg_atr = np.mean(atr_pcts) if atr_pcts else 0
        
        # ADX trend strength
        adx_vals = [f.adx for f in recent if f.adx > 0]
        avg_adx = np.mean(adx_vals) if adx_vals else 0
        
        # RSI extremes
        rsi_vals = [f.rsi_14 for f in recent if f.rsi_14 > 0]
        avg_rsi = np.mean(rsi_vals) if rsi_vals else 50
        
        # Regime logic
        if avg_adx_adx > 0.02 and avg_adx > 25:
            self.regime_state = RegimeState.TREND_UP
            self.regime_confidence = min(avg_adx / 50, 1.0)
        elif avg_return < -0.02 and avg_adx > 25:
            self.regime_state = RegimeState.TREND_DOWN
            self.regime_confidence = min(avg_adx / 50, 1.0)
        elif avg_atr > 0.03 or avg_adx < 15:
            self.regime_state = RegimeState.VOLATILE
            self.regime_confidence = min(avg_atr * 10, 1.0)
        elif avg_adx < 20 and avg_atr < 0.015:
            self.regime_state = RegimeState.RANGE
            self.regime_confidence = 1.0 - min(avg_adx / 20, 1.0)
        else:
            self.regime_state = RegimeState.TRANSITION
            self.regime_confidence = 0.5
        
        self.regime_history.append({
            "state": self.regime_state,
            "confidence": self.regime_confidence,
            "avg_return": avg_return,
            "avg_adx": avg_adx,
            "avg_atr": avg_atr
        })
    
    def _update_key_levels(self, candle: Candle, features: CandleFeatures):
        """Update support/resistance on break or retest."""
        current_price = candle.close
        
        # Check resistance retest
        for level in self.resistance_levels[:]:
            if abs(current_price - level) / level < 0.002:  # Within 0.2%
                # Retest - strengthen or break
                if candle.high > level * 1.005:
                    # Breakout - move to support
                    self.resistance_levels.remove(level)
                    self.support_levels.append(level)
                    self.support_levels = sorted(set(self.support_levels))[-10:]
        
        # Check support retest
        for level in self.support_levels[:]:
            if abs(current_price - level) / level < 0.002:
                if candle.low < level * 0.995:
                    # Breakdown - move to resistance
                    self.support_levels.remove(level)
                    self.resistance_levels.append(level)
                    self.resistance_levels = sorted(set(self.resistance_levels))[-10:]
        
        # Add new levels from volume nodes
        if features.volume_ratio > 2.0:
            # High volume area - potential support/resistance
            node_price = candle.typical_price
            self.volume_nodes.append({
                "price": node_price,
                "volume": candle.volume,
                "timestamp": candle.timestamp
            })
            # Keep last 20 volume nodes
            self.volume_nodes = self.volume_nodes[-20:]
    
    def update_trade_result(self, trade_result: Dict):
        """Update strategy performance stats (Level 4)."""
        strategy = trade_result.get("strategy", "scalp")
        r_multiple = trade_result.get("r_multiple", 0)
        win = r_multiple > 0
        
        if strategy in self.strategy_stats:
            stats = self.strategy_stats[strategy]
            if win:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            
            total = stats["wins"] + stats["losses"]
            stats["total_r"] += r_multiple
            stats["avg_r"] = stats["total_r"] / total if total > 0 else 0
    
    def get_context_for_model(self, mode: str = "normal") -> Dict[str, Any]:
        """
        Get minimal context needed by model.
        Does NOT include full history - only compressed state.
        """
        base = {
            "current_candle": self.active_candle.to_dict() if self.active_candle else {},
            "features": self.active_features.to_list() if self.active_features else [],
            "regime": self.regime_state.value,
            "regime_confidence": self.regime_confidence,
            "key_levels": {
                "support": self.support_levels[-5:],
                "resistance": self.resistance_levels[-5:],
            },
            "session": {
                "vwap": self.session_vwap,
                "high": self.session_high,
                "low": self.session_low,
                "bias": self.session_bias,
            },
            "strategy_stats": self.strategy_stats,
            "buffer_len": len(self.candle_buffer),
        }
        
        if mode == "deep":
            base["recent_candles"] = [
                c.to_dict() for c in list(self.candle_buffer)[-20:]
            ]
            base["regime_history"] = list(self.regime_history)[-10:]
        
        return base
    
    def summary(self) -> Dict[str, Any]:
        """Human-readable state summary."""
        return {
            "candle_buffer": len(self.candle_buffer),
            "regime": self.regime_state.value,
            "regime_confidence": round(self.regime_confidence, 3),
            "support": [round(x, 2) for x in self.support_levels[-3:]],
            "resistance": [round(x, 2) for x in self.resistance_levels[-3:]],
            "session_vwap": round(self.session_vwap, 2),
            "session_bias": self.session_bias,
            "strategy_stats": self.strategy_stats,
        }


# Backward compatibility
def create_incremental_state(config_path: str = "config.yaml") -> IncrementalState:
    return IncrementalState(config_path)