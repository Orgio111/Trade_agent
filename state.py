"""Incremental Candle State - Core candle linking system with compressed memory."""

from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np
import yaml
from pathlib import Path
from models import Candle, CandleFeatures, RegimeState


class IncrementalState:
    """
    4-Level Incremental State Architecture (NO full history recomputation):
    
    Level 0: Active Candle (updated every 60s)
    Level 1: Rolling Buffer (200 candles, fixed size)
    Level 2: Key Levels (persistent, event-driven)
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
        
        # Level 1: Rolling buffers
        self.candle_buffer: deque = deque(maxlen=self.buffer_size)
        self.feature_buffer: deque = deque(maxlen=self.buffer_size)
        
        # Regime detection
        self.regime_state: RegimeState = RegimeState.TRANSITION
        self.regime_confidence: float = 0.0
        self.regime_history: deque = deque(maxlen=50)
        
        # Level 2: Key levels
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.volume_nodes: List[Dict[str, Any]] = []
        
        # Level 3: Session context
        self.session_vwap: float = 0.0
        self.session_high: float = 0.0
        self.session_low: float = float('inf')
        self.session_open: float = 0.0
        self.session_volume: float = 0.0
        self.session_bias: str = "neutral"
        self.day_start_ts: int = 0
        
        # Level 4: Strategy performance
        self.strategy_stats: Dict[str, Dict[str, float]] = {
            "scalp": {"wins": 0, "losses": 0, "avg_r": 0.0, "total_r": 0.0},
            "swing": {"wins": 0, "losses": 0, "avg_r": 0.0, "total_r": 0.0},
        }
        
        # Technical indicator state
        self._prev_closes: deque = deque(maxlen=60)
        self._prev_highs: deque = deque(maxlen=60)
        self._prev_lows: deque = deque(maxlen=60)
        self._prev_volumes: deque = deque(maxlen=60)
        self._ema_9: Optional[float] = None
        self._ema_21: Optional[float] = None
        self._ema_50: Optional[float] = None
        self._ema_12: Optional[float] = None
        self._ema_26: Optional[float] = None
        self._macd_signal: Optional[float] = None
        self._obv: float = 0.0
        self._prev_close: Optional[float] = None
        self._gains: deque = deque(maxlen=14)
        self._losses: deque = deque(maxlen=14)
    
    def on_new_candle(self, candle: Candle) -> CandleFeatures:
        """Process new candle - updates all 4 levels incrementally."""
        self._check_new_day(candle)
        self.active_candle = candle
        
        # Update buffers
        self.candle_buffer.append(candle)
        self._prev_closes.append(candle.close)
        self._prev_highs.append(candle.high)
        self._prev_lows.append(candle.low)
        self._prev_volumes.append(candle.volume)
        
        self._update_session(candle)
        
        # Compute features incrementally
        features = self._compute_features_incremental(candle)
        self.active_features = features
        self.feature_buffer.append(features)
        
        # Update regime
        if len(self.feature_buffer) >= 20:
            self._update_regime()
        
        # Update key levels
        self._update_key_levels(candle)
        
        # Update EMAs
        self._update_emas(candle.close)
        
        return features
    
    def _check_new_day(self, candle: Candle):
        """Reset session on new day."""
        candle_date = candle.timestamp // 86400000
        if self.day_start_ts != candle_date:
            self.session_vwap = 0.0
            self.session_high = 0.0
            self.session_low = float('inf')
            self.session_open = candle.open
            self.session_volume = 0.0
            self.session_bias = "neutral"
            self.day_start_ts = candle.timestamp // 86400000
    
    def _update_session(self, candle: Candle):
        """Update session VWAP and bias."""
        if self.session_volume == 0:
            self.session_vwap = candle.typical_price
        else:
            self.session_vwap = (
                (self.session_vwap * self.session_volume) + candle.vwap_component
            ) / (self.session_volume + candle.volume)
        
        self.session_volume += candle.volume
        self.session_high = max(self.session_high, candle.high)
        self.session_low = min(self.session_low, candle.low)
        
        if self.session_vwap > 0:
            dev = (candle.close - self.session_vwap) / self.session_vwap
            if dev > 0.01:
                self.session_bias = "bullish"
            elif dev < -0.01:
                self.session_bias = "bearish"
            else:
                self.session_bias = "neutral"
    
    def _compute_features_incremental(self, candle: Candle) -> CandleFeatures:
        """Compute all features using only incremental state."""
        f = CandleFeatures()
        
        if len(self._prev_closes) < 2:
            return f
        
        closes = list(self._prev_closes)
        highs = list(self._prev_highs)
        lows = list(self._prev_lows)
        volumes = list(self._prev_volumes)
        
        c = closes[-1]
        h = highs[-1]
        l = lows[-1]
        v = volumes[-1]
        
        # Returns
        if len(closes) >= 2:
            f.ret_1 = (closes[-1] - closes[-2]) / closes[-2]
        if len(closes) >= 6:
            f.ret_5 = (c - closes[-6]) / closes[-6]
        if len(closes) >= 16:
            f.ret_15 = (c - closes[-16]) / closes[-16]
        if len(closes) >= 61:
            f.ret_60 = (c - closes[-61]) / closes[-61]
        
        # RSI
        if len(closes) >= 2:
            chg = closes[-1] - closes[-2]
            self._gains.append(max(chg, 0))
            self._losses.append(max(-chg, 0))
            if len(self._gains) == 14:
                ag = np.mean(self._gains)
                al = np.mean(self._losses)
                if al > 0:
                    f.rsi_14 = 100 - (100 / (1 + ag/al))
                else:
                    f.rsi_14 = 100.0
        
        # EMAs & MACD
        if self._ema_12 is None:
            self._ema_12 = self._ema_26 = c
        else:
            self._ema_12 = 0.1538 * c + 0.8462 * self._ema_12
            self._ema_26 = 0.0741 * c + 0.9259 * self._ema_26
        f.ema_9 = self._ema_12
        f.ema_21 = self._ema_26
        
        if self._ema_12 is not None and self._ema_26 is not None:
            macd = self._ema_12 - self._ema_26
            f.macd = macd
            if self._macd_signal is None:
                self._macd_signal = macd
            else:
                self._macd_signal = 0.2 * macd + 0.8 * self._macd_signal
            f.macd_signal = self._macd_signal
            f.macd_hist = macd - self._macd_signal
        
        # Bollinger Bands
        if len(self._prev_closes) >= 20:
            recent = list(self._prev_closes)[-20:]
            sma = np.mean(recent)
            std = np.std(recent)
            f.bb_width = (4 * std) / sma if sma > 0 else 0
            upper = sma + 2 * std
            lower = sma - 2 * std
            f.bb_pct = (c - lower) / (upper - lower) if upper != lower else 0.5
        
        # ATR
        if len(highs) >= 2 and len(lows) >= 2 and len(closes) >= 2:
            tr = max(h - l, abs(h - closes[-2]), abs(l - closes[-2]))
            f.atr_14 = tr
            f.atr_pct = tr / c
        
        # RSI
        if len(closes) >= 2:
            chg = c - closes[-2]
            self._gains.append(max(chg, 0))
            self._losses.append(max(-chg, 0))
            if len(self._gains) == 14:
                ag = np.mean(self._gains)
                al = np.mean(self._losses)
                f.rsi_14 = 100 - (100 / (1 + ag/al)) if al > 0 else 100
        
        # EMAs
        if self._ema_9 is None:
            self._ema_9 = self._ema_21 = self._ema_50 = c
        else:
            self._ema_9 = 0.2 * c + 0.8 * self._ema_9
            self._ema_21 = 0.0909 * c + 0.9091 * self._ema_21
            self._ema_50 = 0.0392 * c + 0.9608 * self._ema_50
        f.ema_9 = self._ema_9
        f.ema_21 = self._ema_21
        f.ema_50 = self._ema_50
        
        # MACD
        if self._ema_12 is None:
            self._ema_12 = self._ema_26 = c
        else:
            self._ema_12 = 0.1538 * c + 0.8462 * self._ema_12
            self._ema_26 = 0.0741 * c + 0.9259 * self._ema_26
        f.macd = self._ema_12 - self._ema_26
        if self._macd_signal is None:
            self._macd_signal = f.macd
        else:
            self._macd_signal = 0.2 * f.macd + 0.8 * self._macd_signal
        f.macd_signal = self._macd_signal
        f.macd_hist = f.macd - f.macd_signal
        
        # Volume
        if len(self._prev_volumes) >= 21:
            recent = np.mean(list(self._prev_volumes)[-5:])
            prev = np.mean(list(self._prev_volumes)[-20:-5])
            f.volume_ratio = recent / prev if prev > 0 else 1.0
        
        # OBV
        if self._prev_close is not None:
            if c > self._prev_close:
                self._obv += v
            elif c < self._prev_close:
                self._obv -= v
        f.obv = self._obv
        self._prev_close = c
        
        # VWAP deviation
        if hasattr(self, 'session_vwap') and self.session_vwap > 0:
            f.vwap_dev = (c - self.session_vwap) / self.session_vwap
        
        # Market structure
        if len(highs) >= 3:
            f.higher_high = highs[-1] > highs[-2] > highs[-3]
            f.higher_low = lows[-1] > lows[-2] > lows[-3]
            f.lower_high = highs[-1] < highs[-2] < highs[-3]
            f.lower_low = lows[-1] < lows[-2] < lows[-3]
        return f
    
    def _update_emas(self, close: float):
        """Update all EMAs."""
        if self._ema_9 is None:
            self._ema_9 = self._ema_21 = self._ema_50 = close
        else:
            self._ema_9 = 0.2 * close + 0.8 * self._ema_9
            self._ema_21 = 0.0909 * close + 0.9091 * self._ema_21
            self._ema_50 = 0.0392 * close + 0.9608 * self._ema_50
        
        # MACD EMAs
        if self._ema_12 is None:
            self._ema_12 = self._ema_26 = close
        else:
            self._ema_12 = 0.1538 * close + 0.8462 * self._ema_12
            self._ema_26 = 0.0741 * close + 0.9259 * self._ema_26
    
    def _update_regime(self):
        """Streaming regime detection."""
        if len(self.feature_buffer) < 20:
            return
        
        recent = list(self.feature_buffer)[-20:]
        
        # Trend
        rets = [f.ret_15 for f in recent if f.ret_15 != 0]
        avg_ret = np.mean(rets) if rets else 0
        
        # Volatility
        atrs = [f.atr_pct for f in recent if f.atr_pct > 0]
        avg_atr = np.mean(atrs) if atrs else 0
        
        # ADX
        adx_vals = [f.adx for f in recent if f.adx > 0]
        avg_adx = np.mean(adx_vals) if adx_vals else 0
        
        # RSI
        rsi_vals = [f.rsi_14 for f in recent if f.rsi_14 > 0]
        avg_rsi = np.mean(rsi_vals) if rsi_vals else 50
        
        # Regime logic
        if avg_ret > 0.02 and avg_adx > 25:
            self.regime_state = RegimeState.TREND_UP
            self.regime_confidence = min(avg_adx / 50, 1.0)
        elif avg_ret < -0.02 and avg_adx > 25:
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
            "state": self.regime_state.value,
            "confidence": self.regime_confidence
        })
    
    def _update_key_levels(self, candle: Candle):
        """Update support/resistance on break/retest."""
        price = candle.close
        
        # Check resistance
        for level in self.resistance_levels[:]:
            if abs(price - level) / level < 0.002:
                if candle.high > level * 1.005:
                    self.resistance_levels.remove(level)
                    self.support_levels.append(level)
                    self.support_levels = sorted(set(self.support_levels))[-10:]
        
        # Check support
        for level in self.support_levels[:]:
            if abs(price - level) / level < 0.002:
                if candle.low < level * 0.995:
                    self.support_levels.remove(level)
                    self.resistance_levels.append(level)
                    self.resistance_levels = sorted(set(self.resistance_levels))[-10:]
        
        # Add from volume nodes
        if len(self.feature_buffer) > 0 and self.feature_buffer[-1].volume_ratio > 2.0:
            self.volume_nodes.append({"price": candle.typical_price, "volume": candle.volume, "ts": candle.timestamp})
            self.volume_nodes = self.volume_nodes[-20:]
    
    def update_trade_result(self, trade_result: Dict):
        """Level 4: Update strategy performance."""
        strategy = trade_result.get("strategy", "scalp")
        r = trade_result.get("r_multiple", 0)
        win = r > 0
        
        if strategy in self.strategy_stats:
            s = self.strategy_stats[strategy]
            if win:
                s["wins"] += 1
            else:
                s["losses"] += 1
            total = s["wins"] + s["losses"]
            s["total_r"] += r
            s["avg_r"] = s["total_r"] / total if total > 0 else 0
    
    def get_context_for_model(self, mode: str = "normal") -> Dict:
        """Get minimal context for model (NO full history)."""
        base = {
            "current": self.active_candle.to_dict() if self.active_candle else {},
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
            base["recent"] = [c.to_dict() for c in list(self.candle_buffer)[-20:]]
        
        return base
    
    def summary(self) -> Dict:
        return {
            "buffer": len(self.candle_buffer),
            "regime": self.regime_state.value,
            "conf": round(self.regime_confidence, 3),
            "support": [round(x, 2) for x in self.support_levels[-3:]],
            "resistance": [round(x, 2) for x in self.resistance_levels[-3:]],
            "vwap": round(self.session_vwap, 2),
            "bias": self.session_bias,
            "stats": self.strategy_stats,
        }