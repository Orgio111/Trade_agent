"""Feature Extraction - Technical indicators computed incrementally."""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from models import Candle, CandleFeatures


class FeatureExtractor:
    """
    Incremental technical indicator computation.
    All indicators update in O(1) per new candle.
    """
    
    def __init__(self):
        # Price history
        self.closes = deque(maxlen=200)
        self.highs = deque(maxlen=200)
        self.lows = deque(maxlen=200)
        self.volumes = deque(maxlen=200)
        
        # RSI state
        self.gains = deque(maxlen=14)
        self.losses = deque(maxlen=14)
        
        # MACD state
        self.ema_12 = None
        self.ema_26 = None
        self.macd_signal = None
        
        # Bollinger Bands
        self.bb_window = deque(maxlen=20)
        
        # ATR
        self.tr_window = deque(maxlen=14)
        
        # ADX
        self.dm_plus = deque(maxlen=14)
        self.dm_minus = deque(maxlen=14)
        self.tr_adx = deque(maxlen=14)
        
        # OBV
        self.obv = 0.0
        self.prev_close = None
        
        # Volume
        self.volume_window = deque(maxlen=20)
        
    def update(self, candle: Candle) -> Dict[str, float]:
        """
        Update all indicators with new candle.
        Returns dictionary of current indicator values.
        """
        close = candle.close
        high = candle.high
        low = candle.low
        volume = candle.volume
        
        # Update price history
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.volume_window.append(volume)
        self.bb_window.append(close)
        
        indicators = {}
        
        # Returns
        if len(self.closes) >= 2:
            indicators["ret_1"] = (self.closes[-1] - self.closes[-2]) / self.closes[-2]
        if len(self.closes) >= 6:
            indicators["ret_5"] = (self.closes[-1] - self.closes[-6]) / self.closes[-6]
        if len(self.closes) >= 16:
            indicators["ret_15"] = (self.closes[-1] - self.closes[-16]) / self.closes[-16]
        if len(self.closes) >= 61:
            indicators["ret_60"] = (self.closes[-1] - self.closes[-61]) / self.closes[-61]
        
        # RSI (14)
        if len(self.closes) >= 2:
            change = self.closes[-1] - self.closes[-2]
            self.gains.append(max(change, 0))
            self.losses.append(max(-change, 0))
            if len(self.gains) == 14:
                avg_gain = np.mean(self.gains)
                avg_loss = np.mean(self.losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    indicators["rsi"] = 100 - (100 / (1 + rs))
                else:
                    indicators["rsi"] = 100.0
        
        # EMA and MACD
        if self.closes:
            current = self.closes[-1]
            if self.ema_12 is None:
                self.ema_12 = current
                self.ema_26 = current
            else:
                self.ema_12 = 0.1538 * current + 0.8462 * self.ema_12  # 2/(12+1)
                self.ema_26 = 0.0741 * current + 0.9259 * self.ema_26  # 2/(26+1)
            
            indicators["ema_12"] = self.ema_12
            indicators["ema_26"] = self.ema_26
            
            if self.ema_12 is not None and self.ema_26 is not None:
                macd = self.ema_12 - self.ema_26
                indicators["macd"] = macd
                
                if self.macd_signal is None:
                    self.macd_signal = macd
                else:
                    self.macd_signal = 0.2 * macd + 0.8 * self.macd_signal
                
                indicators["macd_signal"] = self.macd_signal
                indicators["macd_hist"] = macd - self.macd_signal
        
        # Bollinger Bands (20)
        if len(self.bb_window) == 20:
            sma = np.mean(self.bb_window)
            std = np.std(self.bb_window)
            indicators["bb_upper"] = sma + 2 * std
            indicators["bb_lower"] = sma - 2 * std
            indicators["bb_mid"] = sma
            indicators["bb_width"] = (4 * std) / sma if sma > 0 else 0
            indicators["bb_pct"] = (self.closes[-1] - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"]) if indicators["bb_upper"] != indicators["bb_lower"] else 0.5
        
        # ATR (14)
        if len(self.highs) >= 2 and len(self.lows) >= 2 and len(self.closes) >= 2:
            tr = max(
                self.highs[-1] - self.lows[-1],
                abs(self.highs[-1] - self.closes[-2]),
                abs(self.lows[-1] - self.closes[-2])
            )
            self.tr_window.append(tr)
            if len(self.tr_window) == 14:
                indicators["atr"] = np.mean(self.tr_window)
                indicators["atr_pct"] = indicators["atr"] / self.closes[-1]
        
        # ADX (14)
        if len(self.highs) >= 2 and len(self.lows) >= 2:
            up_move = self.highs[-1] - self.highs[-2]
            down_move = self.lows[-2] - self.lows[-1]
            
            dm_p = max(up_move, 0) if up_move > down_move else 0
            dm_m = max(down_move, 0) if down_move > up_move else 0
            
            tr = max(
                self.highs[-1] - self.lows[-1],
                abs(self.highs[-1] - self.closes[-2]),
                abs(self.lows[-1] - self.closes[-2])
            )
            
            self.dm_plus.append(dm_p)
            self.dm_minus.append(dm_m)
            self.tr_adx.append(tr)
            
            if len(self.dm_plus) == 14:
                atr = np.mean(self.tr_adx)
                if atr > 0:
                    di_plus = 100 * np.mean(self.dm_plus) / atr
                    di_minus = 100 * np.mean(self.dm_minus) / atr
                    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
                    indicators["adx"] = dx
                    indicators["di_plus"] = di_plus
                    indicators["di_minus"] = di_minus
        
        # OBV
        if self.prev_close is not None:
            if self.closes[-1] > self.prev_close:
                self.obv += self.volumes[-1]
            elif self.closes[-1] < self.prev_close:
                self.obv -= self.volumes[-1]
        self.prev_close = self.closes[-1]
        indicators["obv"] = self.obv
        
        # Volume ratio
        if len(self.volume_window) >= 21:
            recent = np.mean(list(self.volume_window)[-5:])
            prev = np.mean(list(self.volume_window)[-20:-5])
            indicators["vol_ratio"] = recent / prev if prev > 0 else 1.0
        
        # VWAP deviation (would need session VWAP from state)
        # Stochastic
        if len(self.highs) >= 14 and len(self.lows) >= 14:
            high_14 = max(list(self.highs)[-14:])
            low_14 = min(list(self.lows)[-14:])
            if high_14 != low_14:
                indicators["stoch_k"] = 100 * (self.closes[-1] - low_14) / (high_14 - low_14)
        
        # Williams %R
        if len(self.highs) >= 14 and len(self.lows) >= 14:
            high_14 = max(list(self.highs)[-14:])
            low_14 = min(list(self.lows)[-14:])
            if high_14 != low_14:
                indicators["willr"] = -100 * (high_14 - self.closes[-1]) / (high_14 - low_14)
        
        return indicators
    
    def get_feature_vector(self) -> List[float]:
        """Get feature vector in fixed order for model input."""
        # This should match the order expected by the model
        # For now, return empty - use IncrementalState instead
        return []


def extract_features(candles: List[Dict]) -> np.ndarray:
    """
    Batch feature extraction for backtesting.
    """
    if len(candles) < 60:
        return np.array([])
    
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    
    n = len(candles)
    features = np.zeros((n, 20))  # 20 features
    
    # This is a simplified version for batch processing
    # Real implementation would use the incremental FeatureExtractor
    
    return features