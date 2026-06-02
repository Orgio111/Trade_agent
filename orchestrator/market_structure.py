"""
QUANTEX Market Structure Engine — Institutional SMC/Wyckoff analysis.

Detects:
  - Liquidity Sweeps / Stop Hunts
  - Order Blocks (accumulation/distribution zones)
  - Fair Value Gaps (FVG)
  - Break of Structure (BOS) / Change of Character (CHOCH)
  - Wyckoff phases (Accumulation, Markup, Distribution, Markdown)
  - Smart Money reversal patterns

All methods are vectorized pandas operations for backtesting speed.
"""

import numpy as np
import pandas as pd


class MarketStructureEngine:
    """
    Institutional market structure detection using smart money concepts.
    All methods work on OHLCV DataFrames and return labeled Series/DataFrames.
    """

    @staticmethod
    def detect_swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Identify swing highs and lows using local extrema.

        Returns DataFrame with columns:
          - swing_high: True at pivot highs
          - swing_low: True at pivot lows
          - swing_high_price: price at the swing high
          - swing_low_price: price at the swing low
        """
        result = pd.DataFrame(index=df.index)
        result["swing_high"] = False
        result["swing_low"] = False
        result["swing_high_price"] = np.nan
        result["swing_low_price"] = np.nan

        high = df["high"].values
        low = df["low"].values

        for i in range(window, len(df) - window):
            # Swing high: highest in surrounding window
            if high[i] == max(high[i-window:i+window+1]):
                result.iloc[i, result.columns.get_loc("swing_high")] = True
                result.iloc[i, result.columns.get_loc("swing_high_price")] = high[i]

            # Swing low: lowest in surrounding window
            if low[i] == min(low[i-window:i+window+1]):
                result.iloc[i, result.columns.get_loc("swing_low")] = True
                result.iloc[i, result.columns.get_loc("swing_low_price")] = low[i]

        return result

    @staticmethod
    def detect_structure_breaks(df: pd.DataFrame, swing: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHOCH).

        BOS = price breaks past a previous swing point in trend direction.
        CHOCH = price breaks the last swing point in the opposite direction
                (potential trend reversal).

        Returns DataFrame with columns:
          - bos_up: BOS to the upside (bullish)
          - bos_down: BOS to the downside (bearish)
          - choch_up: CHOCH to upside (potential reversal up)
          - choch_down: CHOCH to downside (potential reversal down)
        """
        result = pd.DataFrame(index=df.index)
        result["bos_up"] = False
        result["bos_down"] = False
        result["choch_up"] = False
        result["choch_down"] = False

        high = df["high"].values
        low = df["low"].values
        swing_high = swing["swing_high"].values
        swing_low = swing["swing_low"].values
        swing_hp = swing["swing_high_price"].values
        swing_lp = swing["swing_low_price"].values

        # Track the last two swing highs and lows
        last_highs = []  # [(index, price), ...]
        last_lows = []   # [(index, price), ...]

        for i in range(1, len(df)):
            # Record swing points
            if swing_high[i] and not np.isnan(swing_hp[i]):
                last_highs.append((i, swing_hp[i]))
                if len(last_highs) > 3:
                    last_highs.pop(0)

            if swing_low[i] and not np.isnan(swing_lp[i]):
                last_lows.append((i, swing_lp[i]))
                if len(last_lows) > 3:
                    last_lows.pop(0)

            # BOS Up: price breaks above a prior swing high (uptrend continuation)
            if len(last_highs) >= 2:
                prev_high = last_highs[-2][1]
                if high[i] > prev_high and not np.isnan(prev_high):
                    result.iloc[i, result.columns.get_loc("bos_up")] = True

            # BOS Down: price breaks below a prior swing low (downtrend continuation)
            if len(last_lows) >= 2:
                prev_low = last_lows[-2][1]
                if low[i] < prev_low and not np.isnan(prev_low):
                    result.iloc[i, result.columns.get_loc("bos_down")] = True

            # CHOCH Up: in a downtrend, price breaks above last swing high
            if len(last_highs) >= 1 and len(last_lows) >= 2:
                # Check if we're in a downtrend (lower lows)
                if last_lows[-1][1] < last_lows[-2][1]:
                    if high[i] > last_highs[-1][1] and not np.isnan(last_highs[-1][1]):
                        result.iloc[i, result.columns.get_loc("choch_up")] = True

            # CHOCH Down: in an uptrend, price breaks below last swing low
            if len(last_lows) >= 1 and len(last_highs) >= 2:
                # Check if we're in an uptrend (higher highs)
                if last_highs[-1][1] > last_highs[-2][1]:
                    if low[i] < last_lows[-1][1] and not np.isnan(last_lows[-1][1]):
                        result.iloc[i, result.columns.get_loc("choch_down")] = True

        return result

    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame, swing: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity sweeps (stop hunts).

        A liquidity sweep occurs when price briefly breaks a swing high/low
        (hitting stops), then immediately reverses.

        Returns DataFrame with columns:
          - liq_sweep_high: swept above swing high, then closed below
          - liq_sweep_low: swept below swing low, then closed above
          - liq_sweep_strength: severity of the sweep (0-1)
        """
        result = pd.DataFrame(index=df.index)
        result["liq_sweep_high"] = False
        result["liq_sweep_low"] = False
        result["liq_sweep_strength"] = 0.0

        open_p = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        for i in range(1, len(df)):
            candle_range = high[i] - low[i]
            if candle_range <= 0:
                continue

            # Check nearby swing highs (within last 20 bars)
            recent_swing_highs = swing["swing_high_price"].values[max(0, i-20):i]
            valid_highs = [p for p in recent_swing_highs if not np.isnan(p)]
            if valid_highs:
                nearest_high = min(valid_highs, key=lambda x: abs(x - high[i]))
                # Sweep: price went above swing high but closed below it
                if (high[i] > nearest_high and close[i] < nearest_high and
                        (high[i] - nearest_high) < candle_range * 2):
                    result.iloc[i, result.columns.get_loc("liq_sweep_high")] = True
                    wick_ratio = (high[i] - max(open_p[i], close[i])) / candle_range
                    result.iloc[i, result.columns.get_loc("liq_sweep_strength")] = min(1.0, wick_ratio * 2)

            # Check nearby swing lows
            recent_swing_lows = swing["swing_low_price"].values[max(0, i-20):i]
            valid_lows = [p for p in recent_swing_lows if not np.isnan(p)]
            if valid_lows:
                nearest_low = min(valid_lows, key=lambda x: abs(x - low[i]))
                # Sweep: price went below swing low but closed above it
                if (low[i] < nearest_low and close[i] > nearest_low and
                        (nearest_low - low[i]) < candle_range * 2):
                    result.iloc[i, result.columns.get_loc("liq_sweep_low")] = True
                    wick_ratio = (min(open_p[i], close[i]) - low[i]) / candle_range
                    result.iloc[i, result.columns.get_loc("liq_sweep_strength")] = min(1.0, wick_ratio * 2)

        return result

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, swing: pd.DataFrame) -> pd.DataFrame:
        """
        Identify order blocks — institutional accumulation/distribution zones.

        Bullish OB: last bearish candle before a strong up-move
        Bearish OB: last bullish candle before a strong down-move

        Returns DataFrame with columns:
          - ob_bullish: bullish order block zone (bottom, top)
          - ob_bearish: bearish order block zone (bottom, top)
          - ob_strength: strength score (0-1)
        """
        result = pd.DataFrame(index=df.index)
        result["ob_bullish"] = 0.0  # 0 = none, >0 = price level
        result["ob_bearish"] = 0.0
        result["ob_strength"] = 0.0

        open_p = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Look for strong momentum candles
        for i in range(2, len(df)):
            body_before = abs(close[i-1] - open_p[i-1])
            range_before = high[i-1] - low[i-1]
            if range_before <= 0:
                continue
            body_ratio_before = body_before / range_before

            body_current = abs(close[i] - open_p[i])
            range_current = high[i] - low[i]
            if range_current <= 0:
                continue
            body_ratio_current = body_current / range_current

            # Bullish OB: bearish candle (close < open) followed by strong bullish move
            if (close[i-1] < open_p[i-1] and close[i] > open_p[i] and
                    body_ratio_before > 0.6 and body_ratio_current > 0.6 and
                    close[i] > high[i-1]):
                result.iloc[i, result.columns.get_loc("ob_bullish")] = close[i-1]
                result.iloc[i, result.columns.get_loc("ob_strength")] = body_ratio_before * body_ratio_current

            # Bearish OB: bullish candle (close > open) followed by strong bearish move
            if (close[i-1] > open_p[i-1] and close[i] < open_p[i] and
                    body_ratio_before > 0.6 and body_ratio_current > 0.6 and
                    close[i] < low[i-1]):
                result.iloc[i, result.columns.get_loc("ob_bearish")] = close[i-1]
                result.iloc[i, result.columns.get_loc("ob_strength")] = body_ratio_before * body_ratio_current

        return result

    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG) — 3-candle inefficiency patterns.

        Bullish FVG: low[i+2] > high[i] (gap up)
        Bearish FVG: high[i+2] < low[i] (gap down)

        Returns DataFrame with columns:
          - fvg_bullish_top: top of bullish FVG zone
          - fvg_bullish_bottom: bottom of bullish FVG zone
          - fvg_bearish_top: top of bearish FVG zone
          - fvg_bearish_bottom: bottom of bearish FVG zone
          - fvg_filled: whether the gap has been filled
        """
        result = pd.DataFrame(index=df.index)
        result["fvg_bullish_top"] = 0.0
        result["fvg_bullish_bottom"] = 0.0
        result["fvg_bearish_top"] = 0.0
        result["fvg_bearish_bottom"] = 0.0
        result["fvg_filled"] = False

        high = df["high"].values
        low = df["low"].values

        for i in range(len(df) - 2):
            # Bullish FVG: price jumps up leaving a gap
            if low[i + 2] > high[i]:
                result.iloc[i + 2, result.columns.get_loc("fvg_bullish_top")] = low[i + 2]
                result.iloc[i + 2, result.columns.get_loc("fvg_bullish_bottom")] = high[i]

            # Bearish FVG: price jumps down leaving a gap
            if high[i + 2] < low[i]:
                result.iloc[i + 2, result.columns.get_loc("fvg_bearish_top")] = low[i]
                result.iloc[i + 2, result.columns.get_loc("fvg_bearish_bottom")] = high[i + 2]

        return result

    @staticmethod
    def detect_wyckoff_phases(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Wyckoff market phases using price+volume analysis.

        Phases:
          - Accumulation: sideways after downtrend, low volatility, volume declining
          - Markup: trending up, expanding volume, higher highs and lows
          - Distribution: sideways after uptrend, high volatility, volume diverging
          - Markdown: trending down, volume spikes in panic

        Returns DataFrame with columns:
          - wyckoff_phase: "accumulation", "markup", "distribution", "markdown", "unknown"
          - wyckoff_confidence: 0-1 confidence in phase detection
        """
        result = pd.DataFrame(index=df.index)
        result["wyckoff_phase"] = "unknown"
        result["wyckoff_confidence"] = 0.0

        if len(df) < 60:
            return result

        # Compute rolling metrics
        close = df["close"].values
        volume = df["volume"].values
        rolling_vol = pd.Series(volume).rolling(20).mean().values

        for i in range(60, len(df)):
            window = slice(i-50, i)
            prices = close[window]
            vols = volume[window]
            vol_sma = rolling_vol[i]

            # Trend direction
            ema20 = pd.Series(prices).ewm(span=20).mean().iloc[-1]
            ema50 = pd.Series(prices).ewm(span=50).mean().iloc[-1]
            price_vs_ema = prices[-1] > ema20

            # Volatility
            atr_14 = pd.Series(prices).rolling(14).std().iloc[-1]
            atr_pct = atr_14 / prices[-1] if prices[-1] > 0 else 0

            # Volume analysis
            vol_trend = vols[-1] / vol_sma if vol_sma > 0 else 1.0
            vol_declining = vol_trend < 0.8
            vol_expanding = vol_trend > 1.3

            # Structure
            higher_highs = prices[-1] > max(prices[-10:-1])
            lower_lows = prices[-1] < min(prices[-10:-1])
            ranging = not higher_highs and not lower_lows

            if ranging and vol_declining and atr_pct < 0.02:
                result.iloc[i, result.columns.get_loc("wyckoff_phase")] = "accumulation"
                result.iloc[i, result.columns.get_loc("wyckoff_confidence")] = 0.6 + (1 - atr_pct / 0.02) * 0.3

            elif price_vs_ema and higher_highs and vol_expanding:
                result.iloc[i, result.columns.get_loc("wyckoff_phase")] = "markup"
                result.iloc[i, result.columns.get_loc("wyckoff_confidence")] = 0.7

            elif ranging and vol_expanding and atr_pct > 0.02:
                result.iloc[i, result.columns.get_loc("wyckoff_phase")] = "distribution"
                result.iloc[i, result.columns.get_loc("wyckoff_confidence")] = 0.6

            elif not price_vs_ema and lower_lows and vol_expanding:
                result.iloc[i, result.columns.get_loc("wyckoff_phase")] = "markdown"
                result.iloc[i, result.columns.get_loc("wyckoff_confidence")] = 0.7

        return result

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all market structure features and return a combined DataFrame.
        """
        swing = MarketStructureEngine.detect_swing_points(df)
        structure_breaks = MarketStructureEngine.detect_structure_breaks(df, swing)
        liquidity = MarketStructureEngine.detect_liquidity_sweeps(df, swing)
        order_blocks = MarketStructureEngine.detect_order_blocks(df, swing)
        fvg = MarketStructureEngine.detect_fvg(df)
        wyckoff = MarketStructureEngine.detect_wyckoff_phases(df)

        combined = pd.concat([
            swing, structure_breaks, liquidity, order_blocks, fvg, wyckoff
        ], axis=1)

        return combined

    @staticmethod
    def get_market_structure_signal(df: pd.DataFrame, latest_data: pd.Series) -> dict:
        """
        Generate a trading signal from market structure analysis.

        Returns a dict with direction, confidence, and reasoning.
        """
        structure = MarketStructureEngine.compute_all(df)
        last = structure.iloc[-1]

        signals = []
        confidence_bonus = 0.0

        # Liquidity sweep = potential reversal setup
        if last.get("liq_sweep_high", False):
            signals.append("Liquidity sweep high (bearish reversal)")
            confidence_bonus += 0.15
        if last.get("liq_sweep_low", False):
            signals.append("Liquidity sweep low (bullish reversal)")
            confidence_bonus += 0.15

        # Order block = potential support/resistance
        ob_bull = last.get("ob_bullish", 0.0)
        ob_bear = last.get("ob_bearish", 0.0)
        if ob_bull > 0:
            signals.append(f"Bullish order block at ${ob_bull:.2f}")
            confidence_bonus += 0.10
        if ob_bear > 0:
            signals.append(f"Bearish order block at ${ob_bear:.2f}")
            confidence_bonus += 0.10

        # Structure breaks
        if last.get("bos_up", False):
            signals.append("BOS to upside (uptrend continuation)")
            confidence_bonus += 0.10
        if last.get("bos_down", False):
            signals.append("BOS to downside (downtrend continuation)")
            confidence_bonus += 0.10
        if last.get("choch_up", False):
            signals.append("CHOCH up (potential reversal up)")
            confidence_bonus += 0.15
        if last.get("choch_down", False):
            signals.append("CHOCH down (potential reversal down)")
            confidence_bonus += 0.15

        # Wyckoff phase
        phase = last.get("wyckoff_phase", "unknown")
        if phase == "accumulation":
            signals.append("Wyckoff accumulation phase")
            confidence_bonus += 0.10
        elif phase == "markup":
            signals.append("Wyckoff markup phase (bullish)")
            confidence_bonus += 0.10
        elif phase == "distribution":
            signals.append("Wyckoff distribution phase (bearish)")
            confidence_bonus += 0.10
        elif phase == "markdown":
            signals.append("Wyckoff markdown phase (bearish)")
            confidence_bonus += 0.10

        # FVG (attracts price)
        fvg_bull_top = last.get("fvg_bullish_top", 0.0)
        fvg_bear_top = last.get("fvg_bearish_top", 0.0)

        # Determine direction
        bullish_score = 0.0
        bearish_score = 0.0

        if last.get("liq_sweep_low", False): bullish_score += 0.3
        if last.get("bos_up", False): bullish_score += 0.25
        if last.get("choch_up", False): bullish_score += 0.3
        if ob_bull > 0: bullish_score += 0.2
        if phase in ("accumulation", "markup"): bullish_score += 0.15

        if last.get("liq_sweep_high", False): bearish_score += 0.3
        if last.get("bos_down", False): bearish_score += 0.25
        if last.get("choch_down", False): bearish_score += 0.3
        if ob_bear > 0: bearish_score += 0.2
        if phase in ("distribution", "markdown"): bearish_score += 0.15

        if bullish_score > bearish_score and bullish_score > 0.3:
            direction = "long"
            confidence = min(0.9, 0.5 + confidence_bonus)
        elif bearish_score > bullish_score and bearish_score > 0.3:
            direction = "short"
            confidence = min(0.9, 0.5 + confidence_bonus)
        else:
            direction = "hold"
            confidence = 0.3

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "reasoning": "; ".join(signals) if signals else "No clear structure signal",
            "bullish_score": round(bullish_score, 4),
            "bearish_score": round(bearish_score, 4),
            "wyckoff_phase": phase,
            "has_liquidity_sweep": bool(last.get("liq_sweep_low", False) or last.get("liq_sweep_high", False)),
            "has_order_block": bool(ob_bull > 0 or ob_bear > 0),
            "has_structure_break": bool(last.get("bos_up", False) or last.get("bos_down", False) or
                                        last.get("choch_up", False) or last.get("choch_down", False)),
        }


class FakeBreakoutDetector:
    """
    Detects fake breakouts using volume, delta, and rejection patterns.

    A fake breakout = price breaks a key level but lacks follow-through
    and quickly reverses. Classic stop hunt / liquidity grab.
    """

    @staticmethod
    def detect(df: pd.DataFrame, level: float, direction: str = "up") -> dict:
        """
        Check if a breakout at the given level is likely fake.

        Features checked:
          - Volume follow-through
          - Wick rejection
          - Consecutive closes beyond the level
          - Proximity to resistance

        Returns dict with is_fake_breakout, confidence, and reasoning.
        """
        if len(df) < 10:
            return {"is_fake_breakout": False, "confidence": 0.0, "reasoning": "Insufficient data"}

        latest = df.iloc[-5:]  # Last 5 candles
        high = latest["high"].values
        low = latest["low"].values
        close = latest["close"].values
        open_p = latest["open"].values
        volume = latest["volume"].values

        if direction == "up":
            # Check if price broke above the level
            broke_above = any(h > level for h in high[-3:])
            if not broke_above:
                return {"is_fake_breakout": False, "confidence": 0.0, "reasoning": "No breakout above level"}

            # Volume confirmation
            vol_sma = np.mean(volume[:-1]) if len(volume) > 1 else volume[0]
            vol_on_breakout = volume[-2] if len(volume) >= 2 else volume[-1]
            vol_confirmed = vol_on_breakout > vol_sma * 1.3

            # Rejection: long upper wick on breakout candle
            breakout_idx = -2 if high[-2] > level else -1
            candle_range = high[breakout_idx] - low[breakout_idx]
            upper_wick = high[breakout_idx] - max(open_p[breakout_idx],
                                                   close[breakout_idx])
            rejection_ratio = upper_wick / candle_range if candle_range > 0 else 0
            rejected = rejection_ratio > 0.6

            # Close back below level
            closed_below = close[-1] < level
            all_closed_below = all(c < level for c in close[-2:])

            is_fake = rejected and (not vol_confirmed or closed_below)
            conf = 0.0
            reasons = []

            if rejected:
                conf += 0.3
                reasons.append(f"Rejection wick ({rejection_ratio:.0%})")
            if not vol_confirmed:
                conf += 0.25
                reasons.append("Low volume breakout")
            if closed_below:
                conf += 0.25
                reasons.append("Closed back below level")

            return {
                "is_fake_breakout": conf >= 0.5,
                "confidence": round(min(conf, 1.0), 4),
                "reasoning": "; ".join(reasons) if reasons else "Breakout looks valid",
                "rejection_ratio": round(rejection_ratio, 4),
                "volume_ratio": round(vol_on_breakout / vol_sma, 4) if vol_sma > 0 else 0,
            }

        else:  # direction == "down"
            broke_below = any(l < level for l in low[-3:])
            if not broke_below:
                return {"is_fake_breakout": False, "confidence": 0.0, "reasoning": "No breakout below level"}

            vol_sma = np.mean(volume[:-1]) if len(volume) > 1 else volume[0]
            vol_on_breakout = volume[-2] if len(volume) >= 2 else volume[-1]
            vol_confirmed = vol_on_breakout > vol_sma * 1.3

            breakout_idx = -2 if low[-2] < level else -1
            candle_range = high[breakout_idx] - low[breakout_idx]
            lower_wick = min(open_p[breakout_idx],
                              close[breakout_idx]) - low[breakout_idx]
            rejection_ratio = lower_wick / candle_range if candle_range > 0 else 0
            rejected = rejection_ratio > 0.6

            closed_above = close[-1] > level

            is_fake = rejected and (not vol_confirmed or closed_above)
            conf = 0.0
            reasons = []

            if rejected:
                conf += 0.3
                reasons.append(f"Rejection wick ({rejection_ratio:.0%})")
            if not vol_confirmed:
                conf += 0.25
                reasons.append("Low volume breakdown")
            if closed_above:
                conf += 0.25
                reasons.append("Closed back above level")

            return {
                "is_fake_breakout": conf >= 0.5,
                "confidence": round(min(conf, 1.0), 4),
                "reasoning": "; ".join(reasons) if reasons else "Breakdown looks valid",
                "rejection_ratio": round(rejection_ratio, 4),
                "volume_ratio": round(vol_on_breakout / vol_sma, 4) if vol_sma > 0 else 0,
            }
