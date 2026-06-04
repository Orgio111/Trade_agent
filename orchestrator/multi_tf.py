"""
QUANTEX Multi-TimeFrame Feature Engine — Aligns and aggregates data across Binance timeframes.

Architecture:
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 5m data  │   │ 1H data  │   │ 4H data  │   │ 1D data  │
  ├──────────┤   ├──────────┤   ├──────────┤   ├──────────┤
  │ OHLCV    │   │ OHLCV    │   │ OHLCV    │   │ OHLCV    │
  │ +feat.   │   │ +feat.   │   │ +feat.   │   │ +feat.   │
  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                        ▼
              ┌──────────────────┐
              │  Time Aligner    │  ← aligns all TF to base TF timestamps
              │  (forward-fill)  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Feature Vector   │  ← flattened into obs[40:80]
              └──────────────────┘

Usage:
    mtf = MultiTimeframeEngine(base_interval="5m")
    data = await mtf.fetch_all("BTCUSDT", days=180)
    # data is a dict of {interval: pd.DataFrame}
    
    # At step i, get the full MTF observation vector
    obs = mtf.get_observation(i)  # -> np.ndarray(40,)
    
    # Or get features for each TF separately
    features = mtf.get_features(i)
    # -> {"5m": {...}, "1h": {...}, "4h": {...}, "1d": {...}}
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import DataLoader


# ── Timeframe Configuration ─────────────────────────────────

# Standard Binance intervals and their minute-equivalents for alignment
TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "1w": 10080, "1M": 43200,
}

# ── Feature Computation ──────────────────────────────────────

def compute_tf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a compact feature set for a single timeframe's OHLCV data.

    Returns a DataFrame with the same index and extra columns:
      - returns_1: 1-period return
      - returns_5: 5-period return
      - rsi_14: RSI(14)
      - ema_trend: (close - EMA20) / EMA20 (position in trend)
      - atr_pct: ATR(14) / close
      - volume_ratio: volume / SMA20(volume)

    All values are normalized to roughly [-1, 1] range.
    """
    result = pd.DataFrame(index=df.index)

    # Returns
    result["returns_1"] = df["close"].pct_change(1).fillna(0)
    result["returns_5"] = df["close"].pct_change(5).fillna(0)

    # RSI(14) normalized to [-1, 1]
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    result["rsi_norm"] = (rsi.fillna(50) - 50) / 50  # -> [-1, 1]

    # EMA trend
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    result["ema_trend"] = ((df["close"] - ema20) / ema20.replace(0, np.nan)).fillna(0)
    # Clip to [-1, 1] — extreme divergences > 100% are rare
    result["ema_trend"] = result["ema_trend"].clip(-1, 1)

    # ATR% normalized
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    result["atr_pct"] = (atr / df["close"].replace(0, np.nan)).fillna(0)
    # Typical BTC atr% is 0.01-0.05, cap at 0.10
    result["atr_pct"] = (result["atr_pct"] * 10).clip(-1, 1)

    # Volume ratio (log scale, clipped)
    vol_sma20 = df["volume"].rolling(20).mean().replace(0, np.nan)
    vol_ratio = (df["volume"] / vol_sma20).fillna(1.0)
    result["vol_ratio_norm"] = np.log(vol_ratio.clip(0.1, 10.0)) / np.log(10)  # -> [-1, 1]

    return result


# ── Time Aligner ─────────────────────────────────────────────

def align_to_base(
    base_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align a higher-timeframe DataFrame to the base DataFrame's timestamps
    using forward-fill (pad).

    For example, given 5m base and 1H higher TF:
      - All 5m candles within the same hour get the same 1H feature values
      - The 1H value is the most recent completed 1H candle

    Returns a DataFrame with the same index as base_df and the same columns
    as higher_tf_df, forward-filled.
    """
    # Merge on timestamp, forward-fill higher TF values
    aligned = base_df[[]].join(higher_tf_df, how="left")
    aligned = aligned.ffill().bfill()
    return aligned


# ── Multi-Timeframe Engine ───────────────────────────────────

class MultiTimeframeEngine:
    """
    Engine that fetches, aligns, and extracts features from multiple timeframes.

    Designed for RL training: provides a compact feature vector (~40 elements)
    that can be appended to the base observation.

    Usage:
        mtf = MultiTimeframeEngine(base_interval="5m", higher_intervals=["1h", "4h", "1d"])
        await mtf.fetch_and_prepare("BTCUSDT", days=365)

        # During RL training step:
        obs_mtf = mtf.get_observation_vector(step_index)
        # obs_mtf.shape = (40,)  # Concatenated features from all TFs
    """

    def __init__(
        self,
        base_interval: str = "5m",
        higher_intervals: Optional[list[str]] = None,
    ):
        """
        Args:
            base_interval: The primary trading timeframe (e.g. "5m")
            higher_intervals: Higher timeframes for context (e.g. ["1h", "4h", "1d"])
        """
        self.base_interval = base_interval
        self.higher_intervals = higher_intervals or ["1h", "4h", "1d"]
        self.all_intervals = [base_interval] + self.higher_intervals

        # Store raw data per interval
        self._raw_data: dict[str, pd.DataFrame] = {}
        # Store computed features per interval
        self._features: dict[str, pd.DataFrame] = {}
        # Store aligned features (all aligned to base index)
        self._aligned: dict[str, pd.DataFrame] = {}

        # Number of features per timeframe
        self._features_per_tf = 8  # returns_1, returns_5, rsi_norm, ema_trend, atr_pct, vol_ratio_norm + trend_dir + momentum_osc
        self._total_mtf_features = len(self.all_intervals) * self._features_per_tf

    async def fetch_and_prepare(
        self,
        symbol: str = "BTCUSDT",
        days: int = 365,
    ):
        """
        Fetch OHLCV data for all configured timeframes and compute features.

        Args:
            symbol: Trading pair
            days: How many days of historical data to fetch per interval
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        # Estimate how many base-interval candles we need
        base_minutes = TIMEFRAME_MINUTES.get(self.base_interval, 5)
        estimated_base_candles = (days * 24 * 60) // base_minutes + 1000

        for interval in self.all_intervals:
            interval_minutes = TIMEFRAME_MINUTES.get(interval, 60)
            # For higher timeframes, we need proportionally fewer candles
            ratio = max(1, interval_minutes // base_minutes)
            limit = min(1500, estimated_base_candles // ratio + 200)

            df = await DataLoader.from_binance_api(
                symbol=symbol,
                interval=interval,
                start_time=start,
                end_time=end,
                limit=limit,
            )

            if df.empty:
                print(f"  [WARN] No data for {interval}, using synthetic fallback")
                df = self._synthetic_fallback(interval, estimated_base_candles // ratio)
            else:
                print(f"  [OK] {interval}: {len(df)} candles ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")

            self._raw_data[interval] = df

        self._compute_and_align()

    def _synthetic_fallback(self, interval: str, periods: int) -> pd.DataFrame:
        """Generate synthetic data when Binance API is unavailable."""
        rng = np.random.RandomState(hash(interval) % 2**32)
        interval_minutes = TIMEFRAME_MINUTES.get(interval, 60)
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=f"{interval_minutes}min")
        price = 50000 * np.exp(np.cumsum(rng.normal(0, 0.02 / np.sqrt(24 * 60 / interval_minutes), periods)))
        return pd.DataFrame({
            "open": price * (1 + rng.normal(0, 0.001, periods)),
            "high": price * (1 + abs(rng.normal(0, 0.002, periods))),
            "low": price * (1 - abs(rng.normal(0, 0.002, periods))),
            "close": price,
            "volume": rng.exponential(100, periods),
        }, index=dates)

    def _compute_and_align(self):
        """Compute features for each TF and align all to base TF index."""
        base_df = self._raw_data.get(self.base_interval)
        if base_df is None:
            raise ValueError(f"Base interval {self.base_interval} data not found")

        for interval in self.all_intervals:
            df = self._raw_data.get(interval)
            if df is None or len(df) < 50:
                continue

            # Compute features
            features = compute_tf_features(df)
            self._features[interval] = features

            if interval == self.base_interval:
                # Base TF aligns perfectly
                self._aligned[interval] = features
            else:
                # Higher TF: align to base index via forward-fill
                self._aligned[interval] = align_to_base(base_df, features)

        print(f"  [MTF] Aligned {len(self.all_intervals)} timeframes to '{self.base_interval}' index ({len(base_df)} steps)")

    def get_observation_vector(self, step_index: int) -> np.ndarray:
        """
        Build the MTF observation vector for a given step.

        Returns a flat np.ndarray of shape (total_mtf_features,) with
        features from all timeframes concatenated.

        Feature order: [base_TF(8), higher_TF_1(8), higher_TF_2(8), ...]

        Each TF block:
          [returns_1, returns_5, rsi_norm, ema_trend, atr_pct, vol_ratio_norm, trend_dir, momentum_osc]

        All values normalized to roughly [-1, 1].
        """
        obs = np.zeros(self._total_mtf_features, dtype=np.float32)

        offset = 0
        for interval in self.all_intervals:
            aligned = self._aligned.get(interval)
            if aligned is None or step_index >= len(aligned):
                offset += self._features_per_tf
                continue

            row = aligned.iloc[step_index]

            # Core features (6)
            obs[offset] = float(row.get("returns_1", 0))           # 0: short-term return
            obs[offset + 1] = float(row.get("returns_5", 0))       # 1: medium return
            obs[offset + 2] = float(row.get("rsi_norm", 0))        # 2: RSI position
            obs[offset + 3] = float(row.get("ema_trend", 0))       # 3: EMA position
            obs[offset + 4] = float(row.get("atr_pct", 0))         # 4: volatility
            obs[offset + 5] = float(row.get("vol_ratio_norm", 0))  # 5: volume

            # Derived features (2)
            rsi_val = obs[offset + 2]
            obs[offset + 6] = 1.0 if rsi_val > 0.1 else (-1.0 if rsi_val < -0.1 else 0.0)  # 6: trend direction from RSI

            # Momentum oscillator: difference of returns
            ret1 = obs[offset]     # 1-period return
            ret5 = obs[offset + 1]  # 5-period return
            obs[offset + 7] = float(np.clip((ret1 - ret5 / 5) * 10, -1, 1))  # 7: momentum

            offset += self._features_per_tf

        return obs

    def get_features_dict(self, step_index: int) -> dict:
        """
        Get human-readable feature dict for a given step.

        Returns {interval: {feature_name: value, ...}, ...}
        """
        result = {}
        for interval in self.all_intervals:
            aligned = self._aligned.get(interval)
            if aligned is None or step_index >= len(aligned):
                result[interval] = None
                continue

            row = aligned.iloc[step_index]
            result[interval] = {
                "returns_1": round(float(row.get("returns_1", 0)), 6),
                "returns_5": round(float(row.get("returns_5", 0)), 6),
                "rsi_norm": round(float(row.get("rsi_norm", 0)), 4),
                "ema_trend": round(float(row.get("ema_trend", 0)), 4),
                "atr_pct": round(float(row.get("atr_pct", 0)), 4),
                "vol_ratio_norm": round(float(row.get("vol_ratio_norm", 0)), 4),
            }
        return result

    @property
    def total_mtf_features(self) -> int:
        """Total number of features across all timeframes."""
        return self._total_mtf_features

    @property
    def base_length(self) -> int:
        """Number of candles in the base timeframe."""
        base = self._raw_data.get(self.base_interval)
        return len(base) if base is not None else 0
    
    @property
    def intervals(self) -> list[str]:
        """List of all configured intervals."""
        return self.all_intervals

    def summary(self) -> str:
        """Print a summary of loaded MTF data."""
        lines = ["Multi-Timeframe Engine Summary:"]
        lines.append(f"  Base TF:     {self.base_interval}")
        lines.append(f"  Higher TFs:  {', '.join(self.higher_intervals)}")
        lines.append(f"  Total TF features: {self._total_mtf_features}")
        for interval in self.all_intervals:
            df = self._raw_data.get(interval)
            if df is not None:
                lines.append(f"  {interval}: {len(df):,} candles")
        return "\n".join(lines)


# ── Helper for direct use ────────────────────────────────────

async def fetch_mtf_data(
    symbol: str = "BTCUSDT",
    base_interval: str = "5m",
    higher_intervals: Optional[list[str]] = None,
    days: int = 180,
) -> MultiTimeframeEngine:
    """
    Convenience function: fetch data and return a ready-to-use MTF engine.

    Usage:
        mtf = await fetch_mtf_data()
        print(mtf.summary())
        obs = mtf.get_observation_vector(step_index=100)
    """
    engine = MultiTimeframeEngine(
        base_interval=base_interval,
        higher_intervals=higher_intervals,
    )
    await engine.fetch_and_prepare(symbol=symbol, days=days)
    return engine
