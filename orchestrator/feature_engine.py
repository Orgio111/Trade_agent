"""
Feature engineering pipeline — computes technical indicators,
market microstructure features, and regime classification.
"""
import numpy as np
import pandas as pd


class FeatureEngine:
    """
    Comprehensive feature engineering for AI models.
    Generates the full feature set for the RL environment observation space.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Generate full feature set from OHLCV data."""
        result = df.copy()

        # ── Price Action ────────────────
        result["returns"] = result["close"].pct_change()
        result["log_returns"] = np.log(result["close"] / result["close"].shift(1))
        result["hl_range"] = (result["high"] - result["low"]) / result["close"]
        result["oc_range"] = (result["close"] - result["open"]) / result["open"].replace(0, np.nan)

        # ── Trend ───────────────────────
        for period in [5, 10, 20, 50, 200]:
            result[f"ema_{period}"] = result["close"].ewm(span=period, adjust=False).mean()

        result["ema_5_cross_20"] = np.sign(result["ema_5"] - result["ema_20"])
        result["ema_20_cross_50"] = np.sign(result["ema_20"] - result["ema_50"])
        result["price_vs_ema200"] = (result["close"] - result["ema_200"]) / result["ema_200"].replace(0, np.nan)

        # MACD
        ema12 = result["close"].ewm(span=12, adjust=False).mean()
        ema26 = result["close"].ewm(span=26, adjust=False).mean()
        result["macd"] = ema12 - ema26
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # ── Momentum ────────────────────
        delta = result["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain_14 = gain.rolling(14).mean()
        avg_loss_14 = loss.rolling(14).mean()
        rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.nan)
        result["rsi_14"] = 100 - (100 / (1 + rs_14))

        avg_gain_7 = gain.rolling(7).mean()
        avg_loss_7 = loss.rolling(7).mean()
        rs_7 = avg_gain_7 / avg_loss_7.replace(0, np.nan)
        result["rsi_7"] = 100 - (100 / (1 + rs_7))

        # Stochastic
        low_14 = result["low"].rolling(14).min()
        high_14 = result["high"].rolling(14).max()
        result["stoch_k"] = 100 * (result["close"] - low_14) / (high_14 - low_14).replace(0, np.nan)
        result["stoch_d"] = result["stoch_k"].rolling(3).mean()

        # ── Volatility ──────────────────
        tr = pd.concat([
            result["high"] - result["low"],
            (result["high"] - result["close"].shift()).abs(),
            (result["low"] - result["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        result["atr_14"] = tr.rolling(14).mean()
        result["atr_pct"] = result["atr_14"] / result["close"]

        # Bollinger Bands
        result["bb_mid"] = result["close"].rolling(20).mean()
        bb_std = result["close"].rolling(20).std()
        result["bb_upper"] = result["bb_mid"] + (bb_std * 2)
        result["bb_lower"] = result["bb_mid"] - (bb_std * 2)
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"].replace(0, np.nan)
        result["bb_position"] = (result["close"] - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)

        # ── Volume ──────────────────────
        result["vol_sma_20"] = result["volume"].rolling(20).mean()
        result["vol_ratio"] = result["volume"] / result["vol_sma_20"].replace(0, np.nan)
        result["obv"] = (np.sign(result["close"].diff()) * result["volume"]).fillna(0).cumsum()

        # ── Market Structure ────────────
        result["higher_high"] = (result["high"] > result["high"].shift(1)).astype(int)
        result["lower_low"] = (result["low"] < result["low"].shift(1)).astype(int)

        return result

    @staticmethod
    def detect_regime(df: pd.DataFrame) -> str:
        """
        Classify current market regime using ADX + EMA + volatility.
        Returns: 'strong_uptrend', 'strong_downtrend', 'ranging', 'high_volatility', 'weak_trend'
        """
        if len(df) < 50:
            return "weak_trend"

        recent = df.tail(50)

        # Trend strength via ADX approximation
        tr = pd.concat([
            recent["high"] - recent["low"],
            (recent["high"] - recent["close"].shift()).abs(),
            (recent["low"] - recent["close"].shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else tr.mean()
        baseline_vol = df["atr_14"].mean() if "atr_14" in df.columns else df["close"].pct_change().std()
        vol_regime = "high" if atr > baseline_vol * 1.5 else "normal" if atr > baseline_vol * 0.7 else "low"

        # Direction via EMA200
        ema200 = recent["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        price_above_ema200 = recent["close"].iloc[-1] > ema200

        # ADX calculation
        up_move = recent["high"].diff()
        down_move = -recent["low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr.rolling(14).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / tr_smooth.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / tr_smooth.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(14).mean().iloc[-1] if len(dx) >= 14 else 0

        if adx > 25 and price_above_ema200 and vol_regime != "high":
            return "strong_uptrend"
        elif adx > 25 and not price_above_ema200 and vol_regime != "high":
            return "strong_downtrend"
        elif adx < 20 and vol_regime == "low":
            return "ranging"
        elif vol_regime == "high":
            return "high_volatility"
        else:
            return "weak_trend"

    @staticmethod
    def compute_orderbook_features(orderbook: dict) -> dict:
        """Extract alpha features from order book data."""
        bids = np.array(orderbook.get("bids", []))
        asks = np.array(orderbook.get("asks", []))

        if len(bids) == 0 or len(asks) == 0:
            return {
                "bid_ask_ratio": 1.0,
                "book_imbalance": 0.0,
                "spread_bps": 0.0,
                "bid_depth_5": 0.0,
                "ask_depth_5": 0.0,
            }

        total_bid = bids[:, 1].sum()
        total_ask = asks[:, 1].sum()

        return {
            "bid_ask_ratio": total_bid / (total_ask + 1e-10),
            "book_imbalance": (total_bid - total_ask) / (total_bid + total_ask + 1e-10),
            "spread_bps": (asks[0, 0] - bids[0, 0]) / bids[0, 0] * 10000,
            "bid_depth_5": bids[:5, 1].sum(),
            "ask_depth_5": asks[:5, 1].sum(),
            "top5_imbalance": (bids[:5, 1].sum() - asks[:5, 1].sum()) / (bids[:5, 1].sum() + asks[:5, 1].sum() + 1e-10),
        }
