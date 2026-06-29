"""Brain #2: FreqAI / XGBoost Technical Signals.

Uses frequency-domain features + XGBoost for short-term price prediction.
Weight in Go orchestrator: 0.15.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

# ── .env auto-load ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


class FreqAIBrain(BaseBrain):
    """FreqAI-inspired technical indicator brain.

    Computes RSI, MACD, Bollinger Band position, and volume profile
    from recent OHLCV data. Uses either a pre-trained XGBoost model
    or a rule-based scoring system as fallback.

    Features:
      - Auto-fetches OHLCV from Binance via ccxt on each compute_score()
      - Auto-trains XGBoost model from historical data if enough bars available
      - Persists trained model to disk for reuse across restarts
    """

    @property
    def brain_id(self) -> str:
        return "freqai"

    def __init__(self) -> None:
        self._model = None
        self._ohlcv_buffer: list[dict] = []
        self._max_bars = 200
        self._model_dir = os.getenv(
            "FREQAI_MODEL_DIR",
            str(Path(__file__).resolve().parents[2] / "models" / "freqai"),
        )
        self._model_path = os.getenv("FREQAI_MODEL_PATH", "")
        self._binance_fetched: dict[str, float] = {}  # symbol → last fetch timestamp

    async def warmup(self) -> None:
        """Attempt to load pre-trained XGBoost model, or auto-discover."""
        # If explicit path set and exists, load it
        if self._model_path and os.path.exists(self._model_path):
            try:
                import joblib
                self._model = joblib.load(self._model_path)
                logger.info("[freqai] Loaded XGBoost model from %s", self._model_path)
                return
            except ImportError:
                logger.warning("[freqai] joblib not installed, will train inline")
            except Exception as e:
                logger.warning("[freqai] Model load failed: %s", e)

        # Auto-discover from model dir
        model_dir = Path(self._model_dir)
        model_file = model_dir / "xgboost_freqai.joblib"
        if model_file.exists():
            try:
                import joblib
                self._model = joblib.load(str(model_file))
                logger.info("[freqai] Loaded cached XGBoost model from %s", model_file)
            except Exception as e:
                logger.warning("[freqai] Cached model load failed: %s", e)

    async def compute_score(self, symbol: str) -> BrainSignal:
        # Auto-fetch fresh OHLCV from Binance if buffer is stale
        self._auto_fetch_ohlcv(symbol)

        bars = self._ohlcv_buffer[-100:]
        if len(bars) < 30:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "insufficient_ohlcv"},
            )

        closes = np.array([b["close"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])
        volumes = np.array([b["volume"] for b in bars])

        # Compute technical indicators
        rsi = self._compute_rsi(closes, 14)
        macd_hist = self._compute_macd(closes)
        bb_pos = self._compute_bb_position(closes, 20)
        vol_score = self._compute_volume_score(volumes, 20)

        # Rule-based scoring (fallback when no XGBoost model)
        score = 0.0
        confidence = 0.0

        # RSI signal
        if rsi < 30:
            score += 0.3  # oversold → buy
            confidence += 0.2
        elif rsi > 70:
            score -= 0.3  # overbought → sell
            confidence += 0.2
        else:
            score += (50 - rsi) / 100  # mild mean reversion
            confidence += 0.05

        # MACD histogram signal
        if macd_hist > 0:
            score += min(0.2, macd_hist / closes[-1] * 1000)
        else:
            score -= min(0.2, abs(macd_hist) / closes[-1] * 1000)

        # Bollinger Band position
        if bb_pos < 0.2:
            score += 0.15  # near lower band → buy
        elif bb_pos > 0.8:
            score -= 0.15  # near upper band → sell

        # Volume confirmation
        confidence += vol_score * 0.2

        # Clamp and normalize
        score = float(np.clip(score, -1.0, 1.0))
        confidence = float(np.clip(confidence, 0.1, 0.95))

        # Try XGBoost model if available
        if self._model is not None:
            try:
                features = self._extract_features(closes, highs, lows, volumes)
                model_score = float(self._model.predict([features])[0])
                # Blend rule-based and model: 60% model, 40% rules
                score = 0.6 * model_score + 0.4 * score
                confidence = min(0.95, confidence + 0.15)
            except Exception as e:
                logger.warning("[freqai] Model prediction failed: %s", e)
        elif len(self._ohlcv_buffer) >= 100:
            # Auto-train XGBoost if we have enough data
            self._auto_train(closes, highs, lows, volumes)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=score,
            confidence=confidence,
            metadata={
                "rsi": round(rsi, 2),
                "macd_hist": round(float(macd_hist), 4),
                "bb_pos": round(bb_pos, 3),
                "model_used": self._model is not None,
            },
        )

    def _auto_fetch_ohlcv(self, symbol: str) -> None:
        """Fetch fresh OHLCV from Binance via ccxt if buffer is empty or stale.

        Falls back to yfinance if Binance fails (e.g. API rate-limit, geo-block).
        """
        now = time.time()
        last_fetch = self._binance_fetched.get(symbol, 0)
        # Fetch if never fetched or older than 5 minutes
        if now - last_fetch < 300 and len(self._ohlcv_buffer) >= 50:
            return

        # ── Primary: Binance via ccxt ───────────────────────────────
        try:
            import ccxt
            exchange = ccxt.binance()
            ccxt_symbol = symbol.replace("/", "/")  # already in correct format
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, "1h", limit=200)
            if ohlcv and len(ohlcv) >= 30:
                # Clear and refill buffer
                self._ohlcv_buffer.clear()
                for row in ohlcv:
                    self._ohlcv_buffer.append({
                        "timestamp": row[0],
                        "open": row[1],
                        "high": row[2],
                        "low": row[3],
                        "close": row[4],
                        "volume": row[5],
                    })
                self._binance_fetched[symbol] = now
                logger.info("[freqai] Fetched %d OHLCV bars for %s from Binance",
                            len(ohlcv), symbol)
                return
            logger.warning("[freqai] Binance returned insufficient data (%d bars)", len(ohlcv) if ohlcv else 0)
        except Exception as e:
            logger.warning("[freqai] Binance fetch failed: %s", e)

        # ── Fallback: yfinance ─────────────────────────────────────
        self._fetch_yfinance(symbol, now)

    def _fetch_yfinance(self, symbol: str, now: float) -> None:
        """Fetch OHLCV from yfinance as Binance fallback.

        Converts ccxt-style 'BTC/USDT' to yfinance-style 'BTC-USD'.
        Only used when Binance ccxt fetch fails.
        """
        try:
            import yfinance as yf

            # Map ccxt pair format → yfinance ticker format
            # e.g. "BTC/USDT" → "BTC-USD", "ETH/USDT" → "ETH-USD"
            base = symbol.split("/")[0] if "/" in symbol else symbol
            quote = symbol.split("/")[1] if "/" in symbol else "USDT"
            yf_suffix = "USD" if quote == "USDT" else quote
            yf_ticker = f"{base}-{yf_suffix}"

            # Download 200 days of daily data (yfinance doesn't give 1h reliably)
            tk = yf.Ticker(yf_ticker)
            hist = tk.history(period="200d", interval="1d", auto_adjust=True)

            if hist.empty or len(hist) < 30:
                logger.warning("[freqai] yfinance returned %d rows for %s (ticker=%s)",
                              len(hist), symbol, yf_ticker)
                return

            # Convert DataFrame rows to internal dict format
            self._ohlcv_buffer.clear()
            for idx, row in hist.iterrows():
                self._ohlcv_buffer.append({
                    "timestamp": int(idx.timestamp() * 1000),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                })
            self._binance_fetched[symbol] = now
            logger.info("[freqai] Fetched %d daily bars for %s from yfinance (ticker=%s)",
                        len(hist), symbol, yf_ticker)
        except ImportError:
            logger.debug("[freqai] yfinance not installed, skipping fallback")
        except Exception as e:
            logger.warning("[freqai] yfinance fetch failed for %s: %s", symbol, e)

    def _auto_train(self, closes: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, volumes: np.ndarray) -> None:
        """Auto-train XGBoost on accumulated OHLCV data."""
        try:
            import xgboost as xgb
            import joblib
        except ImportError:
            logger.debug("[freqai] xgboost/joblib not installed, skipping auto-train")
            return

        try:
            # Build training dataset from OHLCV
            n = len(closes)
            if n < 100:
                return

            X, y = [], []
            for i in range(50, n - 1):
                c = closes[:i + 1]
                h = highs[:i + 1]
                l = lows[:i + 1]
                v = volumes[:i + 1]
                feat = self._extract_features(c, h, l, v)
                # Label: next bar return direction (1=up, 0=down)
                next_return = (closes[i + 1] - closes[i]) / closes[i]
                label = 1.0 if next_return > 0 else 0.0
                X.append(feat)
                y.append(label)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
            model.fit(X, y)

            # Persist model
            model_dir = Path(self._model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / "xgboost_freqai.joblib"
            joblib.dump(model, str(model_file))

            self._model = model
            logger.info("[freqai] Auto-trained XGBoost on %d samples, saved to %s",
                        len(X), model_file)
        except Exception as e:
            logger.warning("[freqai] Auto-train failed: %s", e)

    def push_ohlcv(self, bar: dict) -> None:
        """Push a new OHLCV bar into the buffer."""
        self._ohlcv_buffer.append(bar)
        if len(self._ohlcv_buffer) > self._max_bars:
            self._ohlcv_buffer = self._ohlcv_buffer[-self._max_bars:]

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes[-period - 1 :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _compute_macd(self, closes: np.ndarray) -> float:
        if len(closes) < 26:
            return 0.0
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line * 0.2  # simplified signal
        return float(macd_line - signal_line)

    def _compute_bb_position(self, closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.5
        window = closes[-period:]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            return 0.5
        return float((closes[-1] - (mean - 2 * std)) / (4 * std))

    def _compute_volume_score(self, volumes: np.ndarray, period: int = 20) -> float:
        if len(volumes) < period:
            return 0.3
        avg_vol = np.mean(volumes[-period:])
        recent_vol = volumes[-1]
        if avg_vol == 0:
            return 0.3
        ratio = recent_vol / avg_vol
        return float(min(1.0, ratio / 3))

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        multiplier = 2 / (period + 1)
        ema = float(np.mean(data[:period]))
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _extract_features(self, closes, highs, lows, volumes) -> list[float]:
        """Extract feature vector for XGBoost model."""
        rsi = self._compute_rsi(closes, 14)
        macd = self._compute_macd(closes)
        bb = self._compute_bb_position(closes, 20)
        vol_score = self._compute_volume_score(volumes, 20)
        roc5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0.0
        roc10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 else 0.0
        hl_spread = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0.0
        return [rsi / 100, macd, bb, vol_score, roc5, roc10, hl_spread]
