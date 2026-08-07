"""Brain #2: FreqAI / XGBoost Technical Signals.

Uses frequency-domain features + XGBoost for short-term price prediction.
Weight in Go orchestrator: 0.15.

Data pipeline:
  1. Primary: Binance OHLCV via ccxt (async, non-blocking)
  2. Fallback: yfinance daily bars
  3. Explicit research-only XGBoost candidate training; never auto-activated
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
import shutil
import tempfile

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


# ── Import guards (cached at module level) ─────────────
def _has_ccxt() -> bool:
    try:
        import ccxt  # noqa: F401

        return True
    except ImportError:
        return False


def _has_yfinance() -> bool:
    try:
        import yfinance  # noqa: F401

        return True
    except ImportError:
        return False


def _has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


def _has_joblib() -> bool:
    try:
        import joblib  # noqa: F401

        return True
    except ImportError:
        return False


_CCXT_AVAILABLE: bool = _has_ccxt()
_YFINANCE_AVAILABLE: bool = _has_yfinance()
_XGBOOST_AVAILABLE: bool = _has_xgboost()
_JOBLIB_AVAILABLE: bool = _has_joblib()


class FreqAIBrain(BaseBrain):
    """FreqAI-inspired technical indicator brain.

    Computes RSI, MACD, Bollinger Band position, and volume profile
    from recent OHLCV data. Uses either a pre-trained XGBoost model
    or a rule-based scoring system as fallback.

    Features:
      - Auto-fetches OHLCV from Binance via ccxt (non-blocking async)
      - Keeps online training disabled; certification owns model promotion
      - Can stage an explicit unvalidated research candidate outside models/
      - Retry logic with exponential backoff on data fetch failures
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
        self._candidate_dir = os.getenv(
            "FREQAI_CANDIDATE_DIR",
            str(
                Path(__file__).resolve().parents[2]
                / ".local"
                / "alpha-certification"
                / "legacy-candidates"
                / "freqai"
            ),
        )
        self._last_candidate_path: str | None = None
        self._binance_fetched: dict[str, float] = {}  # symbol → last fetch timestamp
        self._fetch_retries: dict[str, int] = {}  # symbol → consecutive failures
        self._data_source: str = (
            "none"  # explicit flag: "binance" | "yfinance" | "none"
        )
        self._exchange = None  # cached ccxt.binance instance

    async def warmup(self) -> None:
        """Attempt to load pre-trained XGBoost model, or auto-discover."""
        if not _JOBLIB_AVAILABLE:
            logger.warning("[freqai] joblib not installed, will train inline")
            return

        # If explicit path set and exists, load it
        if self._model_path and os.path.exists(self._model_path):
            try:
                import joblib

                self._model = joblib.load(self._model_path)
                logger.info("[freqai] Loaded XGBoost model from %s", self._model_path)
                return
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
        # Auto-fetch fresh OHLCV from Binance (non-blocking)
        await self._auto_fetch_ohlcv(symbol)

        bars = self._ohlcv_buffer[-100:]
        if len(bars) < 30:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "insufficient_ohlcv", "bars": len(bars)},
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
        model_used = False
        if self._model is not None:
            try:
                features = self._extract_features(closes, highs, lows, volumes)
                model_score = float(self._model.predict([features])[0])
                # Blend rule-based and model: 60% model, 40% rules
                score = 0.6 * model_score + 0.4 * score
                confidence = min(0.95, confidence + 0.15)
                model_used = True
            except Exception as e:
                logger.warning("[freqai] Model prediction failed: %s", e)
        # Online fitting is deliberately disabled. New models must be trained
        # and promoted through the offline alpha-certification pipeline.

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=score,
            confidence=confidence,
            metadata={
                "rsi": round(rsi, 2),
                "macd_hist": round(float(macd_hist), 4),
                "bb_pos": round(bb_pos, 3),
                "vol_score": round(vol_score, 3),
                "model_used": model_used,
                "bars": len(bars),
                "source": self._get_data_source(),
            },
        )

    def _get_data_source(self) -> str:
        """Return which data source is currently loaded."""
        return self._data_source if self._ohlcv_buffer else "none"

    async def _auto_fetch_ohlcv(self, symbol: str) -> None:
        """Fetch fresh OHLCV from Binance via ccxt (non-blocking async).

        Falls back to yfinance if Binance fails (e.g. API rate-limit, geo-block).
        Uses retry logic with exponential backoff.
        """
        now = time.time()
        last_fetch = self._binance_fetched.get(symbol, 0)
        retries = self._fetch_retries.get(symbol, 0)

        # Fetch if never fetched, or older than 5 minutes, or buffer empty
        if now - last_fetch < 300 and len(self._ohlcv_buffer) >= 50:
            return

        # Skip Binance after 3+ consecutive failures (use yfinance instead)
        if retries >= 3:
            logger.debug("[freqai] Binance skip (retries=%d), trying yfinance", retries)
            await self._fetch_yfinance(symbol, now)
            return

        # ── Primary: Binance via ccxt (non-blocking) ──────────────────
        if _CCXT_AVAILABLE:
            try:
                # Run blocking ccxt call in thread pool to avoid blocking event loop
                ohlcv = await asyncio.wait_for(
                    asyncio.to_thread(self._fetch_binance_ohlcv, symbol),
                    timeout=15.0,
                )
                if ohlcv and len(ohlcv) >= 30:
                    self._ohlcv_buffer.clear()
                    for row in ohlcv:
                        self._ohlcv_buffer.append(
                            {
                                "timestamp": row[0],
                                "open": row[1],
                                "high": row[2],
                                "low": row[3],
                                "close": row[4],
                                "volume": row[5],
                            }
                        )
                    self._binance_fetched[symbol] = now
                    self._fetch_retries[symbol] = 0  # Reset on success
                    self._data_source = "binance"
                    logger.info(
                        "[freqai] Fetched %d OHLCV bars for %s from Binance",
                        len(ohlcv),
                        symbol,
                    )
                    return
                logger.warning(
                    "[freqai] Binance returned insufficient data (%d bars)",
                    len(ohlcv) if ohlcv else 0,
                )
            except asyncio.TimeoutError:
                self._fetch_retries[symbol] = retries + 1
                logger.warning(
                    "[freqai] Binance fetch timed out for %s (retries=%d)",
                    symbol,
                    self._fetch_retries[symbol],
                )
            except Exception as e:
                self._fetch_retries[symbol] = retries + 1
                logger.warning(
                    "[freqai] Binance fetch failed: %s (retries=%d)",
                    e,
                    self._fetch_retries[symbol],
                )
        else:
            logger.debug("[freqai] ccxt not installed, skipping Binance")

        # ── Fallback: yfinance ─────────────────────────────────────
        await self._fetch_yfinance(symbol, now)

    def _fetch_binance_ohlcv(self, symbol: str) -> list[list]:
        """Synchronous Binance OHLCV fetch via ccxt (called in thread pool)."""
        import ccxt

        if self._exchange is None:
            self._exchange = ccxt.binance({"enableRateLimit": True})
        ccxt_symbol = symbol.replace("/", "/")
        return self._exchange.fetch_ohlcv(ccxt_symbol, "1h", limit=200)

    async def _fetch_yfinance(self, symbol: str, now: float) -> None:
        """Fetch OHLCV from yfinance as Binance fallback.

        Converts ccxt-style 'BTC/USDT' to yfinance-style 'BTC-USD'.
        Only used when Binance ccxt fetch fails.
        """
        if not _YFINANCE_AVAILABLE:
            logger.debug("[freqai] yfinance not installed, skipping fallback")
            return

        try:
            # Run blocking yfinance call in thread pool
            hist = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_yfinance_sync, symbol),
                timeout=20.0,
            )

            if hist is None or hist.empty or len(hist) < 30:
                logger.warning(
                    "[freqai] yfinance returned insufficient data for %s", symbol
                )
                return

            # Convert DataFrame rows to internal dict format
            self._ohlcv_buffer.clear()
            for idx, row in hist.iterrows():
                self._ohlcv_buffer.append(
                    {
                        "timestamp": int(idx.timestamp() * 1000),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                    }
                )
            self._binance_fetched[symbol] = now
            self._fetch_retries[symbol] = 0
            self._data_source = "yfinance"
            logger.info(
                "[freqai] Fetched %d daily bars for %s from yfinance", len(hist), symbol
            )
        except asyncio.TimeoutError:
            logger.warning("[freqai] yfinance fetch timed out for %s", symbol)
        except ImportError:
            logger.debug("[freqai] yfinance not installed, skipping fallback")
        except Exception as e:
            logger.warning("[freqai] yfinance fetch failed for %s: %s", symbol, e)

    def _fetch_yfinance_sync(self, symbol: str):
        """Synchronous yfinance fetch (called in thread pool)."""
        import yfinance as yf

        # Map ccxt pair format → yfinance ticker format
        # e.g. "BTC/USDT" → "BTC-USD", "ETH/USDT" → "ETH-USD"
        base = symbol.split("/")[0] if "/" in symbol else symbol
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        yf_suffix = "USD" if quote == "USDT" else quote
        yf_ticker = f"{base}-{yf_suffix}"

        tk = yf.Ticker(yf_ticker)
        return tk.history(period="200d", interval="1d", auto_adjust=True)

    def _auto_train(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> None:
        """Train an unvalidated candidate without changing the active model."""
        if not _XGBOOST_AVAILABLE or not _JOBLIB_AVAILABLE:
            logger.debug("[freqai] xgboost/joblib not installed, skipping auto-train")
            return

        try:
            import xgboost as xgb
            import joblib
        except ImportError:
            return

        try:
            # Build training dataset from OHLCV
            n = len(closes)
            if n < 100:
                return

            X, y = [], []
            for i in range(50, n - 1):
                c = closes[: i + 1]
                h = highs[: i + 1]
                low_window = lows[: i + 1]
                volume_window = volumes[: i + 1]
                feat = self._extract_features(c, h, low_window, volume_window)
                # Label: next bar return direction (1=up, 0=down)
                next_return = (closes[i + 1] - closes[i]) / closes[i]
                label = 1.0 if next_return > 0 else 0.0
                X.append(feat)
                y.append(label)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # Compute class weights for imbalanced data
            n_pos = int(np.sum(y))
            n_neg = len(y) - n_pos
            scale_pos_weight = n_neg / max(n_pos, 1)

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                verbosity=0,
            )
            model.fit(X, y)

            # Evaluate on last 20% of data
            split = int(len(X) * 0.8)
            X_test, y_test = X[split:], y[split:]
            if len(X_test) > 0:
                accuracy = float(np.mean(model.predict(X_test) == y_test))
                logger.info(
                    "[freqai] XGBoost accuracy on holdout: %.2f%%", accuracy * 100
                )

            candidate_root = Path(self._candidate_dir).resolve()
            repository_models = (
                Path(__file__).resolve().parents[2] / "models"
            ).resolve()
            if (
                candidate_root == repository_models
                or repository_models in candidate_root.parents
            ):
                raise ValueError(
                    "FreqAI candidate training cannot target active models"
                )
            candidate_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="freqai-candidate-",
                dir=str(candidate_root),
            ) as temporary:
                temporary_path = Path(temporary) / "candidate.joblib"
                joblib.dump(model, temporary_path, compress=3)
                digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
                destination = candidate_root / digest / "candidate.joblib"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                        raise FileExistsError("FreqAI immutable candidate collision")
                else:
                    with (
                        temporary_path.open("rb") as source,
                        destination.open("xb") as target,
                    ):
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            self._last_candidate_path = str(destination)
            logger.info(
                "[freqai] Trained unvalidated candidate on %d samples "
                "(pos_ratio=%.2f), staged digest=%s; active model unchanged",
                len(X),
                n_pos / len(y),
                digest,
            )
        except Exception as e:
            logger.warning("[freqai] Auto-train failed: %s", e)

    def push_ohlcv(self, bar: dict) -> None:
        """Push a new OHLCV bar into the buffer."""
        self._ohlcv_buffer.append(bar)
        if len(self._ohlcv_buffer) > self._max_bars:
            self._ohlcv_buffer = self._ohlcv_buffer[-self._max_bars :]

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes[-period - 1 :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-10
        if avg_loss < 1e-10:
            return 100.0 if avg_gain > 0 else 50.0
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
