"""Tests for FreqAIBrain: Binance OHLCV, Yfinance fallback, XGBoost auto-train.

All external calls (ccxt, yfinance, xgboost) are mocked so tests run offline &
deterministically.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.brains.freqai_brain import FreqAIBrain
from orchestrator.brains.base_brain import BrainSignal


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_fake_ohlcv(n: int = 100) -> list[list]:
    """Generate n fake OHLCV rows like ccxt fetch_ohlcv returns."""
    base_price = 60_000.0
    rows: list[list] = []
    for i in range(n):
        ts = 1_700_000_000_000 + i * 3_600_000  # 1h bars
        o = base_price + np.random.randn() * 200
        h = o + abs(np.random.randn()) * 300
        l = o - abs(np.random.randn()) * 300
        c = o + np.random.randn() * 200
        v = abs(np.random.randn()) * 1_000 + 500
        rows.append([ts, o, h, l, c, v])
    return rows


def _make_ohlcv_dicts(n: int = 100) -> list[dict]:
    """Generate n fake OHLCV dicts (internal buffer format)."""
    rows = _make_fake_ohlcv(n)
    return [
        {"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
        for r in rows
    ]


def _make_yf_dataframe(n: int = 100) -> MagicMock:
    """Generate a mock yfinance DataFrame with n rows."""
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {
        "Open": np.random.randn(n) * 200 + 60_000,
        "High": np.random.randn(n) * 200 + 60_500,
        "Low": np.random.randn(n) * 200 + 59_500,
        "Close": np.random.randn(n) * 200 + 60_000,
        "Volume": np.abs(np.random.randn(n)) * 1_000 + 500,
    }
    return pd.DataFrame(data, index=dates)


# ── Test: Binance OHLCV fetch ─────────────────────────────────────────────────

class TestBinanceFetch:
    """Verify ccxt Binance OHLCV fetch populates the buffer."""

    @pytest.mark.asyncio
    async def test_binance_fetch_fills_buffer(self):
        brain = FreqAIBrain()
        fake_ohlcv = _make_fake_ohlcv(150)

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = fake_ohlcv

        with patch("orchestrator.brains.freqai_brain.ccxt", create=True) as mock_ccxt_mod:
            # ccxt.binance() returns our mock
            mock_ccxt_mod.binance.return_value = mock_exchange
            # We also need `import ccxt` inside _auto_fetch_ohlcv to resolve
            with patch.dict("sys.modules", {"ccxt": mock_ccxt_mod}):
                brain._auto_fetch_ohlcv("BTC/USDT")

        # Buffer should have entries from our mock
        assert len(brain._ohlcv_buffer) == 150
        first = brain._ohlcv_buffer[0]
        assert "close" in first
        assert "volume" in first

    @pytest.mark.asyncio
    async def test_binance_fetch_sets_timestamp(self):
        brain = FreqAIBrain()
        fake_ohlcv = _make_fake_ohlcv(100)

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = fake_ohlcv

        with patch.dict("sys.modules", {"ccxt": MagicMock(binance=MagicMock(return_value=mock_exchange))}):
            brain._auto_fetch_ohlcv("ETH/USDT")

        assert "BTC/USDT" not in brain._binance_fetched  # we requested ETH
        assert "ETH/USDT" in brain._binance_fetched
        assert brain._binance_fetched["ETH/USDT"] > 0

    @pytest.mark.asyncio
    async def test_binance_fetch_insufficient_data_falls_back(self):
        """If Binance returns < 30 bars, should fall back to yfinance."""
        brain = FreqAIBrain()
        fake_ohlcv = _make_fake_ohlcv(10)  # too few bars

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = fake_ohlcv

        # yfinance fallback: also return insufficient data so buffer stays small
        mock_yf_mod = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = MagicMock(empty=False, __len__=lambda self: 5)
        mock_yf_mod.Ticker.return_value = mock_ticker

        import pandas as pd
        tiny_df = pd.DataFrame({"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]}, index=pd.date_range("2024-01-01", periods=1))
        mock_ticker.history.return_value = tiny_df  # only 1 row, < 30

        with patch.dict("sys.modules", {"ccxt": MagicMock(binance=MagicMock(return_value=mock_exchange)), "yfinance": mock_yf_mod}):
            brain._auto_fetch_ohlcv("BTC/USDT")

        # Binance insufficient + yfinance insufficient → buffer < 30
        assert len(brain._ohlcv_buffer) < 30


# ── Test: Yfinance fallback ────────────────────────────────────────────────────

class TestYfinanceFallback:
    """Verify yfinance fallback when Binance fails."""

    def test_yfinance_fills_buffer(self):
        brain = FreqAIBrain()
        fake_df = _make_yf_dataframe(100)

        mock_yf_mod = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df
        mock_yf_mod.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf_mod}):
            brain._fetch_yfinance("BTC/USDT", 1_700_000_000.0)

        assert len(brain._ohlcv_buffer) == 100
        first = brain._ohlcv_buffer[0]
        assert "close" in first
        assert "volume" in first

    def test_yfinance_ticker_format_conversion(self):
        """ccxt 'BTC/USDT' → yfinance 'BTC-USD', 'ETH/BUSD' → 'ETH-BUSD'."""
        brain = FreqAIBrain()
        captured_tickers: list[str] = []

        mock_yf_mod = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_dataframe(50)
        mock_yf_mod.Ticker = lambda ticker: (captured_tickers.append(ticker), mock_ticker)[1]

        with patch.dict("sys.modules", {"yfinance": mock_yf_mod}):
            brain._fetch_yfinance("BTC/USDT", 1_700_000_000.0)
            brain._fetch_yfinance("ETH/BUSD", 1_700_000_000.0)

        assert "BTC-USD" in captured_tickers
        assert "ETH-BUSD" in captured_tickers

    def test_yfinance_empty_data_no_crash(self):
        """Empty DataFrame from yfinance should not crash."""
        brain = FreqAIBrain()

        mock_yf_mod = MagicMock()
        mock_ticker = MagicMock()
        import pandas as pd
        mock_ticker.history.return_value = pd.DataFrame()  # empty
        mock_yf_mod.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf_mod}):
            brain._fetch_yfinance("BTC/USDT", 1_700_000_000.0)

        assert len(brain._ohlcv_buffer) == 0

    def test_yfinance_not_installed(self):
        """If yfinance is not importable, should log and skip gracefully."""
        brain = FreqAIBrain()
        # Remove yfinance from importable modules
        with patch.dict("sys.modules", {"yfinance": None}):
            brain._fetch_yfinance("BTC/USDT", 1_700_000_000.0)
        # Should not crash, buffer stays empty
        assert len(brain._ohlcv_buffer) == 0


# ── Test: XGBoost auto-train ──────────────────────────────────────────────────

class TestXGBoostTrain:
    """Verify XGBoost auto-train from accumulated OHLCV data."""

    def test_auto_train_creates_model(self, tmp_path):
        brain = FreqAIBrain()
        brain._model_dir = str(tmp_path)
        brain._model = None

        # Generate enough data for training (need 100+ bars)
        n = 120
        base_prices = np.cumsum(np.random.randn(n) * 0.01) + 60_000
        closes = base_prices
        highs = closes + np.abs(np.random.randn(n)) * 100
        lows = closes - np.abs(np.random.randn(n)) * 100
        volumes = np.abs(np.random.randn(n)) * 1000 + 500

        mock_xgb = MagicMock()
        mock_model = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_model
        mock_joblib = MagicMock()

        xgb_mod = MagicMock()
        xgb_mod.XGBClassifier = mock_xgb.XGBClassifier

        with patch.dict("sys.modules", {"xgboost": xgb_mod, "joblib": mock_joblib}):
            brain._auto_train(closes, highs, lows, volumes)

        # XGBClassifier.fit should have been called
        mock_model.fit.assert_called_once()
        # joblib.dump should have been called to persist model
        mock_joblib.dump.assert_called_once()
        # Model should be set on brain
        assert brain._model is mock_model

    def test_auto_train_skips_with_insufficient_data(self):
        brain = FreqAIBrain()
        brain._model = None

        n = 50  # < 100, so auto-train should skip
        closes = np.random.randn(n) + 60_000
        highs = closes + 100
        lows = closes - 100
        volumes = np.abs(np.random.randn(n)) * 1000

        mock_xgb = MagicMock()
        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            brain._auto_train(closes, highs, lows, volumes)

        # Not enough data → XGBClassifier should NOT be instantiated
        mock_xgb.XGBClassifier.assert_not_called()

    def test_auto_train_handles_import_error(self):
        brain = FreqAIBrain()
        brain._model = None

        n = 120
        closes = np.random.randn(n) + 60_000
        highs = closes + 100
        lows = closes - 100
        volumes = np.abs(np.random.randn(n)) * 1000

        # xgboost not importable
        with patch.dict("sys.modules", {"xgboost": None}):
            brain._auto_train(closes, highs, lows, volumes)

        # Should not crash, model stays None
        assert brain._model is None


# ── Test: compute_score with buffer ───────────────────────────────────────────

class TestComputeScore:
    """Verify compute_score produces valid signals with populated buffer."""

    @pytest.mark.asyncio
    async def test_compute_score_returns_signal(self):
        brain = FreqAIBrain()
        brain._ohlcv_buffer = _make_ohlcv_dicts(100)

        # Prevent live fetch inside compute_score
        with patch.object(brain, "_auto_fetch_ohlcv"):
            signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        assert signal.brain_id == "freqai"
        assert signal.symbol == "BTC/USDT"
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_compute_score_insufficient_data(self):
        brain = FreqAIBrain()
        # Clear any leftover buffer and explicitly set too-few bars
        brain._ohlcv_buffer.clear()
        brain._ohlcv_buffer = _make_ohlcv_dicts(10)  # < 30

        # Prevent live fetch inside compute_score
        with patch.object(brain, "_auto_fetch_ohlcv"):
            signal = await brain.compute_score("BTC/USDT")

        assert signal.score == 0.0
        assert signal.confidence == 0.1
        assert signal.metadata.get("reason") == "insufficient_ohlcv"

    @pytest.mark.asyncio
    async def test_compute_score_with_model_blend(self):
        brain = FreqAIBrain()
        brain._ohlcv_buffer = _make_ohlcv_dicts(100)

        # Mock a trained model that always predicts +0.5
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        brain._model = mock_model

        # Prevent live fetch inside compute_score
        with patch.object(brain, "_auto_fetch_ohlcv"):
            signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        assert signal.metadata.get("model_used") is True
        # Score should be blended (0.6 * model + 0.4 * rules)
        # Even if not exact, it should be within bounds
        assert -1.0 <= signal.score <= 1.0


# ── Test: Technical indicators ─────────────────────────────────────────────────

class TestIndicators:
    """Validate internal indicator calculations."""

    def test_rsi_oversold(self):
        brain = FreqAIBrain()
        # 14 consecutive drops → extreme oversold
        closes = np.array([100.0 - i for i in range(16)], dtype=float)
        rsi = brain._compute_rsi(closes, 14)
        assert rsi < 30

    def test_rsi_overbought(self):
        brain = FreqAIBrain()
        # 14 consecutive rises → extreme overbought
        closes = np.array([100.0 + i for i in range(16)], dtype=float)
        rsi = brain._compute_rsi(closes, 14)
        assert rsi > 70

    def test_bb_position_bounded(self):
        brain = FreqAIBrain()
        closes = np.random.randn(30) + 60_000
        bb = brain._compute_bb_position(closes, 20)
        assert 0.0 <= bb <= 1.0

    def test_volume_score_bounded(self):
        brain = FreqAIBrain()
        volumes = np.abs(np.random.randn(30)) * 1000
        vs = brain._compute_volume_score(volumes, 20)
        assert 0.0 <= vs <= 1.0

    def test_extract_features_length(self):
        brain = FreqAIBrain()
        n = 20
        closes = np.random.randn(n) + 60_000
        highs = closes + 100
        lows = closes - 100
        volumes = np.abs(np.random.randn(n)) * 1000

        features = brain._extract_features(closes, highs, lows, volumes)
        assert len(features) == 7


# ── Test: Warmup / push ───────────────────────────────────────────────────────

class TestWarmup:
    """Verify warmup loads model from disk."""

    @pytest.mark.asyncio
    async def test_warmup_no_model_file(self, tmp_path):
        brain = FreqAIBrain()
        brain._model_dir = str(tmp_path)
        brain._model_path = ""

        await brain.warmup()
        # No model file → stays None
        assert brain._model is None

    @pytest.mark.asyncio
    async def test_warmup_loads_model(self, tmp_path):
        brain = FreqAIBrain()
        model_file = tmp_path / "xgboost_freqai.joblib"
        brain._model_dir = str(tmp_path)

        # Create a dummy joblib file
        mock_model = MagicMock()
        mock_joblib = MagicMock()
        mock_joblib.load.return_value = mock_model

        # Write empty file so exists() is True
        model_file.write_bytes(b"fake")

        with patch.dict("sys.modules", {"joblib": mock_joblib}):
            await brain.warmup()

        assert brain._model is mock_model


class TestPushOHLCV:
    """Verify push_ohlcv maintains buffer size."""

    def test_push_maintains_max_bars(self):
        brain = FreqAIBrain()
        brain._max_bars = 5

        for i in range(10):
            brain.push_ohlcv({"close": float(i), "high": 0, "low": 0, "volume": 0, "timestamp": 0})

        assert len(brain._ohlcv_buffer) <= 5
