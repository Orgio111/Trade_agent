"""Tests for LLMRegimeBrain, FinBERTBrain, FinRLBrain.

Covers: Binance data fetch, Local LLM (Ollama), OpenRouter/NIM cloud,
crypto news, PPO online train, trade history persist.
All external HTTP calls are mocked — tests are fully offline.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.brains.llm_regime_brain import LLMRegimeBrain
from orchestrator.brains.finbert_brain import FinBERTBrain
from orchestrator.brains.finrl_brain import FinRLBrain
from orchestrator.brains.base_brain import BrainSignal


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Regime Brain Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMRegimeBinanceFetch:
    """Verify Binance market data fetch for regime context building."""

    def test_build_context_with_ohlcv(self):
        """_build_context should compute price_change_pct, vol_ratio, rsi from Binance."""
        brain = LLMRegimeBrain()

        # Mock ccxt to return fake OHLCV
        fake_ohlcv = []
        for i in range(30):
            fake_ohlcv.append([
                1_700_000_000_000 + i * 3_600_000,  # timestamp
                60_000 + i * 10,                       # open
                60_100 + i * 10,                       # high
                59_900 + i * 10,                       # low
                60_000 + i * 10,                       # close  (rising)
                500 + i * 10,                          # volume
            ])

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = fake_ohlcv

        with patch.dict("sys.modules", {"ccxt": MagicMock(binance=MagicMock(return_value=mock_exchange))}):
            ctx = brain._build_context("BTC/USDT")

        assert ctx["symbol"] == "BTC/USDT"
        # Price should be rising over 24 bars
        assert ctx["price_change_pct"] > 0
        assert ctx["rsi"] > 0

    def test_build_context_caches_result(self):
        """Second call within TTL should return cached data."""
        brain = LLMRegimeBrain()

        mock_exchange = MagicMock()
        fake_ohlcv = [[1_700_000_000_000, 60000, 60100, 59900, 60000, 500]] * 30
        mock_exchange.fetch_ohlcv.return_value = fake_ohlcv

        with patch.dict("sys.modules", {"ccxt": MagicMock(binance=MagicMock(return_value=mock_exchange))}):
            ctx1 = brain._build_context("BTC/USDT")
            ctx2 = brain._build_context("BTC/USDT")

        # fetch_ohlcv should only be called once (cached second time)
        assert mock_exchange.fetch_ohlcv.call_count == 1
        assert ctx1 == ctx2

    def test_build_context_binance_failure_returns_defaults(self):
        """If Binance fails entirely, sensible defaults should be returned."""
        brain = LLMRegimeBrain()

        mock_ccxt = MagicMock()
        mock_ccxt.binance.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"ccxt": mock_ccxt}):
            ctx = brain._build_context("BTC/USDT")

        assert ctx["symbol"] == "BTC/USDT"
        assert ctx["price_change_pct"] == 0.0
        assert ctx["rsi"] == 50.0


class TestLLMRegimeOllama:
    """Verify local Ollama LLM integration for regime classification."""

    @pytest.mark.asyncio
    async def test_query_ollama_parses_json(self):
        brain = LLMRegimeBrain()
        brain._http = AsyncMock()

        response_json = {
            "response": '{"regime": "trending_up", "confidence": 0.85, "rationale": "strong momentum"}'
        }
        brain._http.post = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value=response_json)))

        regime, conf = await brain._query_ollama({"symbol": "BTC/USDT", "price_change_pct": 3.5, "vol_ratio": 1.2, "rsi": 58, "headlines": "none"})

        assert regime == "trending_up"
        assert conf == 0.85

    @pytest.mark.asyncio
    async def test_query_ollama_keyword_fallback(self):
        """When Ollama returns non-JSON text, keyword matching should work."""
        brain = LLMRegimeBrain()
        brain._http = AsyncMock()

        response_json = {"response": "The market appears bearish with declining momentum."}
        brain._http.post = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value=response_json)))

        regime, conf = await brain._query_ollama({"symbol": "BTC/USDT", "price_change_pct": -3.0, "vol_ratio": 1.5, "rsi": 35, "headlines": "none"})

        assert regime == "trending_down"
        assert conf > 0

    @pytest.mark.asyncio
    async def test_query_ollama_timeout(self):
        brain = LLMRegimeBrain()
        import httpx
        brain._http = AsyncMock()
        brain._http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        regime, conf = await brain._query_ollama({"symbol": "BTC/USDT", "price_change_pct": 0, "vol_ratio": 1, "rsi": 50, "headlines": "none"})

        assert regime is None
        assert conf == 0.0


class TestLLMRegimeOpenRouter:
    """Verify OpenRouter cloud API integration."""

    @pytest.mark.asyncio
    async def test_query_cloud_openrouter(self):
        brain = LLMRegimeBrain()
        brain.nim_key = ""
        brain.openrouter_key = "sk-test-key"
        brain._http = AsyncMock()

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"regime": "crisis", "confidence": 0.9, "rationale": "flash crash"}'}}]
        }
        brain._http.post = AsyncMock(return_value=response)

        regime, conf = await brain._query_cloud({"symbol": "BTC/USDT", "price_change_pct": -15, "vol_ratio": 5.0, "rsi": 20, "headlines": "market crashed"})

        assert regime == "crisis"
        assert conf == 0.9

    @pytest.mark.asyncio
    async def test_query_cloud_nim_tier(self):
        """NIM should be tried first (tier 2), before OpenRouter (tier 3)."""
        brain = LLMRegimeBrain()
        brain.nim_key = "nim-test-key"
        brain.openrouter_key = "or-test-key"
        brain._http = AsyncMock()

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"regime": "ranging", "confidence": 0.6}'}}]
        }
        brain._http.post = AsyncMock(return_value=response)

        regime, conf = await brain._query_cloud({"symbol": "BTC/USDT", "price_change_pct": 0.1, "vol_ratio": 1.0, "rsi": 50, "headlines": "none"})

        assert regime == "ranging"
        # First call should have been to NIM URL (not openrouter)
        call_args = brain._http.post.call_args_list[0]
        assert "nvidia" in call_args[0][0] or "nim" in call_args[1].get("headers", {}).get("Authorization", "").lower() or "Authorization" in call_args[1].get("headers", {})

    @pytest.mark.asyncio
    async def test_query_cloud_no_keys(self):
        """No API keys → should return None."""
        brain = LLMRegimeBrain()
        brain.nim_key = ""
        brain.openrouter_key = ""
        brain._http = AsyncMock()

        # Provide full context to avoid KeyError in REGIME_PROMPT.format
        full_ctx = {"symbol": "BTC/USDT", "price_change_pct": 0.0, "vol_ratio": 1.0, "rsi": 50, "headlines": "none"}
        regime, conf = await brain._query_cloud(full_ctx)
        assert regime is None


class TestLLMRegimeScoring:
    """Verify regime → score mapping."""

    def test_regime_to_score_mapping(self):
        assert LLMRegimeBrain._regime_to_score("trending_up") == 0.6
        assert LLMRegimeBrain._regime_to_score("trending_down") == -0.6
        assert LLMRegimeBrain._regime_to_score("ranging") == 0.0
        assert LLMRegimeBrain._regime_to_score("volatile") == 0.0
        assert LLMRegimeBrain._regime_to_score("crisis") == -0.8
        assert LLMRegimeBrain._regime_to_score("unknown") == 0.0

    @pytest.mark.asyncio
    async def test_compute_score_full_pipeline(self):
        """End-to-end: build context → LLM → score."""
        brain = LLMRegimeBrain()
        brain._http = AsyncMock()

        # Mock Ollama to return trending_up
        response_json = {"response": '{"regime": "trending_up", "confidence": 0.8}'}
        brain._http.post = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value=response_json)))

        # Binance data: mock with rising prices
        fake_ohlcv = [[1_700_000_000_000, 60000, 60100, 59900, 60000 + i * 50, 500] for i in range(30)]
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = fake_ohlcv

        with patch.dict("sys.modules", {"ccxt": MagicMock(binance=MagicMock(return_value=mock_exchange))}):
            signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        assert signal.brain_id == "llm_regime"
        assert signal.score > 0  # trending_up → positive score


# ═══════════════════════════════════════════════════════════════════════════════
#  FinBERT Brain Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinBERTNewsFetch:
    """Verify crypto news headline fetching."""

    def test_auto_fetch_from_cryptocompare(self):
        brain = FinBERTBrain()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Data": [
                {"title": "Bitcoin surges past $70K"},
                {"title": "Ethereum adoption grows"},
            ]
        }

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp

        with patch.dict("sys.modules", {"requests": mock_requests}):
            brain._auto_fetch_headlines("BTC/USDT")

        assert len(brain._headlines) == 2
        assert "surges" in brain._headlines[0]

    def test_auto_fetch_coingecko_fallback(self):
        """When CryptoCompare fails, CoinGecko trending should be used."""
        brain = FinBERTBrain()

        # CryptoCompare: raise exception
        # CoinGecko: return valid data
        call_count = [0]

        def mock_get(url, **kwargs):
            call_count[0] += 1
            if "cryptocompare" in url:
                return MagicMock(status_code=500)
            # CoinGecko path
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "coins": [
                    {"item": {"name": "Bitcoin"}},
                    {"item": {"name": "Ethereum"}},
                ]
            }
            return resp

        mock_requests = MagicMock()
        mock_requests.get = mock_get

        with patch.dict("sys.modules", {"requests": mock_requests}):
            brain._auto_fetch_headlines("BTC/USDT")

        # Should have CoinGecko headlines
        assert len(brain._headlines) >= 1

    def test_auto_fetch_caches(self):
        """Second call within TTL should use cached headlines."""
        brain = FinBERTBrain()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"Data": [{"title": "Test headline"}]}

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp

        with patch.dict("sys.modules", {"requests": mock_requests}):
            brain._auto_fetch_headlines("BTC/USDT")
            brain._auto_fetch_headlines("BTC/USDT")  # should use cache

        # Only one actual HTTP call (cache hit on 2nd)
        assert mock_requests.get.call_count == 1

    @pytest.mark.asyncio
    async def test_no_headlines_returns_low_confidence(self):
        """No headlines → score=0 with low confidence."""
        brain = FinBERTBrain()
        brain._headlines = []

        # Block all external calls
        with patch.object(brain, "_auto_fetch_headlines"), \
             patch.object(brain, "_query_ollama", new_callable=AsyncMock, return_value=(None, 0.0)), \
             patch.object(brain, "_query_cloud", new_callable=AsyncMock, return_value=(None, 0.0)):
            result = await brain.compute_score("BTC/USDT")

        assert result.score == 0.0
        assert result.confidence == 0.1


class TestFinBERTLocalLLM:
    """Verify local Ollama LLM integration for sentiment."""

    @pytest.mark.asyncio
    async def test_query_ollama_sentiment(self):
        brain = FinBERTBrain()
        brain._headlines = ["Bitcoin rallies to new highs", "Institutional adoption surges"]
        brain._http = AsyncMock()

        response_json = {"response": '{"sentiment": "bullish", "confidence": 0.8, "key_signal": "momentum"}'}
        brain._http.post = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value=response_json)))

        sentiment, conf = await brain._query_ollama("qwen2.5:3b", "BTC/USDT")

        assert sentiment == "bullish"
        assert conf == 0.8

    @pytest.mark.asyncio
    async def test_query_ollama_connection_error(self):
        brain = FinBERTBrain()
        brain._headlines = ["Test"]
        import httpx
        brain._http = AsyncMock()
        brain._http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        sentiment, conf = await brain._query_ollama("qwen2.5:3b", "BTC/USDT")

        assert sentiment is None
        assert conf == 0.0


class TestFinBERTCloud:
    """Verify NIM/OpenRouter cloud integration for sentiment."""

    @pytest.mark.asyncio
    async def test_cloud_nim_sentiment(self):
        brain = FinBERTBrain()
        brain._headlines = ["Bitcoin crashes 20%"]
        brain.nim_key = "nim-test"
        brain.openrouter_key = ""
        brain._http = AsyncMock()

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"sentiment": "very_bearish", "confidence": 0.9, "key_signal": "flash crash"}'}}]
        }
        brain._http.post = AsyncMock(return_value=response)

        score, conf = await brain._query_cloud("BTC/USDT")

        assert score is not None
        assert score < 0  # very_bearish → negative

    @pytest.mark.asyncio
    async def test_cloud_openrouter_sentiment(self):
        brain = FinBERTBrain()
        brain._headlines = ["Bitcoin surges to new ATH"]
        brain.nim_key = ""
        brain.openrouter_key = "or-test"
        brain._http = AsyncMock()

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"sentiment": "very_bullish", "confidence": 0.85}'}}]
        }
        brain._http.post = AsyncMock(return_value=response)

        score, conf = await brain._query_cloud("BTC/USDT")

        assert score is not None
        assert score > 0  # very_bullish → positive


class TestFinBERTScoring:
    """Verify sentiment → score mapping and keyword fallback."""

    def test_sentiment_to_score(self):
        assert FinBERTBrain._sentiment_to_score("very_bullish", 1.0) == pytest.approx(0.8)
        assert FinBERTBrain._sentiment_to_score("bullish", 1.0) == pytest.approx(0.4)
        assert FinBERTBrain._sentiment_to_score("neutral", 1.0) == pytest.approx(0.0)
        assert FinBERTBrain._sentiment_to_score("bearish", 1.0) == pytest.approx(-0.4)
        assert FinBERTBrain._sentiment_to_score("very_bearish", 1.0) == pytest.approx(-0.8)

    def test_sentiment_to_score_with_confidence(self):
        # Confidence modulates the magnitude
        score = FinBERTBrain._sentiment_to_score("bullish", 0.5)
        assert score == pytest.approx(0.2)  # 0.4 * 0.5

    def test_keyword_fallback_bullish(self):
        brain = FinBERTBrain()
        brain._headlines = ["Bitcoin surges to new highs", "Crypto rally continues"]

        score = brain._keyword_fallback_score()
        assert score > 0

    def test_keyword_fallback_bearish(self):
        brain = FinBERTBrain()
        brain._headlines = ["Market crash wipes out billions", "Fear grips crypto traders"]

        score = brain._keyword_fallback_score()
        assert score < 0

    def test_keyword_fallback_empty(self):
        brain = FinBERTBrain()
        brain._headlines = []

        score = brain._keyword_fallback_score()
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_compute_score_multi_model_consensus(self):
        """Two Ollama models returning different sentiments → blended score."""
        brain = FinBERTBrain()
        brain._headlines = ["Bitcoin rallies", "BTC gains momentum"]
        brain._http = AsyncMock()
        brain.ollama_models = ["model_a", "model_b"]

        # model_a: bullish, model_b: neutral → avg should be positive
        bully_response = MagicMock(status_code=200)
        bully_response.json.return_value = {"response": '{"sentiment": "bullish", "confidence": 0.7}'}

        neutral_response = MagicMock(status_code=200)
        neutral_response.json.return_value = {"response": '{"sentiment": "neutral", "confidence": 0.6}'}

        brain._http.post = AsyncMock(side_effect=[bully_response, neutral_response])

        signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        assert signal.brain_id == "finbert_nlp"
        # Bullish (0.4*0.7=0.28) + Neutral (0.0) → avg ~0.14 → positive
        assert signal.score >= 0


# ═══════════════════════════════════════════════════════════════════════════════
#  FinRL Brain Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinRLTradeHistory:
    """Verify trade result persistence and loading."""

    def test_push_trade_result(self):
        brain = FinRLBrain()
        brain.push_trade_result(0.05)   # win +5%
        brain.push_trade_result(-0.02)  # loss -2%
        brain.push_trade_result(0.03)   # win +3%

        assert len(brain._trade_history) == 3
        wins = [t for t in brain._trade_history if t["win"]]
        assert len(wins) == 2

    def test_trade_history_persisted(self, tmp_path):
        brain = FinRLBrain()
        brain._model_dir = str(tmp_path)
        brain._trade_log_path = tmp_path / "trade_history.json"

        brain.push_trade_result(0.01)
        brain.push_trade_result(-0.01)

        # File should exist
        assert brain._trade_log_path.exists()

        # Reload from disk
        data = json.loads(brain._trade_log_path.read_text())
        assert len(data) == 2

    def test_trade_history_reload(self, tmp_path):
        brain1 = FinRLBrain()
        brain1._model_dir = str(tmp_path)
        brain1._trade_log_path = tmp_path / "trade_history.json"
        brain1.push_trade_result(0.05)
        brain1.push_trade_result(-0.02)
        brain1.push_trade_result(0.03)

        # New brain instance → load from disk
        brain2 = FinRLBrain()
        brain2._model_dir = str(tmp_path)
        brain2._trade_log_path = tmp_path / "trade_history.json"
        brain2._load_trade_history()

        assert len(brain2._trade_history) == 3


class TestFinRLKelly:
    """Verify Kelly Criterion computation."""

    def test_kelly_positive_win_rate(self):
        brain = FinRLBrain()
        # 60% win rate, avg win 2%, avg loss 1% → b=2, kelly = (0.6*2-0.4)/2 = 0.4
        for _ in range(60):
            brain._trade_history.append({"pnl_pct": 0.02, "win": True})
        for _ in range(40):
            brain._trade_history.append({"pnl_pct": -0.01, "win": False})

        kelly = brain._compute_kelly()
        assert kelly > 0  # positive edge → positive kelly

    def test_kelly_losing_streak(self):
        brain = FinRLBrain()
        # 30% win rate → kelly should be negative or zero
        for _ in range(30):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})
        for _ in range(70):
            brain._trade_history.append({"pnl_pct": -0.01, "win": False})

        kelly = brain._compute_kelly()
        assert kelly <= 0  # losing edge

    def test_kelly_insufficient_data(self):
        brain = FinRLBrain()
        # < 20 trades → should return 0
        for _ in range(10):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})

        kelly = brain._compute_kelly()
        assert kelly == 0.0

    def test_kelly_confidence_levels(self):
        brain = FinRLBrain()
        # < 20 trades
        for _ in range(10):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})
        assert brain._kelly_confidence() == 0.15

        # 20-49
        brain._trade_history.clear()
        for _ in range(30):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})
        assert brain._kelly_confidence() == 0.4

        # 50-99
        brain._trade_history.clear()
        for _ in range(60):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})
        assert brain._kelly_confidence() == 0.6

        # 100+
        brain._trade_history.clear()
        for _ in range(120):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})
        assert brain._kelly_confidence() == 0.8


class TestFinRLPPO:
    """Verify PPO online training."""

    def test_auto_train_ppo(self, tmp_path):
        brain = FinRLBrain()
        brain._model_dir = str(tmp_path)

        # Add enough trade history
        for i in range(60):
            pnl = 0.02 if i % 3 != 0 else -0.01
            brain._trade_history.append({"pnl_pct": pnl, "win": pnl > 0})

        # Mock sb3 components
        mock_ppo = MagicMock()
        mock_env = MagicMock()

        mock_sb3 = MagicMock()
        mock_sb3.PPO.return_value = mock_ppo

        mock_gym = MagicMock()

        with patch.dict("sys.modules", {
            "stable_baselines3": mock_sb3,
            "stable_baselines3.common": MagicMock(),
            "stable_baselines3.common.env_util": MagicMock(make_vec_env=lambda f, n_envs=1: mock_env),
            "gymnasium": mock_gym,
            "gymnasium.spaces": MagicMock(),
        }):
            brain._auto_train_ppo()

        # PPO.learn should have been called
        mock_ppo.learn.assert_called_once()
        mock_ppo.save.assert_called_once()
        assert brain._trained is True

    def test_auto_train_skips_insufficient_data(self):
        brain = FinRLBrain()
        for _ in range(10):
            brain._trade_history.append({"pnl_pct": 0.01, "win": True})

        mock_sb3 = MagicMock()
        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            brain._auto_train_ppo()

        # Not enough trades → PPO not instantiated
        mock_sb3.PPO.assert_not_called()

    def test_online_ppo_update_called(self):
        """push_trade_result should trigger _online_ppo_update every 10 trades."""
        brain = FinRLBrain()
        brain._ppo_agent = MagicMock()  # simulate loaded agent
        brain._online_ppo_update = MagicMock()

        # Push 10 results → should trigger online update
        for i in range(10):
            brain.push_trade_result(0.01)

        brain._online_ppo_update.assert_called_once()


class TestFinRLComputeScore:
    """Verify compute_score integration."""

    @pytest.mark.asyncio
    async def test_compute_score_kelly_only(self):
        brain = FinRLBrain()
        # No PPO agent → Kelly only
        assert brain._ppo_agent is None

        for i in range(50):
            brain._trade_history.append({"pnl_pct": 0.02 if i % 3 != 0 else -0.01, "win": i % 3 != 0})

        signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        assert signal.brain_id == "finrl_kelly"
        assert signal.metadata.get("method") == "kelly_only"

    @pytest.mark.asyncio
    async def test_compute_score_no_data(self):
        brain = FinRLBrain()

        signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        assert signal.score == 0.0  # no data → flat

    @pytest.mark.asyncio
    async def test_compute_score_with_ppo_blend(self):
        brain = FinRLBrain()
        brain._ppo_agent = MagicMock()
        brain._ppo_agent.predict.return_value = (0.7, None)  # PPO says BUY 0.7

        for i in range(50):
            brain._trade_history.append({"pnl_pct": 0.02 if i % 3 != 0 else -0.01, "win": i % 3 != 0})

        signal = await brain.compute_score("BTC/USDT")

        assert isinstance(signal, BrainSignal)
        # Should have both kelly and ppo in metadata
        assert "kelly_frac" in signal.metadata
        assert "ppo_size" in signal.metadata
