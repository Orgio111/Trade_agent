"""Unit tests for the MLOps pipeline orchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mlops.pipeline import MLOpsPipeline, PipelineResult


@pytest.fixture
def mock_drift() -> MagicMock:
    detector = MagicMock()
    detector.has_reference = True
    detector.check = MagicMock()
    detector.check.return_value.drift_detected = False
    detector.check.return_value.psi_scores = {}
    detector.check.return_value.max_psi = 0.05
    detector.check.return_value.drift_features = []
    return detector


@pytest.fixture
def price_data() -> np.ndarray:
    """Generate a synthetic price series for tests."""
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, 2000)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


class TestMLOpsPipelineInit:

    def test_init(self, mock_drift):
        pipeline = MLOpsPipeline(mock_drift)
        assert pipeline._drift == mock_drift
        assert pipeline._price_data is None
        assert pipeline._feature_data == {}

    def test_init_with_data(self, mock_drift, price_data):
        features = {"rsi": np.random.randn(100)}
        pipeline = MLOpsPipeline(mock_drift, price_data, feature_data=features)
        assert pipeline._price_data is not None
        assert "rsi" in pipeline._feature_data

    def test_drift_detector_property(self, mock_drift):
        pipeline = MLOpsPipeline(mock_drift)
        assert pipeline.drift_detector == mock_drift


class TestMLOpsPipelineRun:

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self, mock_drift):
        prices = np.array([100.0, 101.0, 102.0])  # too short
        pipeline = MLOpsPipeline(mock_drift, price_data=prices)
        version = await pipeline.run()
        assert version is None

    @pytest.mark.asyncio
    async def test_no_price_data_returns_none(self, mock_drift):
        pipeline = MLOpsPipeline(mock_drift, price_data=None)
        version = await pipeline.run()
        assert version is None

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, mock_drift, price_data):
        """Test the full pipeline with all steps mocked."""
        with patch("mlops.pipeline.retrain_ppo", new_callable=AsyncMock) as mock_retrain:
            mock_retrain.return_value = "/tmp/test_model.zip"

            with patch("mlops.pipeline.evaluate_ppo_model") as mock_eval:
                mock_eval.return_value = {
                    "train_sharpe": 1.8,
                    "test_sharpe": 1.4,
                    "test_max_drawdown": 0.045,
                    "test_win_rate": 0.65,
                    "test_information_ratio": 0.72,
                    "test_calmar_ratio": 4.2,
                    "test_max_consecutive_losses": 3,
                    "train_cumulative_reward": -200.0,
                    "test_cumulative_reward": -150.0,
                    "n_episodes": 20,
                }

                with patch("mlops.pipeline.save_model") as mock_save:
                    mock_save.return_value = 1

                    with patch("mlops.model_registry.prune_models") as mock_prune:
                        mock_prune.return_value = 0

                        # Mock WFO to avoid slow scipy optimization
                        with patch.object(MLOpsPipeline, "_run_wfo", new_callable=AsyncMock) as mock_wfo:
                            from mlops.walk_forward import WFOResult
                            from datetime import datetime
                            mock_wfo.return_value = [
                                WFOResult(
                                    train_start=datetime(2024, 1, 1),
                                    train_end=datetime(2024, 6, 1),
                                    test_start=datetime(2024, 6, 1),
                                    test_end=datetime(2024, 9, 1),
                                    best_params={"lookback": 20, "threshold": 0.001},
                                    train_sharpe=1.5,
                                    test_sharpe=1.2,
                                    test_max_drawdown=0.05,
                                    test_win_rate=0.62,
                                )
                            ]

                            features = {"rsi": np.random.randn(200)}
                            pipeline = MLOpsPipeline(mock_drift, price_data, feature_data=features)
                            version = await pipeline.run()

                            assert version == 1
                            mock_retrain.assert_awaited_once()
                            mock_eval.assert_called_once()
                            mock_save.assert_called_once()
                            mock_prune.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_with_drift_check(self, mock_drift, price_data):
        """Verify drift check runs when features are provided."""
        with patch("mlops.pipeline.retrain_ppo", new_callable=AsyncMock) as mock_retrain:
            mock_retrain.return_value = "/tmp/test_model.zip"
            with patch("mlops.pipeline.evaluate_ppo_model") as mock_eval:
                mock_eval.return_value = {
                    "train_sharpe": 1.8,
                    "test_sharpe": 1.4,
                    "test_max_drawdown": 0.045,
                    "test_win_rate": 0.65,
                    "test_information_ratio": 0.72,
                    "test_calmar_ratio": 4.2,
                    "test_max_consecutive_losses": 3,
                    "train_cumulative_reward": -200.0,
                    "test_cumulative_reward": -150.0,
                    "n_episodes": 20,
                }
                with patch("mlops.pipeline.save_model") as mock_save:
                    mock_save.return_value = 2
                    with patch("mlops.model_registry.prune_models"):
                        with patch.object(MLOpsPipeline, "_run_wfo", new_callable=AsyncMock) as mock_wfo:
                            from mlops.walk_forward import WFOResult
                            from datetime import datetime
                            mock_wfo.return_value = []
                            mock_drift.has_reference = True

                            features = {"rsi": np.random.randn(500)}
                            pipeline = MLOpsPipeline(mock_drift, price_data, feature_data=features)
                            version = await pipeline.run(feature_data=features)

                            mock_drift.check.assert_called_with(features)
                            assert version == 2

    @pytest.mark.asyncio
    async def test_pipeline_exception_handling(self, mock_drift, price_data):
        """Pipeline should handle exceptions gracefully."""
        with patch.object(MLOpsPipeline, "_run_wfo", new_callable=AsyncMock) as mock_wfo:
            mock_wfo.side_effect = RuntimeError("WFO failed")

            with patch("mlops.pipeline.save_model"):
                pipeline = MLOpsPipeline(mock_drift, price_data)
                version = await pipeline.run()
                assert version is None

    @pytest.mark.asyncio
    async def test_pipeline_returns_none_on_retrain_failure(self, mock_drift, price_data):
        with patch("mlops.pipeline.retrain_ppo", new_callable=AsyncMock) as mock_retrain:
            mock_retrain.side_effect = RuntimeError("retrain failed")

            with patch.object(MLOpsPipeline, "_run_wfo", new_callable=AsyncMock) as mock_wfo:
                from mlops.walk_forward import WFOResult
                from datetime import datetime
                mock_wfo.return_value = []

                pipeline = MLOpsPipeline(mock_drift, price_data)
                version = await pipeline.run()
                assert version is None


class TestPipelineResult:

    def test_default_state(self):
        result = PipelineResult(success=False)
        assert result.success is False
        assert result.model_version is None
        assert result.error is None
        assert result.wfo_results == []
        assert result.training_duration_s == 0.0

    def test_success_state(self):
        result = PipelineResult(
            success=True,
            model_version=5,
            train_sharpe=1.8,
            test_sharpe=1.4,
            total_timesteps=500_000,
        )
        assert result.model_version == 5
        assert result.train_sharpe == 1.8


class TestSerializeWFO:

    def test_serialize_wfo_result(self):
        from mlops.walk_forward import WFOResult
        from datetime import datetime

        w = WFOResult(
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 1),
            test_start=datetime(2024, 6, 1),
            test_end=datetime(2024, 9, 1),
            best_params={"lookback": 20},
            train_sharpe=1.5,
            test_sharpe=1.2,
            test_max_drawdown=0.05,
            test_win_rate=0.62,
        )
        serialized = MLOpsPipeline._serialize_wfo(w)
        assert serialized["train_sharpe"] == 1.5
        assert serialized["test_sharpe"] == 1.2
        assert serialized["best_params"] == {"lookback": 20}
        assert "train_start" in serialized
        assert "train_end" in serialized
        assert "test_start" in serialized
        assert "test_end" in serialized
