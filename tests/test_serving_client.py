"""Unit tests for the Ray Serve PPO client."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agents.serving_client import PPOPrediction, RayServePPOClient


@pytest.fixture
def sample_obs() -> np.ndarray:
    return np.zeros(40, dtype=np.float32)


class TestPPOPrediction:

    def test_defaults(self):
        pred = PPOPrediction(action=1, source="local")
        assert pred.action == 1
        assert pred.source == "local"
        assert pred.model_version == 0
        assert pred.latency_ms == 0.0
        assert pred.deterministic is True


class TestRayServePPOClientInit:

    def test_init_defaults(self):
        client = RayServePPOClient(serve_url="", fallback_model_path="/tmp/nonexistent.zip")
        assert client._serve_url == ""
        assert client._local_model is None
        assert client._latest_registry_version == 0
        assert not client.is_remote_available

    def test_init_with_serve_url(self):
        client = RayServePPOClient(serve_url="http://ray-head:8765")
        assert client.is_remote_available


class TestRayServePPOPredict:

    @pytest.mark.asyncio
    async def test_fallback_when_no_model_and_no_serve(self, sample_obs):
        client = RayServePPOClient(serve_url="", fallback_model_path="/tmp/nonexistent.zip")
        await client.start()
        pred = await client.predict(sample_obs)
        assert pred.action == 0
        assert pred.source == "fallback"
        assert pred.latency_ms >= 0.0
        await client.stop()

    @pytest.mark.asyncio
    async def test_remote_predict_success(self, sample_obs):
        """Mock _remote_predict at the client level to avoid aiohttp dependency."""
        client = RayServePPOClient(serve_url="http://ray-head:8765", fallback_model_path="/tmp/nonexistent.zip")
        await client.start()

        async def mock_remote(obs_list, deterministic, timeout_s):
            return (2, 5)  # (action, version)

        with patch.object(client, "_remote_predict", mock_remote):
            pred = await client.predict(sample_obs, timeout_s=10.0)

        assert pred.action == 2
        assert pred.source == "ray_serve"
        assert pred.model_version == 5
        assert pred.latency_ms >= 0.0

        await client.stop()

    @pytest.mark.asyncio
    async def test_remote_failure_falls_to_local(self, sample_obs):
        client = RayServePPOClient(serve_url="http://ray-head:8765", fallback_model_path="/tmp/nonexistent.zip")

        # Inject a mock local model
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([3]), None)
        client._local_model = mock_model
        client._local_version = 2

        await client.start()

        # Make remote call fail by mocking _remote_predict to raise
        async def mock_fail(*args, **kwargs):
            raise RuntimeError("connection refused")

        with patch.object(client, "_remote_predict", mock_fail):
            pred = await client.predict(sample_obs)

        assert pred.action == 3
        assert pred.source == "local"
        assert pred.model_version == 2

        await client.stop()

    @pytest.mark.asyncio
    async def test_remote_and_local_failure_falls_to_fallback(self, sample_obs):
        client = RayServePPOClient(serve_url="http://ray-head:8765", fallback_model_path="/tmp/nonexistent.zip")
        await client.start()

        # Make remote call fail
        async def mock_fail(*args, **kwargs):
            raise RuntimeError("connection refused")

        with patch.object(client, "_remote_predict", mock_fail):
            pred = await client.predict(sample_obs)

        assert pred.action == 0
        assert pred.source == "fallback"

        await client.stop()


class TestLocalModelLoading:

    @pytest.mark.asyncio
    async def test_load_local_from_registry(self, monkeypatch):
        """Verify registry lookup is attempted first."""
        client = RayServePPOClient(serve_url="", fallback_model_path="/tmp/nonexistent.zip")

        with patch("mlops.model_registry.get_model") as mock_get:
            mock_get.return_value = (42, "/tmp/ppo_v0042.zip")

            with patch.object(RayServePPOClient, "_load_sb3") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                loaded = await client._try_load_local()
                assert loaded is True
                assert client._local_model == mock_model
                assert client._local_version == 42
                assert client._latest_registry_version == 42

    @pytest.mark.asyncio
    async def test_registry_failure_falls_to_path(self, monkeypatch):
        client = RayServePPOClient(serve_url="", fallback_model_path="/tmp/existing_model.zip")

        with patch("mlops.model_registry.get_model") as mock_get:
            mock_get.return_value = None

            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True

                with patch.object(RayServePPOClient, "_load_sb3") as mock_load:
                    mock_model = MagicMock()
                    mock_load.return_value = mock_model

                    loaded = await client._try_load_local()
                    assert loaded is True
                    assert client._local_model == mock_model
                    assert client._local_version == 0

    def test_load_sb3_success(self):
        """Test that _load_sb3 returns a model when SB3 import succeeds."""
        import sys
        mock_sb3 = MagicMock()
        mock_sb3.PPO = MagicMock()
        mock_sb3.PPO.load.return_value = "model"
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}):
            result = RayServePPOClient._load_sb3("/tmp/test.zip")
            assert result == "model"
            mock_sb3.PPO.load.assert_called_with("/tmp/test.zip")

    def test_load_sb3_failure(self):
        """Test that _load_sb3 returns None when SB3 import fails."""
        import sys
        mock_sb3 = MagicMock()
        mock_sb3.PPO = MagicMock()
        mock_sb3.PPO.load.side_effect = Exception("corrupt model")
        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}):
            result = RayServePPOClient._load_sb3("/tmp/bad.zip")
            assert result is None


class TestAutoReload:

    @pytest.mark.asyncio
    async def test_check_for_newer_model_detects_update(self):
        client = RayServePPOClient(serve_url="http://test:8765")
        client._latest_registry_version = 1

        mock_meta = MagicMock()
        mock_meta.version = 3

        with patch("mlops.model_registry.list_models") as mock_list:
            mock_list.return_value = [mock_meta]

            with patch("mlops.model_registry.get_model") as mock_get:
                mock_get.return_value = (3, "/tmp/v3.zip")

                with patch.object(RayServePPOClient, "_load_sb3") as mock_load:
                    mock_model = MagicMock()
                    mock_load.return_value = mock_model

                    await client._check_for_newer_model()

                    assert client._local_version == 3
                    assert client._latest_registry_version == 3
                    mock_load.assert_called_with("/tmp/v3.zip")

    @pytest.mark.asyncio
    async def test_check_skips_when_no_newer_version(self):
        client = RayServePPOClient(serve_url="http://test:8765")
        client._latest_registry_version = 5

        mock_meta = MagicMock()
        mock_meta.version = 3  # older

        with patch("mlops.model_registry.list_models") as mock_list:
            mock_list.return_value = [mock_meta]

            with patch.object(RayServePPOClient, "_load_sb3") as mock_load:
                await client._check_for_newer_model()
                mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_reload_loop_stops_on_cancel(self):
        client = RayServePPOClient(serve_url="http://test:8765")
        client._running = True
        client._auto_reload_interval = 1  # short sleep so test doesn't hang

        async def fake_check():
            client._running = False  # stop after one loop

        with patch.object(client, "_check_for_newer_model", fake_check):
            await client._auto_reload_loop()

        # No exception means it stopped cleanly
