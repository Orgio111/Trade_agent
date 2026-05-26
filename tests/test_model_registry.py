"""Unit tests for the Model Registry."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from mlops.model_registry import (
    ModelMetadata,
    compare_models,
    get_model,
    list_models,
    prune_models,
    save_model,
)


@pytest.fixture(autouse=True)
def _patch_registry_root(monkeypatch: pytest.MonkeyPatch) -> str:
    """Redirect registry to a temp directory for each test."""
    from core.config import get_settings
    get_settings.cache_clear()
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("MODEL_REGISTRY_PATH", tmpdir)
    return tmpdir


class TestSaveModel:

    def test_save_first_model(self):
        meta = ModelMetadata(
            version=0,
            created_at="2024-06-01T00:00:00",
            model_type="ppo_execution",
            total_timesteps=100_000,
            train_sharpe=1.5,
            test_sharpe=1.2,
            test_max_drawdown=0.05,
            test_win_rate=0.62,
            feature_count=40,
        )
        version = save_model(__file__, meta)
        assert version == 1

        # Verify directory was created
        registry = os.environ["MODEL_REGISTRY_PATH"]
        meta_path = os.path.join(registry, f"v{version:04d}", "metadata.json")
        assert os.path.isfile(meta_path)

        with open(meta_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["train_sharpe"] == 1.5

    def test_save_increments_version(self):
        meta = ModelMetadata(version=0, created_at="", total_timesteps=1000)
        v1 = save_model(__file__, meta)
        v2 = save_model(__file__, meta)
        assert v2 == v1 + 1


class TestListModels:

    def test_empty_registry_returns_empty(self):
        assert list_models() == []

    def test_returns_sorted_by_version_desc(self, _patch_registry_root):
        meta = ModelMetadata(version=0, created_at="", total_timesteps=1000)
        v1 = save_model(__file__, meta)
        v2 = save_model(__file__, meta)
        models = list_models()
        assert len(models) == 2
        assert models[0].version == v2
        assert models[1].version == v1


class TestGetModel:

    def test_get_latest(self, _patch_registry_root):
        meta = ModelMetadata(version=0, created_at="", total_timesteps=1000)
        v1 = save_model(__file__, meta)
        v2 = save_model(__file__, meta)
        result = get_model()
        assert result is not None
        ver, path = result
        assert ver == v2
        assert os.path.isfile(path)

    def test_get_specific_version(self, _patch_registry_root):
        meta = ModelMetadata(version=0, created_at="", total_timesteps=1000)
        v1 = save_model(__file__, meta)
        result = get_model(v1)
        assert result is not None
        assert result[0] == v1

    def test_get_nonexistent_returns_none(self):
        result = get_model(999)
        assert result is None


class TestCompareModels:

    def test_compare_returns_deltas(self, _patch_registry_root):
        meta_a = ModelMetadata(
            version=0, created_at="", total_timesteps=100_000,
            train_sharpe=1.0, test_sharpe=0.8,
        )
        meta_b = ModelMetadata(
            version=0, created_at="", total_timesteps=200_000,
            train_sharpe=1.5, test_sharpe=1.2,
        )
        v1 = save_model(__file__, meta_a)
        v2 = save_model(__file__, meta_b)
        comp = compare_models(v1, v2)
        assert comp["deltas"]["train_sharpe"] == pytest.approx(0.5)
        assert comp["deltas"]["test_sharpe"] == pytest.approx(0.4)
        assert comp["deltas"]["timesteps"] == 100_000

    def test_compare_missing_raises(self, _patch_registry_root):
        with pytest.raises(ValueError):
            compare_models(1, 999)


class TestPruneModels:

    def test_keep_last_n(self, _patch_registry_root):
        meta = ModelMetadata(version=0, created_at="", total_timesteps=1000)
        versions = [save_model(__file__, meta) for _ in range(5)]
        assert len(list_models()) == 5
        pruned = prune_models(keep_last=3)
        assert pruned == 2
        remaining = list_models()
        assert len(remaining) == 3
        assert remaining[0].version == versions[-1]

    def test_no_prune_needed(self, _patch_registry_root):
        meta = ModelMetadata(version=0, created_at="", total_timesteps=1000)
        save_model(__file__, meta)
        assert prune_models(keep_last=10) == 0


class TestModelMetadata:

    def test_short_summary_format(self):
        meta = ModelMetadata(
            version=3,
            created_at="2024-06-01",
            train_sharpe=1.5,
            test_sharpe=1.2,
            test_max_drawdown=0.05,
            test_win_rate=0.62,
            total_timesteps=500_000,
        )
        summary = meta.short_summary()
        assert "v3" in summary
        assert "1.50" in summary
        assert "1.20" in summary
        assert "5.0%" in summary or "5.00%" in summary
        assert "500,000" in summary
