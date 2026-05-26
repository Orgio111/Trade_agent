"""Versioned model registry for the Trade Agent.

Saves PPO model checkpoints with performance metadata and provides
compare / rollback / prune operations.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Registry metadata ─────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """Metadata stored alongside each model checkpoint."""
    version: int
    created_at: str                      # ISO timestamp
    model_type: str = "ppo_execution"    # e.g. ppo_execution, ppo_routing
    training_duration_s: float = 0.0
    total_timesteps: int = 0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0
    test_win_rate: float = 0.0
    test_information_ratio: float = 0.0
    test_calmar_ratio: float = 0.0
    test_max_consecutive_losses: int = 0
    psi_scores: dict[str, float] = field(default_factory=dict)
    feature_count: int = 40
    notes: str = ""

    def short_summary(self) -> str:
        return (
            f"v{self.version}  Sharpe(train={self.train_sharpe:.2f} test={self.test_sharpe:.2f})  "
            f"DD={self.test_max_drawdown:.2%}  WR={self.test_win_rate:.1%}  "
            f"IR={self.test_information_ratio:.2f}  Calmar={self.test_calmar_ratio:.2f}  "
            f"maxLoss={self.test_max_consecutive_losses}  "
            f"steps={self.total_timesteps:,}"
        )


# ── Registry path helpers ─────────────────────────────────────────────────────

def _registry_root() -> str:
    from core.config import get_settings
    return get_settings().model_registry_path


def _model_dir(version: int) -> str:
    return os.path.join(_registry_root(), f"v{version:04d}")


def _metadata_path(version: int) -> str:
    return os.path.join(_model_dir(version), "metadata.json")


def _model_file(version: int) -> str:
    return os.path.join(_model_dir(version), "ppo_model.zip")


# ── Registry operations ───────────────────────────────────────────────────────

def list_models() -> list[ModelMetadata]:
    """Return all registered models sorted by version descending."""
    root = _registry_root()
    if not os.path.isdir(root):
        return []
    versions: list[ModelMetadata] = []
    for entry in sorted(os.listdir(root), reverse=True):
        vpath = os.path.join(root, entry)
        meta_path = os.path.join(vpath, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                data = json.load(f)
            versions.append(ModelMetadata(**data))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Skipping corrupt metadata at %s: %s", meta_path, exc)
    return versions


def get_model(version: int | None = None) -> tuple[int, str] | None:
    """Return (version, model_file_path) for the requested or latest version."""
    if version is not None:
        mf = _model_file(version)
        if os.path.isfile(mf):
            return version, mf
        logger.warning("Model version %d not found at %s", version, mf)
        return None

    models = list_models()
    if not models:
        return None
    best = models[0]  # sorted desc, first is latest
    return best.version, _model_file(best.version)


def save_model(
    model_path: str,
    metadata: ModelMetadata,
) -> int:
    """Copy a trained model into the registry and write metadata.

    Returns the assigned version number.
    """
    models = list_models()
    next_version = (models[0].version + 1) if models else 1

    dst_dir = _model_dir(next_version)
    os.makedirs(dst_dir, exist_ok=True)

    # Copy model file
    dst_model = _model_file(next_version)
    shutil.copy2(model_path, dst_model)
    logger.info("Model v%d saved to %s", next_version, dst_model)

    # Write metadata
    meta = asdict(metadata)
    meta["version"] = next_version
    meta["created_at"] = datetime.utcnow().isoformat()
    with open(_metadata_path(next_version), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Write latest_version.txt so Ray Serve and other consumers
    # can discover the newest checkpoint without scanning the directory.
    latest_flag = os.path.join(_registry_root(), "latest_version.txt")
    try:
        with open(latest_flag, "w") as f:
            f.write(str(next_version))
    except OSError as exc:
        logger.warning("Failed to write latest_version.txt: %s", exc)

    logger.info("Model metadata written for v%d", next_version)
    return next_version


def load_model(version: int | None = None) -> Any | None:
    """Load a PPO model from the registry.

    Returns the loaded stable-baselines3 PPO model, or None on failure.
    """
    result = get_model(version)
    if result is None:
        logger.warning("No model found in registry (version=%s)", version)
        return None

    ver, model_file = result
    try:
        from stable_baselines3 import PPO  # type: ignore[import]
        model = PPO.load(model_file)
        logger.info("PPO model v%d loaded from %s", ver, model_file)
        return model
    except Exception as exc:
        logger.error("Failed to load PPO model v%d: %s", ver, exc)
        return None


def compare_models(version_a: int, version_b: int) -> dict[str, Any]:
    """Compare two model versions by their metadata.

    Returns a dict with keys ``a``, ``b``, and ``deltas``.
    """
    meta_list = list_models()
    meta_map = {m.version: m for m in meta_list}
    a = meta_map.get(version_a)
    b = meta_map.get(version_b)
    if not a or not b:
        raise ValueError(
            f"One or both versions not found: {version_a}, {version_b}"
        )

    deltas = {
        "train_sharpe": round(b.train_sharpe - a.train_sharpe, 4),
        "test_sharpe": round(b.test_sharpe - a.test_sharpe, 4),
        "max_drawdown": round(b.test_max_drawdown - a.test_max_drawdown, 4),
        "win_rate": round(b.test_win_rate - a.test_win_rate, 4),
        "information_ratio": round(b.test_information_ratio - a.test_information_ratio, 4),
        "calmar_ratio": round(b.test_calmar_ratio - a.test_calmar_ratio, 4),
        "max_consecutive_losses": b.test_max_consecutive_losses - a.test_max_consecutive_losses,
        "timesteps": b.total_timesteps - a.total_timesteps,
    }

    return {
        "a": asdict(a),
        "b": asdict(b),
        "deltas": deltas,
    }


def prune_models(keep_last: int = 10) -> int:
    """Remove older models, keeping only the *keep_last* most recent.

    Returns the number of deleted model directories.
    """
    models = list_models()
    if len(models) <= keep_last:
        return 0

    to_delete = models[keep_last:]
    deleted = 0
    for m in to_delete:
        d = _model_dir(m.version)
        if os.path.isdir(d):
            shutil.rmtree(d)
            logger.info("Pruned model v%d (%s)", m.version, d)
            deleted += 1
    return deleted
