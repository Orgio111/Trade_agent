"""
QUANTEX Model Registry — Version management for all trained models.

Tracks:
  - ML model versions (Random Forest, XGBoost)
  - RL model versions (PPO, DQN)
  - Training metadata (data range, accuracy, params)
  - Performance history per version

Storage:
  - JSON metadata in models/registry/
  - Actual model weights in models/ml/ or models/rl/

Usage:
    registry = ModelRegistry()
    version = registry.register_model("ml_rf", "v1", {"accuracy": 0.96})
    best = registry.get_best_model("ml_rf")
    history = registry.get_model_history("ml_rf")
"""

import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelVersion:
    """A specific model version."""
    model_type: str      # "ml_rf", "ppo_rl", "dqn_rl", "ensemble"
    version: str         # "v1", "v2", "latest"
    metrics: dict        # accuracy, sharpe, pnl, etc.
    metadata: dict       # training params, data range, timestamp
    file_path: Optional[str] = None  # Path to saved model file
    created_at: float = 0.0
    promoted: bool = False   # Whether this is the current production model

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(**data)


class ModelRegistry:
    """
    Model registry with version tracking and promotion.

    File structure:
      models/
      ├── registry/
      │   └── registry.json       ← All version metadata
      ├── ml/
      │   ├── rf_v1.pkl
      │   └── rf_latest.pkl
      ├── rl/
      │   ├── ppo_v1.zip
      │   └── ppo_latest.zip

    Usage:
        registry = ModelRegistry()
        registry.register_model("ml_rf", "v1", {"accuracy": 0.95})
        versions = registry.list_models("ml_rf")
        best = registry.get_best_model("ml_rf", metric="test_accuracy")
        registry.promote_model("ml_rf", "v2")
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or Path("models/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self._registry: dict[str, list[dict]] = self._load_registry()

    # ── Registry I/O ──────────────────────────────────────────

    def _load_registry(self) -> dict:
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_registry(self):
        """Persist registry to disk."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    # ── Model Registration ────────────────────────────────────

    def register_model(
        self,
        model_type: str,
        version: str,
        metrics: dict,
        metadata: Optional[dict] = None,
        file_path: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_type: "ml_rf", "ppo_rl", "dqn_rl", "ensemble"
            version: Version string (e.g. "v1", "v2")
            metrics: Performance metrics dict
            metadata: Training metadata dict
            file_path: Path to saved model file

        Returns:
            ModelVersion
        """
        if model_type not in self._registry:
            self._registry[model_type] = []

        mv = ModelVersion(
            model_type=model_type,
            version=version,
            metrics=metrics,
            metadata=metadata or {},
            file_path=file_path,
        )

        self._registry[model_type].append(mv.to_dict())
        self._save_registry()
        return mv

    def get_model(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        versions = self._registry.get(model_type, [])
        for v in versions:
            if v["version"] == version:
                return ModelVersion.from_dict(v)
        return None

    def get_latest_model(self, model_type: str) -> Optional[ModelVersion]:
        """Get the latest (most recently registered) model of a type."""
        versions = self._registry.get(model_type, [])
        if not versions:
            return None
        latest = max(versions, key=lambda v: v["created_at"])
        return ModelVersion.from_dict(latest)

    def get_best_model(self, model_type: str, metric: str = "test_accuracy") -> Optional[ModelVersion]:
        """
        Get the best model based on a metric.

        Args:
            model_type: Model type to search
            metric: Metric key to sort by

        Returns:
            Best ModelVersion or None
        """
        versions = self._registry.get(model_type, [])
        if not versions:
            return None

        scored = []
        for v in versions:
            val = v.get("metrics", {}).get(metric)
            if val is not None:
                scored.append((val, v))

        if not scored:
            # Fall back to latest
            return max(versions, key=lambda v: v["created_at"])

        scored.sort(key=lambda x: x[0], reverse=True)
        return ModelVersion.from_dict(scored[0][1])

    def get_promoted_model(self, model_type: str) -> Optional[ModelVersion]:
        """Get the currently promoted (production) model."""
        versions = self._registry.get(model_type, [])
        for v in versions:
            if v.get("promoted"):
                return ModelVersion.from_dict(v)
        return None

    def promote_model(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """
        Promote a model version to production.

        De-promotes any currently promoted model of the same type.

        Args:
            model_type: Model type
            version: Version to promote

        Returns:
            Promoted ModelVersion or None
        """
        versions = self._registry.get(model_type, [])
        target = None

        for v in versions:
            if v["version"] == version:
                v["promoted"] = True
                target = v
            else:
                v["promoted"] = False

        if target:
            self._save_registry()
            logger.info(f"Promoted {model_type} {version} to production")
            return ModelVersion.from_dict(target)

        return None

    def list_models(self, model_type: Optional[str] = None) -> list[dict]:
        """
        List all registered models.

        Args:
            model_type: Optional filter by type

        Returns:
            List of model version dicts
        """
        if model_type:
            return self._registry.get(model_type, [])
        all_models = []
        for mtype, versions in self._registry.items():
            for v in versions:
                all_models.append(v)
        return all_models

    def delete_model(self, model_type: str, version: str) -> bool:
        """Delete a model version from the registry."""
        versions = self._registry.get(model_type, [])
        for i, v in enumerate(versions):
            if v["version"] == version:
                # Delete file if exists
                if v.get("file_path"):
                    try:
                        Path(v["file_path"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                versions.pop(i)
                self._save_registry()
                return True
        return False

    def get_summary(self) -> dict:
        """Get summary of registry contents."""
        summary = {}
        for model_type, versions in self._registry.items():
            summary[model_type] = {
                "versions": len(versions),
                "latest": versions[-1]["version"] if versions else None,
                "promoted": next((v["version"] for v in versions if v.get("promoted")), None),
                "metrics": {
                    k: [v["metrics"].get(k) for v in versions if v["metrics"].get(k) is not None]
                    for k in ["accuracy", "test_accuracy", "sharpe", "pnl"] if versions
                },
            }
        return summary


# Helper logger for promote_model
import logging
logger = logging.getLogger("quantex.model_registry")
