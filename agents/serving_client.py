"""
Ray Serve client for real-time PPO inference.

Provides a ``RayServePPOClient`` that calls the remote ``PPOExecution``
deployment over HTTP, periodically polls the model registry for newer
versions, and falls back to a local SB3 model when Ray Serve is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PPOPrediction:
    """Result of a PPO inference call."""
    action: int
    source: str                # "ray_serve", "local", "fallback"
    model_version: int = 0
    latency_ms: float = 0.0
    deterministic: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Ray Serve HTTP Client
# ═══════════════════════════════════════════════════════════════════════════════

class RayServePPOClient:
    """Asynchronous HTTP client for the remote Ray Serve PPO deployment.

    Usage::

        client = RayServePPOClient()
        result = await client.predict(obs_array)
        print(f"Action {result.action} via {result.source}")
    """

    def __init__(
        self,
        serve_url: str | None = None,
        model_registry_path: str | None = None,
        auto_reload_interval_s: int = 300,
        fallback_model_path: str | None = None,
    ) -> None:
        """
        Args:
            serve_url: Base URL of the Ray Serve cluster (e.g. ``http://ray-head:8765``).
                       If None, reads from ``RAY_SERVE_URL`` env var or disables remote.
            model_registry_path: Path to the versioned model registry.
                                 If None, reads from ``MODEL_REGISTRY_PATH`` env var.
            auto_reload_interval_s: How often to check the registry for a newer model.
            fallback_model_path: Path to a local SB3 model for fallback.
        """
        cfg = get_settings()
        self._serve_url = serve_url or os.environ.get("RAY_SERVE_URL", "")
        self._registry_path = model_registry_path or cfg.model_registry_path
        self._auto_reload_interval = auto_reload_interval_s
        self._fallback_path = fallback_model_path or cfg.ppo_model_path

        # Local fallback model (loaded when Ray Serve is unreachable)
        self._local_model: Any = None
        self._local_version: int = 0

        # Current known latest version from registry
        self._latest_registry_version: int = 0

        # HTTP session (lazy-initialised)
        self._session: Any = None
        self._session_lock = asyncio.Lock()

        # Background auto-reload task
        self._reload_task: asyncio.Task[None] | None = None
        self._running = False

        logger.info(
            "RayServePPOClient: serve_url=%s registry=%s fallback=%s",
            self._serve_url or "(disabled)",
            self._registry_path,
            self._fallback_path,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background auto-reload loop."""
        self._running = True
        # Initial load attempt
        await self._try_load_local()
        if self._serve_url:
            self._reload_task = asyncio.create_task(self._auto_reload_loop())
            logger.info("Ray Serve auto-reload started (interval=%ds)", self._auto_reload_interval)

    async def stop(self) -> None:
        """Stop the background auto-reload loop and close the HTTP session."""
        self._running = False
        if self._reload_task is not None:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
        await self._close_session()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_remote_available(self) -> bool:
        """Whether a Ray Serve URL is configured."""
        return bool(self._serve_url)

    async def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        timeout_s: float = 5.0,
    ) -> PPOPrediction:
        """Predict an action from a 40-dim observation.

        Tries Ray Serve first, then local model, then returns action 0 as
        a hard fallback.
        """
        t0 = time.monotonic()

        # ── Try Ray Serve ────────────────────────────────────────────────
        if self._serve_url:
            try:
                action, version = await self._remote_predict(
                    obs.tolist(), deterministic, timeout_s,
                )
                latency = (time.monotonic() - t0) * 1000
                logger.debug("Ray Serve predict: action=%d version=%d (%.1fms)", action, version, latency)
                return PPOPrediction(
                    action=action, source="ray_serve",
                    model_version=version, latency_ms=latency,
                    deterministic=deterministic,
                )
            except Exception as exc:
                logger.warning("Ray Serve predict failed: %s — trying local model", exc)

        # ── Try local model ──────────────────────────────────────────────
        if self._local_model is not None:
            try:
                act, _ = self._local_model.predict(obs, deterministic=deterministic)
                action_val = int(act[0] if hasattr(act, "__len__") else act)
                latency = (time.monotonic() - t0) * 1000
                return PPOPrediction(
                    action=action_val, source="local",
                    model_version=self._local_version,
                    latency_ms=latency, deterministic=deterministic,
                )
            except Exception as exc:
                logger.warning("Local model predict failed: %s", exc)

        # ── Hard fallback ────────────────────────────────────────────────
        latency = (time.monotonic() - t0) * 1000
        return PPOPrediction(
            action=0, source="fallback",
            latency_ms=latency, deterministic=deterministic,
        )

    # ── Remote inference via HTTP ─────────────────────────────────────────

    async def _remote_predict(
        self,
        obs_list: list[float],
        deterministic: bool,
        timeout_s: float,
    ) -> tuple[int, int]:
        """Call the Ray Serve PPO deployment via HTTP POST.

        Returns (action, model_version).
        """
        session = await self._get_session()
        async with session.post(
            f"{self._serve_url}/ppo",
            json={"obs": obs_list, "deterministic": deterministic},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return int(data.get("action", 0)), int(data.get("version", 0))

    async def _get_session(self) -> Any:
        """Lazy-initialized aiohttp session."""
        if self._session is None:
            async with self._session_lock:
                if self._session is None:
                    import aiohttp  # type: ignore[import]
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={"User-Agent": "TradeAgent/1.0"},
                    )
        return self._session

    async def _close_session(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ── Local model loading ──────────────────────────────────────────────

    async def _try_load_local(self) -> bool:
        """Try to load a local SB3 PPO model from the registry or fallback path.

        Returns True if a model was successfully loaded (or already loaded
        on the latest version).
        """
        # First check the model registry for the latest version
        try:
            from mlops.model_registry import get_model
            result = get_model()  # returns (version, file_path) or None
            if result is not None:
                ver, model_file = result
                if ver > self._local_version:
                    model = self._load_sb3(model_file)
                    if model is not None:
                        self._local_model = model
                        self._local_version = ver
                        self._latest_registry_version = ver
                        logger.info("Loaded model v%d from registry: %s", ver, model_file)
                        return True
        except Exception as exc:
            logger.debug("Registry load attempt failed: %s", exc)

        # Fall back to the configured PPO model path
        if self._local_model is None and os.path.exists(self._fallback_path):
            model = self._load_sb3(self._fallback_path)
            if model is not None:
                self._local_model = model
                self._local_version = 0
                logger.info("Loaded fallback model from %s", self._fallback_path)
                return True

        return False

    @staticmethod
    def _load_sb3(path: str) -> Any | None:
        """Load a stable-baselines3 PPO model, returning None on failure."""
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
            return PPO.load(path)
        except Exception as exc:
            logger.warning("Failed to load SB3 model from %s: %s", path, exc)
            return None

    # ── Auto-reload loop ─────────────────────────────────────────────────

    async def _auto_reload_loop(self) -> None:
        """Periodically check the model registry for newer versions.

        When a newer version is found, reload it into the local fallback
        model so that local predictions also benefit from retrained models.
        """
        while self._running:
            try:
                await asyncio.sleep(self._auto_reload_interval)
                await self._check_for_newer_model()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Auto-reload check failed: %s", exc)

    async def _check_for_newer_model(self) -> None:
        """Poll the model registry and load a newer version if found."""
        try:
            from mlops.model_registry import get_model, list_models

            models = list_models()
            if not models:
                return

            latest = models[0]  # sorted desc
            if latest.version <= self._latest_registry_version:
                return

            logger.info(
                "New model v%d detected (current: v%d) — reloading",
                latest.version, self._latest_registry_version,
            )

            result = get_model(latest.version)
            if result is None:
                return

            ver, model_file = result
            model = self._load_sb3(model_file)
            if model is not None:
                self._local_model = model
                self._local_version = ver
                self._latest_registry_version = ver
                logger.info("Auto-reloaded to model v%d", ver)
        except Exception as exc:
            logger.warning("Auto-reload check error: %s", exc)


# ── Top-level helper ─────────────────────────────────────────────────────────

_client_instance: RayServePPOClient | None = None
_client_lock = asyncio.Lock()


async def get_ppo_client() -> RayServePPOClient:
    """Singleton accessor for the global Ray Serve PPO client."""
    global _client_instance
    if _client_instance is None:
        async with _client_lock:
            if _client_instance is None:
                cfg = get_settings()
                _client_instance = RayServePPOClient(
                    serve_url=os.environ.get("RAY_SERVE_URL", ""),
                    auto_reload_interval_s=cfg.ray_serve_auto_reload_s,
                    fallback_model_path=cfg.ppo_model_path,
                )
                await _client_instance.start()
    return _client_instance
