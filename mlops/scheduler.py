"""
Async scheduler for the MLOps pipeline.

Periodically checks feature drift and triggers the full retraining pipeline
(Walk-Forward Optimization → PPO retrain → Model Registry) when needed or
on a configured interval.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from core.config import get_settings
from core.observability import RETRAIN_COUNTER

logger = logging.getLogger(__name__)


@dataclass
class SchedulerState:
    """Track scheduler state across cycles."""
    last_check_at: datetime | None = None
    last_retrain_at: datetime | None = None
    total_checks: int = 0
    total_retrains: int = 0
    consecutive_drift_detections: int = 0
    is_running: bool = False
    errors: list[str] = field(default_factory=list)


# ── Scheduled retrain reason ──────────────────────────────────────────────────

class RetrainReason:
    DRIFT = "drift_detected"
    SCHEDULED = "scheduled_interval"
    STARTUP = "startup"
    MANUAL = "manual"


# ── Scheduler ─────────────────────────────────────────────────────────────────

class MLOpsScheduler:
    """Async scheduler that manages the MLOps pipeline lifecycle.

    Usage::

        scheduler = MLOpsScheduler(drift_detector, pipeline)
        await scheduler.start()
        # ... runs in background ...
        await scheduler.stop()
    """

    def __init__(
        self,
        drift_detector: Any,
        pipeline: Any,
        check_interval_s: int | None = None,
        retrain_interval_hours: int | None = None,
        retrain_cooldown_hours: int | None = None,
    ) -> None:
        """
        Args:
            drift_detector: DriftDetector instance.
            pipeline: MLOpsPipeline instance.
            check_interval_s: How often to check for drift (default from config).
            retrain_interval_hours: Max age before forced retrain (default from config).
            retrain_cooldown_hours: Min time between retrains (default from config).
        """
        self._drift = drift_detector
        self._pipeline = pipeline
        self._state = SchedulerState()
        self._task: asyncio.Task | None = None
        self._current_features: dict[str, np.ndarray] = {}

        cfg = get_settings()
        self._check_interval = check_interval_s or cfg.mlops_check_interval_s
        self._retrain_interval = (
            retrain_interval_hours or cfg.mlops_retrain_interval_hours
        )
        self._cooldown = (
            retrain_cooldown_hours or cfg.mlops_retrain_cooldown_hours
        )

    # ── Feature data feed ────────────────────────────────────────────────────

    def update_features(self, features: dict[str, np.ndarray]) -> None:
        """Push the latest feature window for drift checking."""
        self._current_features = features

    @property
    def state(self) -> SchedulerState:
        return self._state

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background scheduler loop."""
        if self._state.is_running:
            logger.warning("MLOps scheduler already running — skipping start")
            return
        self._state.is_running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "MLOps scheduler started (check_interval=%ds, retrain_interval=%dh, cooldown=%dh)",
            self._check_interval, self._retrain_interval, self._cooldown,
        )

    async def stop(self) -> None:
        """Stop the background scheduler loop."""
        self._state.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("MLOps scheduler stopped (checks=%d, retrains=%d)",
                     self._state.total_checks, self._state.total_retrains)

    async def trigger_retrain(self, reason: str = RetrainReason.MANUAL) -> str | None:
        """Manually trigger a full retrain cycle.

        Returns the model version registered, or None on failure.
        """
        logger.info("Manual retrain triggered (reason=%s)", reason)
        return await self._run_pipeline(reason)

    # ── Internal loop ─────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main scheduler loop — runs until stopped."""
        cfg = get_settings()

        # Run an initial check after a short warm-up
        await asyncio.sleep(10)
        if not self._drift.has_reference:
            logger.info("Scheduler: no drift reference set — skipping initial check")
        else:
            # Force a startup check even without drift
            if self._should_retrain_by_time():
                await self._run_pipeline(RetrainReason.STARTUP)

        while self._state.is_running:
            try:
                await asyncio.sleep(self._check_interval)
                self._state.total_checks += 1
                self._state.last_check_at = datetime.utcnow()

                # Check if reference is set
                if not self._drift.has_reference:
                    logger.debug("Scheduler: no drift reference yet — skipping")
                    continue

                # Check time-based forced retrain
                if self._should_retrain_by_time():
                    await self._run_pipeline(RetrainReason.SCHEDULED)
                    continue

                # Check drift (only if features have been pushed)
                if self._current_features and self._drift.needs_retrain(self._current_features):
                    self._state.consecutive_drift_detections += 1
                    if self._state.consecutive_drift_detections >= 2:
                        await self._run_pipeline(RetrainReason.DRIFT)
                else:
                    self._state.consecutive_drift_detections = 0

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("MLOps scheduler error: %s", exc, exc_info=True)
                self._state.errors.append(str(exc))

    def _should_retrain_by_time(self) -> bool:
        """Check if the model has exceeded its retrain interval."""
        last = self._state.last_retrain_at
        if last is None:
            return True
        elapsed = (datetime.utcnow() - last).total_seconds()
        return elapsed > self._retrain_interval * 3600

    async def _run_pipeline(self, reason: str) -> str | None:
        """Execute the full retrain pipeline with cooldown check."""
        # Cooldown check
        if self._state.last_retrain_at is not None:
            since_last = (datetime.utcnow() - self._state.last_retrain_at).total_seconds()
            if since_last < self._cooldown * 3600:
                logger.info(
                    "Retrain skipped — cooldown active (%.1fh < %dh)",
                    since_last / 3600, self._cooldown,
                )
                return None

        logger.info("MLOps pipeline triggered (reason=%s)", reason)
        try:
            version = await self._pipeline.run()
            if version is not None:
                self._state.last_retrain_at = datetime.utcnow()
                self._state.total_retrains += 1
                self._state.consecutive_drift_detections = 0
                logger.info("MLOps pipeline complete — registered model v%d", version)
            return version
        except Exception as exc:
            logger.error("MLOps pipeline failed: %s", exc, exc_info=True)
            self._state.errors.append(str(exc))
            return None
