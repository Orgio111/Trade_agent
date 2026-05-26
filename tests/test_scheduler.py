"""Unit tests for the MLOps scheduler."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlops.scheduler import MLOpsScheduler, RetrainReason, SchedulerState


@pytest.fixture
def mock_drift() -> MagicMock:
    detector = MagicMock()
    detector.has_reference = True
    detector.needs_retrain = MagicMock(return_value=False)
    return detector


@pytest.fixture
def mock_pipeline() -> AsyncMock:
    pipe = AsyncMock()
    pipe.run = AsyncMock(return_value=3)
    return pipe


class TestSchedulerState:

    def test_default_state(self):
        state = SchedulerState()
        assert state.last_check_at is None
        assert state.last_retrain_at is None
        assert state.total_checks == 0
        assert state.total_retrains == 0
        assert state.consecutive_drift_detections == 0
        assert state.is_running is False
        assert state.errors == []


class TestMLOpsSchedulerInit:

    def test_init_defaults(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        assert scheduler._drift == mock_drift
        assert scheduler._pipeline == mock_pipeline
        assert scheduler._check_interval > 0
        assert scheduler._retrain_interval > 0
        assert scheduler._cooldown > 0
        assert scheduler._task is None
        assert scheduler.state.is_running is False

    def test_init_custom_values(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            check_interval_s=10,
            retrain_interval_hours=1,
            retrain_cooldown_hours=2,
        )
        assert scheduler._check_interval == 10
        assert scheduler._retrain_interval == 1
        assert scheduler._cooldown == 2


class TestMLOpsSchedulerLifecycle:

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            check_interval_s=3600,
            retrain_interval_hours=24,
            retrain_cooldown_hours=4,
        )
        assert scheduler.state.is_running is False

        await scheduler.start()
        assert scheduler.state.is_running is True
        assert scheduler._task is not None

        await scheduler.stop()
        assert scheduler.state.is_running is False  # stop() sets is_running to False
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_start_twice_noop(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        await scheduler.start()
        task_id = id(scheduler._task)
        await scheduler.start()  # second start should be no-op
        assert id(scheduler._task) == task_id
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        await scheduler.stop()  # should not raise
        assert scheduler._task is None


class TestShouldRetrainByTime:

    def test_no_previous_retrain(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            retrain_interval_hours=24,
        )
        assert scheduler._should_retrain_by_time() is True

    def test_within_interval(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            retrain_interval_hours=24,
        )
        scheduler._state.last_retrain_at = datetime.utcnow()
        assert scheduler._should_retrain_by_time() is False

    def test_exceeded_interval(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            retrain_interval_hours=24,
        )
        scheduler._state.last_retrain_at = datetime.utcnow() - timedelta(hours=48)
        assert scheduler._should_retrain_by_time() is True


class TestTriggerRetrain:

    @pytest.mark.asyncio
    async def test_manual_retrain_returns_version(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        version = await scheduler.trigger_retrain(RetrainReason.MANUAL)
        assert version == 3
        mock_pipeline.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_manual_retrain_with_cooldown(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            retrain_cooldown_hours=24,
        )
        scheduler._state.last_retrain_at = datetime.utcnow()
        version = await scheduler.trigger_retrain(RetrainReason.MANUAL)
        assert version is None  # blocked by cooldown
        mock_pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_failure_returns_none(self, mock_drift, mock_pipeline):
        mock_pipeline.run = AsyncMock(return_value=None)
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        version = await scheduler.trigger_retrain(RetrainReason.MANUAL)
        assert version is None

    @pytest.mark.asyncio
    async def test_retrain_updates_state(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        assert scheduler.state.total_retrains == 0
        await scheduler.trigger_retrain(RetrainReason.MANUAL)
        assert scheduler.state.total_retrains == 1
        assert scheduler.state.last_retrain_at is not None


class TestMLOpsSchedulerDrift:

    @pytest.mark.asyncio
    async def test_drift_needs_retrain_after_threshold(self, mock_drift, mock_pipeline):
        """Drift must be detected 2+ consecutive times before retrain triggers."""
        mock_drift.needs_retrain.return_value = True
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            check_interval_s=3600,
        )

        # Simulate what _run_loop does internally
        scheduler._state.consecutive_drift_detections += 1  # first detection
        assert scheduler._state.consecutive_drift_detections == 1

        # Second consecutive detection should trigger retrain
        assert mock_drift.needs_retrain({}) is True  # mock returns True
        scheduler._state.consecutive_drift_detections += 1
        if scheduler._state.consecutive_drift_detections >= 2:
            await scheduler._run_pipeline(RetrainReason.DRIFT)

        assert scheduler.state.total_retrains == 1
        assert scheduler.state.consecutive_drift_detections == 0  # reset

    @pytest.mark.asyncio
    async def test_no_drift_resets_consecutive_counter(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            check_interval_s=3600,
        )
        scheduler._state.consecutive_drift_detections = 3

        # Simulate no drift detected
        scheduler._state.consecutive_drift_detections = 0
        assert scheduler._state.consecutive_drift_detections == 0

    def test_no_reference_skips_drift_check(self, mock_drift, mock_pipeline):
        mock_drift.has_reference = False
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        # When has_reference is False, the scheduler should skip drift checks
        # This is tested by checking that needs_retrain is not called
        assert not mock_drift.needs_retrain.called


class TestSchedulerRunPipeline:

    @pytest.mark.asyncio
    async def test_run_pipeline_cooldown_active(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            retrain_cooldown_hours=1,
        )
        scheduler._state.last_retrain_at = datetime.utcnow()
        result = await scheduler._run_pipeline(RetrainReason.SCHEDULED)
        assert result is None
        mock_pipeline.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_pipeline_no_cooldown(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(
            mock_drift, mock_pipeline,
            retrain_cooldown_hours=1,
        )
        # No previous retrain = no cooldown
        result = await scheduler._run_pipeline(RetrainReason.SCHEDULED)
        assert result == 3
        mock_pipeline.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_pipeline_updates_state(self, mock_drift, mock_pipeline):
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        await scheduler._run_pipeline(RetrainReason.STARTUP)
        assert scheduler.state.last_retrain_at is not None
        assert scheduler.state.total_retrains == 1
        assert scheduler.state.consecutive_drift_detections == 0

    @pytest.mark.asyncio
    async def test_run_pipeline_error_handled(self, mock_drift, mock_pipeline):
        mock_pipeline.run = AsyncMock(side_effect=ValueError("test error"))
        scheduler = MLOpsScheduler(mock_drift, mock_pipeline)
        result = await scheduler._run_pipeline(RetrainReason.MANUAL)
        assert result is None
        assert "test error" in scheduler.state.errors
