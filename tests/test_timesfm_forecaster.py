"""Unit tests for the TimesFM forecaster wrapper.

These do not require the heavyweight ``timesfm`` package: model loading is
either exercised for its graceful-degradation path or bypassed with a stub.
"""
import numpy as np

from orchestrator.timesfm_forecaster import ForecastResult, TimesFMForecaster


class _StubModel:
    """Minimal stand-in for a compiled TimesFM model."""

    def __init__(self, slope: float):
        self.slope = slope

    def forecast(self, horizon, inputs):
        last = float(np.asarray(inputs[0])[-1])
        point = np.array([[last + self.slope * (i + 1) for i in range(horizon)]])
        # quantile head: (1, horizon, 10) -> mean + q10..q90
        quant = np.repeat(point[:, :, None], 10, axis=2)
        return point, quant


def test_graceful_when_model_unavailable():
    fc = TimesFMForecaster(checkpoint="does-not-exist/timesfm")
    result = fc.forecast([1.0, 2.0, 3.0], horizon=5)
    if not result.available:
        assert result.error is not None
        assert result.direction == "hold"


def test_insufficient_history_returns_error():
    fc = TimesFMForecaster()
    result = fc.forecast([100.0], horizon=5)
    assert result.available is False
    assert "at least 2" in (result.error or "")


def test_direction_long_on_upward_forecast():
    fc = TimesFMForecaster(direction_threshold=0.001)
    fc._model = _StubModel(slope=5.0)
    result = fc.forecast(list(np.linspace(100, 110, 50)), horizon=10)
    assert isinstance(result, ForecastResult)
    assert result.available is True
    assert result.direction == "long"
    assert result.expected_return > 0
    assert len(result.point_forecast) == 10
    assert "q10" in result.quantiles


def test_direction_short_on_downward_forecast():
    fc = TimesFMForecaster(direction_threshold=0.001)
    fc._model = _StubModel(slope=-5.0)
    result = fc.forecast(list(np.linspace(110, 100, 50)), horizon=10)
    assert result.direction == "short"
    assert result.expected_return < 0


def test_direction_hold_on_flat_forecast():
    fc = TimesFMForecaster(direction_threshold=0.05)
    fc._model = _StubModel(slope=0.0)
    result = fc.forecast([100.0] * 50, horizon=10)
    assert result.direction == "hold"
