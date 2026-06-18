"""TimesFM time-series forecasting integration for QUANTEX.

Wraps Google Research's TimesFM (Time Series Foundation Model) to produce
short-horizon price forecasts from recent candle data and derive a simple
directional trading bias from them.

TimesFM is a heavyweight, optional dependency: the 200M-parameter checkpoint is
downloaded from Hugging Face on first use and requires a torch backend. To keep
the core service and CI lightweight, the model is imported and loaded lazily and
the forecaster degrades gracefully (returning an ``error`` field) when TimesFM
is not installed or the checkpoint cannot be loaded.

Install the optional dependency with::

    pip install "timesfm[torch]>=2.0,<3"

Reference: https://github.com/google-research/timesfm
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Quantile levels emitted by TimesFM's continuous quantile head (10th..90th).
_QUANTILE_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


@dataclass
class ForecastResult:
    """Outcome of a single TimesFM forecast call."""

    horizon: int
    last_price: float
    point_forecast: list[float] = field(default_factory=list)
    quantiles: dict[str, list[float]] = field(default_factory=dict)
    direction: str = "hold"  # "long" | "short" | "hold"
    confidence: float = 0.0
    expected_return: float = 0.0
    model: str = ""
    available: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "last_price": self.last_price,
            "point_forecast": self.point_forecast,
            "quantiles": self.quantiles,
            "direction": self.direction,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "model": self.model,
            "available": self.available,
            "error": self.error,
        }


class TimesFMForecaster:
    """Lazy wrapper around a pretrained TimesFM checkpoint.

    Parameters
    ----------
    checkpoint:
        Hugging Face checkpoint id to load.
    max_context:
        Maximum number of historical points fed to the model.
    max_horizon:
        Maximum forecast horizon the compiled model supports.
    direction_threshold:
        Minimum absolute expected return (fraction of last price) required to
        emit a non-``hold`` directional bias.
    """

    DEFAULT_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        max_context: int = 1024,
        max_horizon: int = 256,
        direction_threshold: float = 0.002,
    ):
        self.checkpoint = checkpoint
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.direction_threshold = direction_threshold
        self._model = None
        self._load_error: str | None = None

    # ── Model lifecycle ────────────────────────────────────────────
    def _ensure_model(self) -> None:
        """Load and compile the TimesFM model on first use."""
        if self._model is not None or self._load_error is not None:
            return
        try:
            import timesfm
            import torch

            torch.set_float32_matmul_precision("high")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            self._model = model
        except ImportError as e:
            self._load_error = (
                "TimesFM is not installed. Install it with "
                f'`pip install "timesfm[torch]>=2.0,<3"`. ({e})'
            )
        except Exception as e:  # checkpoint download / compile failure
            self._load_error = f"Failed to load TimesFM checkpoint '{self.checkpoint}': {e}"

    @property
    def available(self) -> bool:
        """Whether the model is (or can be) loaded."""
        self._ensure_model()
        return self._model is not None

    # ── Forecasting ────────────────────────────────────────────────
    def forecast(self, prices: list[float] | np.ndarray, horizon: int = 12) -> ForecastResult:
        """Forecast the next ``horizon`` steps of a univariate price series."""
        series = np.asarray(prices, dtype=np.float64).ravel()
        series = series[np.isfinite(series)]
        last_price = float(series[-1]) if series.size else 0.0
        horizon = max(1, min(int(horizon), self.max_horizon))

        if series.size < 2:
            return ForecastResult(
                horizon=horizon,
                last_price=last_price,
                model=self.checkpoint,
                available=False,
                error="Need at least 2 historical points to forecast.",
            )

        self._ensure_model()
        if self._model is None:
            return ForecastResult(
                horizon=horizon,
                last_price=last_price,
                model=self.checkpoint,
                available=False,
                error=self._load_error,
            )

        context = series[-self.max_context :]
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=horizon, inputs=[context]
        )

        point = np.asarray(point_forecast)[0]
        quantiles: dict[str, list[float]] = {}
        q_arr = np.asarray(quantile_forecast)[0]  # (horizon, 10): mean + 10..90
        if q_arr.ndim == 2 and q_arr.shape[1] >= len(_QUANTILE_LEVELS) + 1:
            for idx, level in enumerate(_QUANTILE_LEVELS, start=1):
                quantiles[f"q{level}"] = [float(v) for v in q_arr[:, idx]]

        return self._build_result(last_price, horizon, point, quantiles)

    def _build_result(
        self,
        last_price: float,
        horizon: int,
        point: np.ndarray,
        quantiles: dict[str, list[float]],
    ) -> ForecastResult:
        target = float(point[-1])
        expected_return = (target - last_price) / last_price if last_price else 0.0

        if expected_return > self.direction_threshold:
            direction = "long"
        elif expected_return < -self.direction_threshold:
            direction = "short"
        else:
            direction = "hold"

        # Confidence: scale expected move, tightened when the quantile band is wide.
        confidence = float(min(1.0, abs(expected_return) / (self.direction_threshold * 10)))
        if "q10" in quantiles and "q90" in quantiles and last_price:
            band = (quantiles["q90"][-1] - quantiles["q10"][-1]) / last_price
            if band > 0:
                confidence *= float(1.0 / (1.0 + band * 10))

        return ForecastResult(
            horizon=horizon,
            last_price=last_price,
            point_forecast=[float(v) for v in point],
            quantiles=quantiles,
            direction=direction,
            confidence=round(confidence, 4),
            expected_return=round(expected_return, 6),
            model=self.checkpoint,
            available=True,
        )

    def forecast_dataframe(self, df, horizon: int = 12, column: str = "close") -> ForecastResult:
        """Convenience wrapper to forecast from an OHLCV DataFrame."""
        if column not in df.columns:
            return ForecastResult(
                horizon=horizon,
                last_price=0.0,
                model=self.checkpoint,
                available=False,
                error=f"Column '{column}' not found in DataFrame.",
            )
        return self.forecast(df[column].to_numpy(), horizon=horizon)
