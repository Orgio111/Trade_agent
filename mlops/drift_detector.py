"""
Population Stability Index (PSI) based model drift detector.
Triggers retraining when PSI > 0.2 (significant distribution shift).
"""
from __future__ import annotations

import logging

import numpy as np

from core.config import get_settings
from core.observability import PSI_GAUGE, RETRAIN_COUNTER

logger = logging.getLogger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    < 0.1 : no drift
    0.1-0.2: moderate drift
    > 0.2 : significant drift → retrain
    """
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_pct, _ = np.histogram(reference, bins=bins)
    cur_pct, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_pct + epsilon) / (len(reference) + n_bins * epsilon)
    cur_pct = (cur_pct + epsilon) / (len(current) + n_bins * epsilon)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


class DriftDetector:
    """Monitors feature distributions and triggers retraining on drift."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._reference: dict[str, np.ndarray] = {}

    def set_reference(self, features: dict[str, np.ndarray]) -> None:
        self._reference = {k: v.copy() for k, v in features.items()}
        logger.info("Drift detector reference set for %s (%d features)", self.model_name, len(features))

    def check(self, current_features: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = get_settings()
        psi_scores: dict[str, float] = {}
        drift_detected = False

        for name, ref in self._reference.items():
            if name not in current_features:
                continue
            cur = current_features[name]
            if len(cur) < 10:
                continue
            psi = compute_psi(ref, cur)
            psi_scores[name] = psi
            PSI_GAUGE.labels(model=f"{self.model_name}.{name}").set(psi)

            if psi > cfg.psi_drift_threshold:
                logger.warning(
                    "DRIFT DETECTED: %s.%s PSI=%.3f > threshold %.2f",
                    self.model_name,
                    name,
                    psi,
                    cfg.psi_drift_threshold,
                )
                drift_detected = True

        if drift_detected:
            RETRAIN_COUNTER.labels(model=self.model_name).inc()

        return psi_scores

    def needs_retrain(self, current_features: dict[str, np.ndarray]) -> bool:
        cfg = get_settings()
        scores = self.check(current_features)
        return any(v > cfg.psi_drift_threshold for v in scores.values())
