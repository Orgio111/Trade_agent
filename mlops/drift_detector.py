"""
Population Stability Index (PSI) + Kolmogorov-Smirnov based model drift detector.
Triggers retraining when PSI > threshold or KS p-value < threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import get_settings
from core.observability import PSI_GAUGE, RETRAIN_COUNTER

logger = logging.getLogger(__name__)


# ── Drift report ──────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """Result of a drift check across all features."""
    psi_scores: dict[str, float] = field(default_factory=dict)
    ks_scores: dict[str, float] = field(default_factory=dict)
    ks_pvalues: dict[str, float] = field(default_factory=dict)
    drift_detected: bool = False
    drift_features: list[str] = field(default_factory=list)
    mean_psi: float = 0.0
    max_psi: float = 0.0
    drift_severity: str = "none"  # none | moderate | significant


# ── PSI computation ───────────────────────────────────────────────────────────

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


# ── KS test ───────────────────────────────────────────────────────────────────

def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Returns (D_statistic, p_value).
    D > 0.2 with p < 0.05 indicates significant distribution shift.
    """
    from scipy.stats import ks_2samp  # type: ignore[import]

    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


# ── DriftDetector ─────────────────────────────────────────────────────────────

class DriftDetector:
    """Monitors feature distributions via PSI + KS and triggers retraining on drift."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._reference: dict[str, np.ndarray] = {}
        self._last_report: DriftReport | None = None

    def set_reference(self, features: dict[str, np.ndarray]) -> None:
        self._reference = {k: v.copy() for k, v in features.items()}
        logger.info(
            "Drift detector reference set for %s (%d features)",
            self.model_name, len(features),
        )

    @property
    def has_reference(self) -> bool:
        return len(self._reference) > 0

    def check(self, current_features: dict[str, np.ndarray]) -> DriftReport:
        """Run PSI + KS on each feature and return a DriftReport."""
        cfg = get_settings()
        report = DriftReport()
        drift_features: list[str] = []

        for name, ref in self._reference.items():
            if name not in current_features:
                continue
            cur = current_features[name]
            if len(cur) < 10:
                continue

            # PSI
            psi = compute_psi(ref, cur)
            report.psi_scores[name] = psi
            PSI_GAUGE.labels(model=f"{self.model_name}.{name}.psi").set(psi)

            # KS
            ks_stat, ks_pval = compute_ks(ref, cur)
            report.ks_scores[name] = ks_stat
            report.ks_pvalues[name] = ks_pval
            PSI_GAUGE.labels(model=f"{self.model_name}.{name}.ks").set(ks_stat)

            # Drift logic: PSI > threshold OR (KS > 0.2 AND p < 0.05)
            threshold = cfg.psi_drift_threshold
            psi_drift = psi > threshold
            ks_drift = ks_stat > 0.2 and ks_pval < 0.05

            if psi_drift or ks_drift:
                drift_features.append(name)
                logger.warning(
                    "DRIFT DETECTED: %s.%s  PSI=%.3f (threshold=%.2f)  KS=%.3f (p=%.4f)",
                    self.model_name, name, psi, threshold, ks_stat, ks_pval,
                )

        report.drift_detected = len(drift_features) > 0
        report.drift_features = drift_features
        report.mean_psi = float(np.mean(list(report.psi_scores.values()) or [0]))
        report.max_psi = float(np.max(list(report.psi_scores.values()) or [0]))

        # Severity classification
        if report.max_psi > cfg.psi_drift_threshold:
            report.drift_severity = "significant"
        elif report.max_psi > cfg.psi_drift_threshold * 0.5:
            report.drift_severity = "moderate"

        self._last_report = report

        if report.drift_detected:
            RETRAIN_COUNTER.labels(model=self.model_name).inc()

        return report

    def needs_retrain(self, current_features: dict[str, np.ndarray]) -> bool:
        return self.check(current_features).drift_detected
