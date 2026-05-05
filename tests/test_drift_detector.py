"""Unit tests for PSI-based drift detector."""
from __future__ import annotations

import numpy as np
import pytest

from mlops.drift_detector import DriftDetector, compute_psi


class TestPSI:
    def test_identical_distributions_zero_psi(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_high_psi(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)  # large shift
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_small_shift_moderate_psi(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0.5, 1, 1000)
        psi = compute_psi(ref, cur)
        assert 0.05 < psi < 0.5

    def test_psi_non_negative(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(1, 2, 500)
        psi = compute_psi(ref, cur)
        assert psi >= 0


class TestDriftDetector:
    def test_no_drift_detected(self):
        detector = DriftDetector("test_model")
        rng = np.random.default_rng(42)
        ref = {"returns": rng.normal(0, 0.02, 500)}
        detector.set_reference(ref)
        cur = {"returns": rng.normal(0, 0.02, 500)}
        assert not detector.needs_retrain(cur)

    def test_drift_detected(self):
        detector = DriftDetector("test_model")
        rng = np.random.default_rng(42)
        ref = {"returns": rng.normal(0, 0.01, 500)}
        detector.set_reference(ref)
        cur = {"returns": rng.normal(0.5, 0.05, 500)}
        assert detector.needs_retrain(cur)

    def test_missing_feature_skipped(self):
        detector = DriftDetector("test_model")
        ref = {"feature_a": np.random.randn(200)}
        detector.set_reference(ref)
        scores = detector.check({"feature_b": np.random.randn(200)})
        assert scores == {}
