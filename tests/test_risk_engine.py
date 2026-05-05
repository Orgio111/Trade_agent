"""Unit tests for risk_engine.py — no network calls needed."""
from __future__ import annotations

import numpy as np
import pytest

from agents.risk_engine import (
    fractional_kelly,
    historical_var,
    parametric_var,
)


class TestHistoricalVaR:
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        var95, _ = historical_var(returns, 0.95)
        var99, cvar99 = historical_var(returns, 0.99)
        assert var95 > 0
        assert var99 > var95, "VaR99 should be > VaR95"
        assert cvar99 >= var99, "CVaR should be >= VaR"

    def test_thin_data_returns_conservative_defaults(self):
        returns = np.array([0.01, -0.02, 0.005])
        var99, cvar99 = historical_var(returns, 0.99)
        assert var99 == pytest.approx(0.05)

    def test_var_positive(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(-0.005, 0.03, 300)
        var99, _ = historical_var(returns, 0.99)
        assert var99 > 0, "VaR must always be a positive loss"


class TestKelly:
    def test_positive_edge(self):
        k = fractional_kelly(0.6, 0.02, 0.01, 1.0)
        assert k > 0

    def test_no_edge(self):
        k = fractional_kelly(0.5, 0.01, 0.01, 1.0)
        assert k == pytest.approx(0.0)

    def test_fractional_reduces_size(self):
        k_full = fractional_kelly(0.6, 0.02, 0.01, 1.0)
        k_quarter = fractional_kelly(0.6, 0.02, 0.01, 0.25)
        assert k_quarter == pytest.approx(k_full * 0.25)

    def test_zero_avg_loss_returns_zero(self):
        k = fractional_kelly(0.6, 0.02, 0.0, 0.25)
        assert k == 0.0

    def test_never_negative(self):
        k = fractional_kelly(0.1, 0.005, 0.05, 0.25)
        assert k == 0.0, "Kelly must not go negative"


class TestParametricVaR:
    def test_consistent_with_historical(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.02, 1000)
        h_var, _ = historical_var(returns, 0.99)
        p_var = parametric_var(returns, 0.99)
        # Should be within 50% of each other for normal data
        assert abs(h_var - p_var) / h_var < 0.5
