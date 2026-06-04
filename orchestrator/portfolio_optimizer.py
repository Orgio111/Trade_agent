"""
QUANTEX Portfolio Optimizer — Kelly Criterion and Markowitz Mean-Variance optimization.

Provides:
  - Half-Kelly position sizing for individual trades
  - Markowitz efficient frontier for multi-asset allocation
  - Risk-budgeting (equal risk contribution)
  - Dynamic rebalancing based on correlation changes

Usage:
    optimizer = PortfolioOptimizer()
    
    # Half-Kelly for single trade
    size = optimizer.kelly_size(win_rate=0.55, avg_win=2.0, avg_loss=1.0)
    
    # Markowitz for portfolio
    weights = optimizer.markowitz_allocation(returns_df)
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    weights: dict[str, float]       # symbol -> weight
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: str                     # "kelly", "markowitz", "risk_parity"


class PortfolioOptimizer:
    """
    Portfolio optimization using Kelly Criterion and Markowitz Mean-Variance.

    Kelly Criterion:
      f* = (p * b - q) / b
      where p = win probability, q = loss probability, b = win/loss ratio
      Half-Kelly: use f*/2 for safety

    Markowitz:
      Maximize Sharpe = (w'μ - rf) / sqrt(w'Σw)
      subject to sum(w) = 1, w >= 0 (long-only)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    # ── Kelly Criterion ───────────────────────────────────────

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        half_kelly: bool = True,
        max_fraction: float = 0.25,
    ) -> dict:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade PnL (positive)
            avg_loss: Average losing trade PnL (positive, for ratio)
            half_kelly: Use half-Kelly for safety (default True)
            max_fraction: Maximum fraction of capital to risk

        Returns:
            dict with kelly_fraction, recommended_fraction, edge, odds
        """
        if win_rate <= 0 or win_rate >= 1:
            return {"kelly_fraction": 0.0, "recommended_fraction": 0.0, "edge": 0.0, "odds": 0.0}

        # Win/loss ratio (b in Kelly formula)
        if avg_loss <= 0:
            return {"kelly_fraction": 0.0, "recommended_fraction": 0.0, "edge": 0.0, "odds": 0.0}

        odds = avg_win / avg_loss
        q = 1.0 - win_rate

        # Full Kelly: f* = (p * b - q) / b
        kelly = (win_rate * odds - q) / odds if odds > 0 else 0.0

        # Apply half-Kelly for safety
        recommended = kelly * 0.5 if half_kelly else kelly

        # Cap at max_fraction
        recommended = max(0.0, min(recommended, max_fraction))

        return {
            "kelly_fraction": round(kelly, 4),
            "recommended_fraction": round(recommended, 4),
            "edge": round(win_rate - q / odds if odds > 0 else 0, 4),
            "odds": round(odds, 4),
            "half_kelly": half_kelly,
        }

    # ── Markowitz Mean-Variance ───────────────────────────────

    def markowitz_allocation(
        self,
        returns: pd.DataFrame,
        max_weight: float = 0.4,
        min_weight: float = 0.0,
    ) -> PortfolioAllocation:
        """
        Compute Markowitz mean-variance optimal allocation.

        Uses scipy.optimize to maximize Sharpe ratio.

        Args:
            returns: DataFrame of asset returns (columns = symbols)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            PortfolioAllocation with optimal weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback: equal weight
            n = len(returns.columns)
            w = {col: 1.0 / n for col in returns.columns}
            return PortfolioAllocation(
                weights=w,
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                method="equal_weight_fallback",
            )

        if returns.empty or len(returns.columns) < 1:
            return PortfolioAllocation(
                weights={}, expected_return=0.0, expected_risk=0.0,
                sharpe_ratio=0.0, method="empty",
            )

        # Calculate expected returns and covariance
        mu = returns.mean() * 252  # Annualized
        sigma = returns.cov() * 252  # Annualized
        n_assets = len(returns.columns)

        if n_assets == 1:
            symbol = returns.columns[0]
            exp_ret = float(mu.iloc[0])
            risk = float(np.sqrt(sigma.iloc[0, 0]))
            sharpe = (exp_ret - self.risk_free_rate) / risk if risk > 0 else 0
            return PortfolioAllocation(
                weights={symbol: 1.0},
                expected_return=round(exp_ret, 4),
                expected_risk=round(risk, 4),
                sharpe_ratio=round(sharpe, 4),
                method="single_asset",
            )

        # Objective: negative Sharpe ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_risk <= 0:
                return 0.0
            return -(port_return - self.risk_free_rate) / port_risk

        # Constraints: sum(weights) = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

        # Initial guess: equal weight
        w0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            # Fall back to equal weight
            w = {col: 1.0 / n_assets for col in returns.columns}
            return PortfolioAllocation(
                weights=w, expected_return=0.0, expected_risk=0.0,
                sharpe_ratio=0.0, method="optimization_failed",
            )

        weights = result.x
        port_return = np.dot(weights, mu)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0

        return PortfolioAllocation(
            weights={col: round(float(w), 4) for col, w in zip(returns.columns, weights)},
            expected_return=round(float(port_return), 4),
            expected_risk=round(float(port_risk), 4),
            sharpe_ratio=round(float(sharpe), 4),
            method="markowitz_sharpe",
        )

    # ── Risk Parity ───────────────────────────────────────────

    def risk_parity_allocation(
        self,
        returns: pd.DataFrame,
        max_weight: float = 0.4,
    ) -> PortfolioAllocation:
        """
        Equal risk contribution (risk parity) allocation.

        Each asset contributes equally to portfolio risk.

        Args:
            returns: DataFrame of asset returns
            max_weight: Maximum weight per asset

        Returns:
            PortfolioAllocation
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            n = len(returns.columns)
            w = {col: 1.0 / n for col in returns.columns}
            return PortfolioAllocation(
                weights=w, expected_return=0.0, expected_risk=0.0,
                sharpe_ratio=0.0, method="equal_weight_fallback",
            )

        if returns.empty or len(returns.columns) < 2:
            return self.markowitz_allocation(returns)

        sigma = returns.cov() * 252
        n = len(returns.columns)

        def risk_contribution(weights):
            port_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_risk <= 0:
                return np.zeros(n)
            return (weights * np.dot(sigma, weights)) / port_risk

        def risk_parity_objective(weights):
            rc = risk_contribution(weights)
            target = 1.0 / n
            return np.sum((rc - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, max_weight) for _ in range(n)]
        w0 = np.array([1.0 / n] * n)

        result = minimize(
            risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            w = {col: 1.0 / n for col in returns.columns}
            return PortfolioAllocation(
                weights=w, expected_return=0.0, expected_risk=0.0,
                sharpe_ratio=0.0, method="risk_parity_failed",
            )

        weights = result.x
        port_return = np.dot(weights, returns.mean() * 252)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0

        return PortfolioAllocation(
            weights={col: round(float(w), 4) for col, w in zip(returns.columns, weights)},
            expected_return=round(float(port_return), 4),
            expected_risk=round(float(port_risk), 4),
            sharpe_ratio=round(float(sharpe), 4),
            method="risk_parity",
        )

    # ── Convenience ───────────────────────────────────────────

    def compute_trade_size(
        self,
        balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_risk_pct: float = 0.02,
    ) -> dict:
        """
        Compute optimal trade dollar amount.

        Combines Kelly with balance and max risk constraints.

        Args:
            balance: Current account balance
            win_rate: Historical win rate
            avg_win: Average winning trade PnL
            avg_loss: Average losing trade PnL
            max_risk_pct: Maximum risk per trade as % of balance

        Returns:
            dict with dollar_amount, fraction, and details
        """
        kelly = self.kelly_size(win_rate, avg_win, avg_loss)
        fraction = kelly["recommended_fraction"]

        # Apply max risk constraint
        dollar_amount = balance * fraction
        max_risk_amount = balance * max_risk_pct

        # If Kelly suggests more than max risk, cap it
        if dollar_amount > max_risk_amount:
            fraction = max_risk_pct
            dollar_amount = max_risk_amount

        return {
            "dollar_amount": round(dollar_amount, 2),
            "fraction_of_balance": round(fraction, 4),
            "max_risk_amount": round(max_risk_amount, 2),
            "kelly_details": kelly,
        }
