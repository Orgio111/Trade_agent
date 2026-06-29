---
title: PPO Portfolio Manager
type: entity
tags: [ppo, reinforcement-learning, portfolio, allocation, markowitz, rl]
created: 2026-06-30
updated: 2026-06-30
source_file: ppo_portfolio_manager.py
status: active
---

# PPO Portfolio Manager

## Overview

Reinforcement learning-based capital allocation system. Uses a Proximal Policy Optimization (PPO) policy trained on historical returns to learn optimal capital distribution across N assets. Falls back to Markowitz, Risk Parity, or Equal Weight when PPO is unavailable.

## Architecture

```
PortfolioAllocEnv (gymnasium) ──→ SB3 PPO ──→ Allocation Weights
       │                              ↑
       │                              │
  Market returns               PPOPortfolioManager
  + portfolio state            (PPO + Markowitz fallback)
```

## Files

| File | Purpose |
|------|---------|
| `orchestrator/rl/portfolio_env.py` | `PortfolioAllocEnv` (gym.Env) — state ~30+N*6 dims, softmax-normalized actions |
| `orchestrator/ppo_portfolio_manager.py` | `PPOPortfolioManager` — SB3 PPO wrapper with allocate/fallback methods |
| `orchestrator/train_ppo_portfolio.py` | Training script — synthetic/CSV data, CLI interface, eval comparison |

## Training

```bash
# Train with synthetic data (4 assets, 100K timesteps)
python -m orchestrator.train_ppo_portfolio

# Train with historical returns CSV
python -m orchestrator.train_ppo_portfolio --data path/to/returns.csv --timesteps 200000 --eval

# Quick benchmark (10K timesteps)
python -m orchestrator.train_ppo_portfolio --benchmark
```

## Inference

```python
from orchestrator import PPOPortfolioManager

manager = PPOPortfolioManager(n_assets=4, asset_names=["BTC", "ETH", "SOL", "USDC"])
manager._load_model()

# PPO allocation
result = manager.allocate(returns_df, confidence=0.5)
print(result.weights)   # {"BTC": 0.3, "ETH": 0.25, ...}
print(result.method)    # "ppo" or "markowitz" (fallback)

# Blended allocation (60% PPO + 40% Markowitz)
result = manager.allocate_blended(returns_df, ppo_weight=0.6)
```

## PortfolioAllocEnv State Space

```
State (~30+N*6 dims):
  [asset_features × N]      : ret_1, ret_5, volatility, momentum, vol_ratio, corr
  [portfolio_features × 7]  : drawdown, sharpe, win_rate, consec_losses, pnl%, trades, exposure
  [current_weights × N]     : current allocation vector

Action (continuous Box):
  N_asset weights (softmax → sum ≈ 1.0)

Reward:
  port_return - turnover_penalty - concentration_penalty - drawdown_penalty
```

## Fallback Chain

```
PPO (trained) ──→ Markowitz ──→ Risk Parity ──→ Equal Weight
```

The `PPOPortfolioManager` automatically falls back through this chain if PPO is unavailable.

## How it fits in the system

- Part of the RL module (`orchestrator/rl/`) — alongside `TradingEnv`, `PortfolioEnv`, `StrategyEvolver`
- Feeds allocation decisions into the [[broker-abstraction-layer]]
- State space includes features from [[brain-ecosystem]] brain outputs
- Training uses [[nats-event-system]] RL events for reward signals

## Related

- [[brain-ecosystem]] — brain signals inform portfolio allocation
- [[broker-abstraction-layer]] — executes allocation decisions
- [[signal-aggregation-logic]] — aggregated signals feed into allocation
- [[local-trading-ai-architecture]] — overall system architecture

## Sources

- Internal code: `orchestrator/ppo_portfolio_manager.py`, `orchestrator/rl/portfolio_env.py`
- Internal design: AGENTS.md §PPO PORTFOLIO MANAGER
