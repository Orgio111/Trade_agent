"""
RL Learning Service — PPO-based capital allocator and strategy evolution.

Provides:
- PPO agent for capital allocation across strategies
- Online learning from trade outcomes
- Strategy performance tracking and evolution
- Portfolio optimization with RL
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StrategyPerformance:
    """Track performance of a strategy."""
    name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    current_allocation: float = 0.0
    target_allocation: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

    def update(self, pnl: float, is_win: bool):
        """Update with new trade result."""
        self.total_trades += 1
        self.total_pnl += pnl
        if is_win:
            self.wins += 1
            self.avg_win = (self.avg_win * (self.wins - 1) + pnl) / self.wins
        else:
            self.losses += 1
            self.avg_loss = (self.avg_loss * (self.losses - 1) + pnl) / self.losses

        self.win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0

        # Update max drawdown
        if pnl < 0:
            self.max_drawdown = max(self.max_drawdown, abs(pnl))

        self._compute_sharpe()
        self.last_update = datetime.now()

    def _compute_sharpe(self):
        """Simplified Sharpe calculation."""
        if self.total_trades > 10:
            returns = [self.avg_win] * self.wins + [self.avg_loss] * self.losses
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            self.sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
        else:
            self.sharpe = 0


@dataclass
class PortfolioState:
    """Current portfolio state for RL."""
    total_value: float
    cash: float
    strategy_allocations: dict[str, float]
    strategy_values: dict[str, float]
    daily_pnl: float
    total_drawdown: float
    positions: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert to state vector for RL."""
        # Fixed strategy order for consistent vector
        strategies = sorted(self.strategy_allocations.keys())
        alloc_vec = np.array([self.strategy_allocations.get(s, 0) for s in strategies])
        value_vec = np.array([self.strategy_values.get(s, 0) for s in strategies])

        return np.concatenate([
            [self.total_value, self.cash, self.daily_pnl, self.total_drawdown],
            alloc_vec,
            value_vec,
        ]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# PPO NETWORK (if torch available)
# ═══════════════════════════════════════════════════════════════════

class PPONetwork(nn.Module if TORCH_AVAILABLE else object):
    """PPO Actor-Critic network for capital allocation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        if TORCH_AVAILABLE:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1),
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.actor = None
            self.critic = None

    def forward(self, state: np.ndarray) -> tuple:
        """Forward pass returning action probs and value."""
        if not TORCH_AVAILABLE:
            # Fallback: uniform distribution
            return np.ones(action_dim) / action_dim, 0.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        value = self.critic(state_tensor)
        return action_probs.detach().numpy()[0], value.item()

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        """Sample action from policy."""
        if not TORCH_AVAILABLE:
            return 0, 1.0, 0.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)

        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

        log_prob = torch.log(action_probs[0, action]).item()
        return action, log_prob, value.item()


# ═══════════════════════════════════════════════════════════════════
# PPO AGENT FOR CAPITAL ALLOCATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    """PPO configuration for capital allocator."""
    # Network
    hidden_dim: int = 128

    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training
    epochs_per_update: int = 4
    batch_size: int = 64
    buffer_size: int = 2048

    # Allocation
    min_allocation: float = 0.0
    max_allocation: float = 1.0
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalance


class PPOCapitalAllocator:
    """
    PPO-based capital allocator across strategies.
    Learns optimal allocation from historical performance.
    """

    def __init__(
        self,
        strategies: list[str],
        config: PPOConfig | None = None,
    ):
        self.strategies = sorted(strategies)
        self.n_strategies = len(strategies)
        self.config = config or PPOConfig()

        # State/action dims
        self.state_dim = 4 + self.n_strategies * 2  # 4 base + alloc + values
        self.action_dim = self.n_strategies  # Allocation to each strategy

        if TORCH_AVAILABLE:
            self.network = PPONetwork(self.state_dim, self.action_dim, self.config.hidden_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        else:
            self.network = None
            self.optimizer = None

        # Replay buffer
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_values = []
        self.buffer_log_probs = []
        self.buffer_dones = []

        # Performance tracking
        self.strategy_performance: dict[str, StrategyPerformance] = {
            s: StrategyPerformance(name=s) for s in self.strategies
        }

        # Allocation
        self.current_allocation = {s: 1.0 / self.n_strategies for s in self.strategies}
        self.target_allocation = self.current_allocation.copy()

        # Training state
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0

        # Model persistence
        self.model_path = "models/ppo_allocator.pkl"
        os.makedirs("models", exist_ok=True)
        self._load_model()

    def update_performance(self, strategy: str, pnl: float, is_win: bool):
        """Update strategy performance from trade result."""
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy].update(pnl, is_win)

    def get_state(self, portfolio: PortfolioState) -> np.ndarray:
        """Convert portfolio to state vector."""
        return portfolio.to_vector()

    def allocate(self, portfolio: PortfolioState) -> dict[str, float]:
        """
        Get target allocation from PPO policy.
        Returns dictionary of strategy -> target allocation fraction.
        """
        if not TORCH_AVAILABLE or self.network is None:
            # Fallback: equal weight with performance bias
            return self._fallback_allocation()

        state = self.get_state(portfolio)
        action_probs, _ = self.network.forward(state)

        # Apply constraints
        action_probs = np.clip(action_probs, self.config.min_allocation, self.config.max_allocation)
        action_probs = action_probs / action_probs.sum()

        # Update target allocation
        self.target_allocation = {s: float(action_probs[i]) for i, s in enumerate(self.strategies)}

        return self.target_allocation

    def _fallback_allocation(self) -> dict[str, float]:
        """Fallback allocation using performance-based weights."""
        weights = {}
        for s in self.strategies:
            perf = self.strategy_performance[s]
            if perf.total_trades > 5:
                # Weight by Sharpe * win_rate
                weight = max(perf.sharpe, 0) * perf.win_rate
            else:
                weight = 1.0

            weights[s] = max(weight, 0.1)

        # Normalize
        total = sum(weights.values())
        return {s: w / total for s, w in weights.items()}

    def should_rebalance(self, portfolio: PortfolioState) -> bool:
        """Check if rebalancing is needed."""
        for s in self.strategies:
            current = portfolio.strategy_allocations.get(s, 0)
            target = self.target_allocation.get(s, 0)
            if abs(current - target) > self.config.rebalance_threshold:
                return True
        return False

    def record_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Record transition in replay buffer."""
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_next_states.append(next_state)
        self.buffer_dones.append(done)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

        self.steps += 1
        self.total_reward += reward

        # Trigger update if buffer full
        if len(self.buffer_states) >= self.config.buffer_size:
            self.update()

    def update(self):
        """PPO update from buffer."""
        if not TORCH_AVAILABLE or len(self.buffer_states) < self.config.batch_size:
            return

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer_states))
        actions = torch.LongTensor(self.buffer_actions)
        rewards = torch.FloatTensor(self.buffer_rewards)
        next_states = torch.FloatTensor(np.array(self.buffer_next_states))
        dones = torch.BoolTensor(self.buffer_dones)
        old_log_probs = torch.FloatTensor(self.buffer_log_probs)
        old_values = torch.FloatTensor(self.buffer_values)

        # Compute returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0

        with torch.no_grad():
            next_values = self.network.critic(torch.FloatTensor(next_states)).squeeze()

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = old_values[i + 1]

            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - old_values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            returns.insert(0, gae + old_values[i])
            advantages.insert(0, gae)

        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(self.config.epochs_per_update):
            # Sample minibatch
            indices = torch.randperm(len(states))[:self.config.batch_size]

            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_returns = returns[indices]
            batch_advantages = advantages[indices]

            # Forward
            action_probs = self.network.actor(batch_states)
            new_log_probs = torch.log(action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze())
            new_values = self.network.critic(batch_states).squeeze()

            # Ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # Clipped surrogate
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(new_values, batch_returns)

            # Entropy bonus
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            entropy_loss = -self.config.entropy_coef * entropy

            # Total loss
            loss = actor_loss + self.config.value_coef * critic_loss + entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # Clear buffer
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_next_states.clear()
        self.buffer_dones.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

        logger.info(f"[PPOAllocator] Update complete. Steps: {self.steps}")

    def compute_reward(self, portfolio: PortfolioState, prev_portfolio: PortfolioState) -> float:
        """Compute reward from portfolio change."""
        # Primary: portfolio return
        portfolio_return = (portfolio.total_value - prev_portfolio.total_value) / prev_portfolio.total_value

        # Penalty: drawdown
        drawdown_penalty = -portfolio.total_drawdown * 2

        # Penalty: high volatility (measured by allocation changes)
        alloc_change = sum(
            abs(portfolio.strategy_allocations.get(s, 0) - prev_portfolio.strategy_allocations.get(s, 0))
            for s in self.strategies
        )
        turnover_penalty = -alloc_change * 0.1

        # Bonus: positive Sharpe
        sharpe_bonus = 0
        for s in self.strategies:
            perf = self.strategy_performance[s]
            if perf.sharpe > 1.0:
                sharpe_bonus += 0.01

        return portfolio_return + drawdown_penalty + turnover_penalty + sharpe_bonus

    def save_model(self):
        """Save model to disk."""
        if not TORCH_AVAILABLE:
            return

        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'strategies': self.strategies,
            'config': self.config,
            'current_allocation': self.current_allocation,
            'strategy_performance': self.strategy_performance,
        }, self.model_path)
        logger.info(f"[PPOAllocator] Model saved to {self.model_path}")

    def _load_model(self):
        """Load model from disk."""
        if not TORCH_AVAILABLE or not os.path.exists(self.model_path):
            return

        try:
            checkpoint = torch.load(self.model_path)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_allocation = checkpoint.get('current_allocation', self.current_allocation)
            self.strategy_performance = checkpoint.get('strategy_performance', self.strategy_performance)
            logger.info(f"[PPOAllocator] Model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"[PPOAllocator] Failed to load model: {e}")


# ═══════════════════════════════════════════════════════════════════
# STRATEGY EVOLUTION (Genetic Algorithm + RL)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvolvedStrategy:
    """Evolved strategy with parameters."""
    name: str
    parameters: dict[str, float]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class StrategyEvolver:
    """
    Genetic algorithm for strategy parameter evolution.
    Combines with RL for meta-learning.
    """

    def __init__(
        self,
        base_strategies: list[str],
        param_bounds: dict[str, tuple[float, float, float]] | None = None,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
    ):
        self.base_strategies = base_strategies
        self.param_bounds = param_bounds or self._default_bounds()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        self.population: list[EvolvedStrategy] = []
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_strategy = None

    def _default_bounds(self) -> dict[str, tuple[float, float]]:
        """Default parameter bounds for common strategies."""
        bounds = {}
        for s in self.base_strategies:
            bounds[f"{s}_ema_fast"] = (5, 20)
            bounds[f"{s}_ema_slow"] = (20, 60)
            bounds[f"{s}_rsi_period"] = (7, 21)
            bounds[f"{s}_rsi_overbought"] = (65, 85)
            bounds[f"{s}_rsi_oversold"] = (15, 35)
            bounds[f"{s}_bb_period"] = (10, 30)
            bounds[f"{s}_bb_std"] = (1.5, 3.0)
        return bounds

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for i in range(self.population_size):
            params = {}
            for param, (low, high) in self.param_bounds.items():
                params[param] = random.uniform(low, high)

            self.population.append(EvolvedStrategy(
                name=f"gen0_strat_{i}",
                parameters=params,
                generation=0,
            ))
        logger.info(f"[StrategyEvolver] Initialized population of {self.population_size}")

    def evaluate_population(self, backtest_fn: Callable[[dict], float]):
        """Evaluate all strategies via backtesting."""
        for strat in self.population:
            fitness = backtest_fn(strat.parameters)
            strat.fitness = fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_strategy = strat
                logger.info(f"[StrategyEvolver] New best: {strat.name} fitness={fitness:.4f}")

    def evolve(self):
        """Create next generation."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Keep elite
        new_population = self.population[:self.elite_size]

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                # Mutation only
                parent = self._tournament_select()
                child = self._mutate(parent)

            new_population.append(child)

        self.population = new_population
        self.generation += 1
        logger.info(f"[StrategyEvolver] Generation {self.generation} complete. Best: {self.best_fitness:.4f}")

    def _tournament_select(self, k: int = 3) -> EvolvedStrategy:
        """Tournament selection."""
        contestants = random.sample(self.population, k)
        return max(contestants, key=lambda x: x.fitness)

    def _crossover(self, p1: EvolvedStrategy, p2: EvolvedStrategy) -> EvolvedStrategy:
        """Uniform crossover."""
        child_params = {}
        for param in self.param_bounds:
            child_params[param] = p1.parameters[param] if random.random() < 0.5 else p2.parameters[param]

        return EvolvedStrategy(
            name=f"gen{self.generation}_cross_{p1.name}_{p2.name}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[p1.name, p2.name],
        )

    def _mutate(self, parent: EvolvedStrategy) -> EvolvedStrategy:
        """Gaussian mutation."""
        child_params = {}
        for param, (low, high) in self.param_bounds.items():
            val = parent.parameters[param]
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                sigma = (high - low) * 0.1
                val = val + random.gauss(0, sigma)
                val = max(low, min(high, val))
            child_params[param] = val

        return EvolvedStrategy(
            name=f"gen{self.generation}_mut_{parent.name}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[parent.name],
        )

    def get_best_parameters(self) -> dict[str, float] | None:
        """Get best strategy parameters."""
        return self.best_strategy.parameters if self.best_strategy else None


# ═══════════════════════════════════════════════════════════════════
# RL LEARNING SERVICE (Integration)
# ═══════════════════════════════════════════════════════════════════

class RLLearningService:
    """
    Integrated RL learning service.
    Combines PPO capital allocator + Strategy evolution.
    """

    def __init__(
        self,
        strategies: list[str],
        ppo_config: PPOConfig | None = None,
        evolve_config: dict | None = None,
    ):
        self.strategies = strategies
        self.ppo = PPOCapitalAllocator(strategies, ppo_config)

        # Strategy evolution
        if evolve_config:
            self.evolver = StrategyEvolver(strategies, **evolve_config)
            self.evolver.initialize_population()
        else:
            self.evolver = None

        # State
        self.portfolio_history: deque = deque(maxlen=1000)
        self._last_portfolio: PortfolioState | None = None
        self._training_task: asyncio.Task | None = None

    def update_from_trade(self, strategy: str, pnl: float, is_win: bool):
        """Update from trade result."""
        self.ppo.update_performance(strategy, pnl, is_win)

    def get_allocation(self, portfolio: PortfolioState) -> dict[str, float]:
        """Get target allocation."""
        return self.ppo.allocate(portfolio)

    def record_portfolio(self, portfolio: PortfolioState):
        """Record portfolio state for RL."""
        if self._last_portfolio is not None:
            # Compute reward
            reward = self.ppo.compute_reward(portfolio, self._last_portfolio)

            # Record transition
            prev_state = self._last_portfolio.to_vector()
            next_state = portfolio.to_vector()

            # Simple: use target allocation as action
            action = list(self.ppo.target_allocation.values()).index(
                max(self.ppo.target_allocation.values())
            ) if self.ppo.target_allocation else 0

            self.ppo.record_step(
                prev_state, action, reward, next_state, False, 0.0, 0.0
            )

        self._last_portfolio = portfolio
        self.portfolio_history.append(portfolio)

    async def start_training(self, interval_seconds: int = 300):
        """Start background training."""
        async def train_loop():
            while True:
                try:
                    # Periodic model save
                    self.ppo.save_model()
                    logger.info("[RLService] Periodic model save")

                    # Evolution step if available
                    if self.evolver and len(self.evolver.population) > 0:
                        self.evolver.evolve()
                        best_params = self.evolver.get_best_parameters()
                        if best_params:
                            logger.info(f"[RLService] Best evolved params: {best_params}")

                except Exception as e:
                    logger.error(f"[RLService] Training error: {e}")

                await asyncio.sleep(interval_seconds)

        self._training_task = asyncio.create_task(train_loop())

    async def stop_training(self):
        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass

    def get_status(self) -> dict:
        """Get service status."""
        return {
            "strategies": self.strategies,
            "current_allocation": self.ppo.current_allocation,
            "target_allocation": self.ppo.target_allocation,
            "performance": {
                s: {
                    "trades": p.total_trades,
                    "win_rate": p.win_rate,
                    "sharpe": p.sharpe,
                    "allocation": p.current_allocation,
                }
                for s, p in self.ppo.strategy_performance.items()
            },
            "ppo_steps": self.ppo.steps,
            "evolution_generation": self.evolver.generation if self.evolver else 0,
            "best_evolved_fitness": self.evolver.best_fitness if self.evolver else 0,
        }


def create_rl_learning_service(
    strategies: list[str],
    ppo_config: PPOConfig | None = None,
    evolve_config: dict | None = None,
) -> RLLearningService:
    """Factory for RLLearningService."""
    return RLLearningService(strategies, ppo_config, evolve_config)


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test PPO allocator
        strategies = ["trend", "scalping", "mean_reversion", "breakout"]
        allocator = create_rl_learning_service(strategies)

        # Mock portfolio
        portfolio = PortfolioState(
            total_value=100000,
            cash=20000,
            strategy_allocations={"trend": 0.3, "scalping": 0.3, "mean_reversion": 0.2, "breakout": 0.2},
            strategy_values={"trend": 30000, "scalping": 30000, "mean_reversion": 20000, "breakout": 20000},
            daily_pnl=500,
            total_drawdown=0.02,
        )

        allocation = allocator.get_allocation(portfolio)
        print(f"Allocation: {allocation}")

        # Update performance
        allocator.update_from_trade("trend", 100, True)
        allocator.update_from_trade("scalping", -50, False)

        # Record portfolio
        new_portfolio = PortfolioState(
            total_value=100500,
            cash=19500,
            strategy_allocations={"trend": 0.31, "scalping": 0.29, "mean_reversion": 0.2, "breakout": 0.2},
            strategy_values={"trend": 31000, "scalping": 29000, "mean_reversion": 20000, "breakout": 20000},
            daily_pnl=500,
            total_drawdown=0.019,
        )
        allocator.record_portfolio(new_portfolio)

        print(f"Status: {allocator.get_status()}")

    asyncio.run(test())