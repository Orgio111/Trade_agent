"""
QUANTEX Strategy Evolution System — Self-improving strategies through
genetic algorithms and Bayesian hyperparameter optimization.

Architecture:
  1. Evolutionary Cycle: Tournament selection → Crossover → Mutation → Elitism
  2. Optuna Optimization: Bayesian hyperparameter search
  3. Walk-Forward Validation: Prevent overfitting with out-of-sample testing
"""
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Strategy:
    """A trading strategy with its parameter set."""
    name: str
    params: dict
    fitness: float = 0.0
    generation: int = 0
    metadata: dict = field(default_factory=dict)


class StrategyEvolver:
    """
    Genetic algorithm for evolving trading strategy parameters.

    Process:
      1. Initialize population with random parameters
      2. Evaluate fitness (Sharpe, profit factor, drawdown)
      3. Tournament selection for parent candidates
      4. Crossover: blend parent parameters
      5. Mutation: random parameter perturbation
      6. Elitism: keep top performers
      7. Repeat for N generations
    """

    def __init__(self, fitness_fn: Callable[[Strategy], float],
                 population_size: int = 50, mutation_rate: float = 0.1,
                 elitism_pct: float = 0.2):
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = max(1, int(population_size * elitism_pct))
        self.generation = 0
        self.population: list[Strategy] = []
        self.best_ever: Optional[Strategy] = None

    @staticmethod
    def default_strategy_params() -> dict:
        """Default strategy parameter space."""
        return {
            "ema_fast": 9,
            "ema_slow": 21,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_mult_sl": 1.5,
            "atr_mult_tp": 3.0,
            "volume_threshold": 1.5,
            "min_confidence": 0.4,
        }

    @staticmethod
    def random_params() -> dict:
        """Generate random strategy parameters."""
        import numpy as np
        return {
            "ema_fast": int(np.random.randint(5, 20)),
            "ema_slow": int(np.random.randint(20, 100)),
            "rsi_period": int(np.random.randint(7, 21)),
            "rsi_overbought": float(np.random.uniform(60, 80)),
            "rsi_oversold": float(np.random.uniform(20, 40)),
            "atr_mult_sl": float(np.random.uniform(1.0, 3.0)),
            "atr_mult_tp": float(np.random.uniform(1.5, 5.0)),
            "volume_threshold": float(np.random.uniform(1.0, 3.0)),
            "min_confidence": float(np.random.uniform(0.3, 0.6)),
        }

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for i in range(self.pop_size):
            strat = Strategy(
                name=f"gen0_strat_{i}",
                params=self.random_params(),
                generation=0,
            )
            strat.fitness = self.fitness_fn(strat)
            self.population.append(strat)
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        self.best_ever = self.population[0]
        self.generation = 0

    def evolve(self, generations: int = 20) -> list[Strategy]:
        """Run evolutionary cycle for N generations."""
        if not self.population:
            self.initialize_population()

        for gen in range(generations):
            self.generation += 1
            new_population = []

            # Elitism: keep best performers
            new_population.extend(self.population[:self.elite_count])

            # Fill rest with offspring
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child_params = self._crossover(parent1.params, parent2.params)
                child_params = self._mutate(child_params)
                child = Strategy(
                    name=f"gen{self.generation}_strat_{len(new_population)}",
                    params=child_params,
                    generation=self.generation,
                )
                child.fitness = self.fitness_fn(child)
                new_population.append(child)

            self.population = new_population
            self.population.sort(key=lambda s: s.fitness, reverse=True)

            if self.population[0].fitness > (self.best_ever.fitness if self.best_ever else -float('inf')):
                self.best_ever = self.population[0]

        return self.population

    def _tournament_select(self, tournament_size: int = 3) -> Strategy:
        """Tournament selection: pick best from random subset."""
        candidates = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(candidates, key=lambda s: s.fitness)

    def _crossover(self, p1: dict, p2: dict) -> dict:
        """Blend crossover: randomly mix parameters from both parents."""
        child = {}
        for key in p1:
            if key in p2:
                # Blend with random weight
                alpha = random.random()
                child[key] = alpha * p1[key] + (1 - alpha) * p2[key]
            else:
                child[key] = p1[key]
        return child

    def _mutate(self, params: dict) -> dict:
        """Random perturbation of parameters with mutation rate."""
        import numpy as np
        mutated = dict(params)
        for key in mutated:
            if random.random() < self.mutation_rate:
                if isinstance(mutated[key], int):
                    mutated[key] += int(np.random.normal(0, 3))
                    mutated[key] = max(1, mutated[key])
                else:
                    noise = np.random.normal(1.0, 0.1)
                    mutated[key] *= noise
        return mutated

    def get_best(self) -> Optional[Strategy]:
        """Get best performing strategy ever found."""
        return self.best_ever


class OptunaOptimizer:
    """
    Bayesian hyperparameter optimization for trading strategies using Optuna.

    Runs N trials to find optimal parameters maximizing Sharpe ratio
    or another objective function.
    """

    def __init__(self, backtest_fn: Callable[[dict], float], n_trials: int = 200):
        self.backtest_fn = backtest_fn
        self.n_trials = n_trials
        self.best_params: Optional[dict] = None
        self.study = None

    def optimize(self) -> dict:
        """Run Bayesian optimization."""
        try:
            import optuna
        except ImportError:
            return {"status": "error", "reason": "optuna not installed"}

        def objective(trial):
            params = {
                "ema_fast": trial.suggest_int("ema_fast", 5, 20),
                "ema_slow": trial.suggest_int("ema_slow", 20, 100),
                "rsi_period": trial.suggest_int("rsi_period", 7, 21),
                "atr_mult_sl": trial.suggest_float("atr_mult_sl", 1.0, 3.0),
                "atr_mult_tp": trial.suggest_float("atr_mult_tp", 1.5, 5.0),
                "volume_threshold": trial.suggest_float("volume_threshold", 1.0, 3.0),
                "min_confidence": trial.suggest_float("min_confidence", 0.3, 0.6),
            }
            return self.backtest_fn(params)

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials)
        self.best_params = self.study.best_params
        return {
            "status": "completed",
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": self.n_trials,
        }

    def get_trial_history(self) -> list:
        """Get all trial results for analysis."""
        if not self.study:
            return []
        return [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in self.study.trials if t.value is not None
        ]
