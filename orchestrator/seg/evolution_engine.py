"""
Evolution Engine — Genetic Algorithm + RL Selection for Strategy Evolution.

Core concept: Strategies evolve like biological organisms.
Best survive → Mutate → Compete → Next generation.

Architecture:
  Strategy Pool → Selection (Tournament/Fitness) → Mutation → Crossover → New Generation
                                    ↓
                          Backtest Evaluation (Fitness)
                                    ↓
                          Archive / Deploy Best
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .seg_generator import StrategyDSL, EntryRule, ExitRule, IndicatorConfig, RiskConfig
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, create_backtest_engine

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine."""

    # Population
    population_size: int = 50
    elite_size: int = 5              # Top N preserved unchanged
    tournament_size: int = 3         # Tournament selection size

    # Genetic operators
    mutation_rate: float = 0.3
    crossover_rate: float = 0.4
    mutation_strength: float = 0.2   # How much to mutate numeric values

    # Selection
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"
    fitness_sharing: bool = True          # Penalize similar strategies

    # Generations
    max_generations: int = 100
    convergence_threshold: float = 0.001  # Stop if improvement < threshold
    stagnation_limit: int = 10            # Generations without improvement

    # Backtest
    backtest_config: BacktestConfig | None = None
    min_trades_for_fitness: int = 20
    fitness_metric: str = "rl_fitness"    # "rl_fitness", "sharpe", "win_rate", "custom"

    # Diversity
    min_hamming_distance: int = 2         # Minimum rule differences
    max_similar_strategies: int = 3       # Max similar in population

    # Persistence
    archive_dir: str = "strategies/evolution_archive"
    checkpoint_dir: str = "strategies/checkpoints"
    save_checkpoint_every: int = 5        # Generations

    # Parallel
    parallel_backtest: bool = False       # Not implemented yet (would need multiprocessing)
    max_workers: int = 4

    # Callback
    on_generation_complete: Optional[callable] = None  # Callback(gen, best_strategy, stats)


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    std_fitness: float
    best_strategy_name: str
    unique_signatures: int
    elapsed_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class Selection:
    """Selection strategies for genetic algorithm."""

    @staticmethod
    def tournament(
        population: list[StrategyDSL],
        tournament_size: int = 3,
        fitness_attr: str = "fitness",
    ) -> StrategyDSL:
        """Tournament selection."""
        contestants = random.sample(population, min(tournament_size, len(population)))
        return max(contestants, key=lambda s: getattr(s, fitness_attr))

    @staticmethod
    def roulette(
        population: list[StrategyDSL],
        fitness_attr: str = "fitness",
    ) -> StrategyDSL:
        """Roulette wheel (fitness-proportionate) selection."""
        fitnesses = [max(0.001, getattr(s, fitness_attr)) for s in population]
        total = sum(fitnesses)
        probs = [f / total for f in fitnesses]
        return np.random.choice(population, p=probs)

    @staticmethod
    def rank(
        population: list[StrategyDSL],
        fitness_attr: str = "fitness",
    ) -> StrategyDSL:
        """Rank-based selection."""
        sorted_pop = sorted(population, key=lambda s: getattr(s, fitness_attr))
        ranks = list(range(1, len(sorted_pop) + 1))
        total = sum(ranks)
        probs = [r / total for r in ranks]
        return np.random.choice(sorted_pop, p=probs)

    @staticmethod
    def select_parents(
        population: list[StrategyDSL],
        method: str = "tournament",
        tournament_size: int = 3,
        fitness_attr: str = "fitness",
        n: int = 2,
    ) -> list[StrategyDSL]:
        """Select n parents."""
        parents = []
        for _ in range(n):
            if method == "tournament":
                parents.append(Selection.tournament(population, tournament_size, fitness_attr))
            elif method == "roulette":
                parents.append(Selection.roulette(population, fitness_attr))
            elif method == "rank":
                parents.append(Selection.rank(population, fitness_attr))
            else:
                parents.append(Selection.tournament(population, tournament_size, fitness_attr))
        return parents


class Crossover:
    """Crossover (recombination) operators."""

    @staticmethod
    def uniform(parent1: StrategyDSL, parent2: StrategyDSL, rate: float = 0.5) -> StrategyDSL:
        """Uniform crossover - each gene randomly from either parent."""
        import copy
        child = copy.deepcopy(parent1)
        child.name = f"{parent1.name}_x_{parent2.name}_{uuid.uuid4().hex[:6]}"
        child.parent_strategy = f"{parent1.name}+{parent2.name}"
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.fitness = 0.0
        child.trades_count = 0
        child.win_rate = 0.0
        child.sharpe = 0.0
        child.max_drawdown = 0.0
        child.total_pnl_pct = 0.0

        # Crossover indicators
        for attr in ["rsi_period", "macd_fast", "macd_slow", "macd_signal",
                     "bb_period", "bb_std", "ema_fast", "ema_slow", "ema_trend",
                     "atr_period", "volume_period"]:
            if random.random() < rate:
                setattr(child.indicators, attr, getattr(parent2.indicators, attr))

        # Crossover entry rules
        if random.random() < rate:
            child.entry_rules = copy.deepcopy(parent2.entry_rules)

        # Crossover exit rules
        if random.random() < rate:
            child.exit_rules = copy.deepcopy(parent2.exit_rules)

        # Crossover risk
        for attr in ["max_drawdown", "max_position_pct", "leverage",
                     "stop_loss_pct", "take_profit_pct", "max_hold_bars"]:
            if random.random() < rate:
                setattr(child.risk, attr, getattr(parent2.risk, attr))

        return child

    @staticmethod
    def single_point(parent1: StrategyDSL, parent2: StrategyDSL) -> StrategyDSL:
        """Single-point crossover on rule lists."""
        import copy
        child = copy.deepcopy(parent1)
        child.name = f"{parent1.name}_x_{parent2.name}_{uuid.uuid4().hex[:6]}"
        child.parent_strategy = f"{parent1.name}+{parent2.name}"
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.fitness = 0.0
        child.trades_count = 0
        child.win_rate = 0.0
        child.sharpe = 0.0
        child.max_drawdown = 0.0
        child.total_pnl_pct = 0.0

        # Crossover entry rules at random point
        if parent1.entry_rules and parent2.entry_rules:
            point = random.randint(0, min(len(parent1.entry_rules), len(parent2.entry_rules)))
            child.entry_rules = (
                copy.deepcopy(parent1.entry_rules[:point]) +
                copy.deepcopy(parent2.entry_rules[point:])
            )

        # Crossover exit rules
        if parent1.exit_rules and parent2.exit_rules:
            point = random.randint(0, min(len(parent1.exit_rules), len(parent2.exit_rules)))
            child.exit_rules = (
                copy.deepcopy(parent1.exit_rules[:point]) +
                copy.deepcopy(parent2.exit_rules[point:])
            )

        return child


class Mutation:
    """Mutation operators."""

    @staticmethod
    def mutate_numeric(value: float, strength: float = 0.2, min_val: float = None, max_val: float = None) -> float:
        """Mutate a numeric value with Gaussian noise."""
        noise = random.gauss(0, strength * abs(value) if value != 0 else strength)
        new_val = value + noise
        if min_val is not None:
            new_val = max(min_val, new_val)
        if max_val is not None:
            new_val = min(max_val, new_val)
        return new_val

    @staticmethod
    def mutate_int(value: int, strength: float = 0.2, min_val: int = None, max_val: int = None) -> int:
        """Mutate an integer value."""
        # Use normal distribution around current value
        delta = int(random.gauss(0, max(1, strength * abs(value))))
        new_val = value + delta
        if min_val is not None:
            new_val = max(min_val, new_val)
        if max_val is not None:
            new_val = min(max_val, new_val)
        return new_val

    @staticmethod
    def mutate_strategy(
        strategy: StrategyDSL,
        rate: float = 0.3,
        strength: float = 0.2,
    ) -> StrategyDSL:
        """Apply mutations to a strategy."""
        import copy
        mutated = copy.deepcopy(strategy)
        mutated.name = f"{strategy.name}_mut{mutated.generation + 1}_{uuid.uuid4().hex[:6]}"
        mutated.parent_strategy = strategy.name
        mutated.generation += 1
        mutated.fitness = 0.0
        mutated.trades_count = 0
        mutated.win_rate = 0.0
        mutated.sharpe = 0.0
        mutated.max_drawdown = 0.0
        mutated.total_pnl_pct = 0.0

        # Mutate indicators
        ind = mutated.indicators
        if random.random() < rate:
            ind.rsi_period = Mutation.mutate_int(ind.rsi_period, strength, 5, 30)
        if random.random() < rate:
            ind.macd_fast = Mutation.mutate_int(ind.macd_fast, strength, 5, 20)
        if random.random() < rate:
            ind.macd_slow = Mutation.mutate_int(ind.macd_slow, strength, 15, 40)
        if random.random() < rate:
            ind.macd_signal = Mutation.mutate_int(ind.macd_signal, strength, 5, 15)
        if random.random() < rate:
            ind.bb_period = Mutation.mutate_int(ind.bb_period, strength, 10, 50)
        if random.random() < rate:
            ind.bb_std = Mutation.mutate_numeric(ind.bb_std, strength, 1.0, 3.0)
        if random.random() < rate:
            ind.ema_fast = Mutation.mutate_int(ind.ema_fast, strength, 3, 20)
        if random.random() < rate:
            ind.ema_slow = Mutation.mutate_int(ind.ema_slow, strength, 10, 60)
        if random.random() < rate:
            ind.ema_trend = Mutation.mutate_int(ind.ema_trend, strength, 20, 100)
        if random.random() < rate:
            ind.atr_period = Mutation.mutate_int(ind.atr_period, strength, 5, 30)
        if random.random() < rate:
            ind.volume_period = Mutation.mutate_int(ind.volume_period, strength, 10, 50)

        # Mutate entry rules
        for rule in mutated.entry_rules:
            if random.random() < rate:
                rule.value = Mutation.mutate_numeric(rule.value, strength, 0.0, 100.0)
            if random.random() < rate * 0.5:
                rule.operator = random.choice([">", "<", ">=", "<=", "crosses_above", "crosses_below"])
            if random.random() < rate * 0.3:
                rule.indicator = random.choice(["rsi", "macd_hist", "bb_position", "ema_cross", "volume_spike", "atr"])
            if random.random() < rate:
                rule.weight = Mutation.mutate_numeric(rule.weight, strength, 0.1, 2.0)

        # Add/remove entry rules
        if random.random() < rate * 0.3:
            if mutated.entry_rules and random.random() < 0.5:
                mutated.entry_rules.pop(random.randrange(len(mutated.entry_rules)))
            else:
                mutated.entry_rules.append(EntryRule(
                    indicator=random.choice(["rsi", "macd_hist", "bb_position", "ema_cross", "volume_spike", "atr"]),
                    operator=random.choice([">", "<", ">=", "<="]),
                    value=random.uniform(0.1, 0.9),
                    weight=random.uniform(0.5, 1.5),
                ))

        # Mutate exit rules
        for rule in mutated.exit_rules:
            if random.random() < rate:
                rule.value = Mutation.mutate_numeric(rule.value, strength, 0.001, 0.5)
            if random.random() < rate * 0.3:
                rule.type = random.choice(["tp", "sl", "trailing_sl", "time", "signal_reverse"])
            if random.random() < rate:
                rule.weight = Mutation.mutate_numeric(rule.weight, strength, 0.1, 2.0)

        # Mutate risk
        risk = mutated.risk
        if random.random() < rate:
            risk.leverage = Mutation.mutate_int(risk.leverage, strength, 1, 10)
        if random.random() < rate:
            risk.max_drawdown = Mutation.mutate_numeric(risk.max_drawdown, strength, 0.01, 0.1)
        if random.random() < rate:
            risk.max_position_pct = Mutation.mutate_numeric(risk.max_position_pct, strength, 0.01, 0.2)
        if random.random() < rate:
            risk.stop_loss_pct = Mutation.mutate_numeric(risk.stop_loss_pct, strength, 0.002, 0.05)
        if random.random() < rate:
            risk.take_profit_pct = Mutation.mutate_numeric(risk.take_profit_pct, strength, 0.005, 0.1)
        if random.random() < rate:
            risk.max_hold_bars = Mutation.mutate_int(risk.max_hold_bars, strength, 10, 500)
        if random.random() < rate:
            risk.correlation_limit = Mutation.mutate_numeric(risk.correlation_limit, strength, 0.3, 0.9)

        return mutated


class DiversityManager:
    """Maintains population diversity."""

    @staticmethod
    def hamming_distance(s1: StrategyDSL, s2: StrategyDSL) -> int:
        """Compute structural Hamming distance between strategies."""
        distance = 0

        # Compare indicators
        for attr in ["rsi_period", "macd_fast", "macd_slow", "macd_signal",
                     "bb_period", "ema_fast", "ema_slow", "ema_trend"]:
            if getattr(s1.indicators, attr) != getattr(s2.indicators, attr):
                distance += 1

        # Compare entry rules (simplified)
        s1_entry_sig = tuple(sorted((r.indicator, r.operator, round(r.value, 2)) for r in s1.entry_rules))
        s2_entry_sig = tuple(sorted((r.indicator, r.operator, round(r.value, 2)) for r in s2.entry_rules))
        distance += len(set(s1_entry_sig) ^ set(s2_entry_sig))

        # Compare exit rules
        s1_exit_sig = tuple(sorted((r.type, round(r.value, 3)) for r in s1.exit_rules))
        s2_exit_sig = tuple(sorted((r.type, round(r.value, 3)) for r in s2.exit_rules))
        distance += len(set(s1_exit_sig) ^ set(s2_exit_sig))

        return distance

    @staticmethod
    def filter_diverse(
        population: list[StrategyDSL],
        min_distance: int = 2,
        max_similar: int = 3,
    ) -> list[StrategyDSL]:
        """Filter population to maintain diversity."""
        if not population:
            return []

        # Sort by fitness descending
        sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)

        diverse = [sorted_pop[0]]  # Always keep best

        for candidate in sorted_pop[1:]:
            # Count similar strategies in diverse set
            similar_count = sum(
                1 for d in diverse
                if DiversityManager.hamming_distance(candidate, d) < min_distance
            )

            if similar_count < max_similar:
                diverse.append(candidate)

        return diverse


class EvolutionEngine:
    """
    Main evolution engine — runs generational evolution of strategies.
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        backtest_engine: BacktestEngine | None = None,
        initial_population: list[StrategyDSL] | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.backtest_engine = backtest_engine or create_backtest_engine(self.config.backtest_config)

        # Population
        self.population: list[StrategyDSL] = initial_population or []
        self.generation = 0
        self.history: list[GenerationStats] = []

        # Best ever
        self.best_ever: StrategyDSL | None = None
        self.best_ever_fitness: float = -float('inf')

        # Stagnation tracking
        self._stagnation_count = 0
        self._last_best_fitness = -float('inf')

        # Directories
        Path(self.config.archive_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"[EvolutionEngine] Initialized with population={len(self.population)}")

    def add_strategies(self, strategies: list[StrategyDSL]):
        """Add strategies to population."""
        self.population.extend(strategies)
        # Trim to population size
        self._trim_population()

    def _trim_population(self):
        """Trim population to configured size, keeping best and diverse."""
        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)

        # Apply diversity filter
        self.population = DiversityManager.filter_diverse(
            self.population,
            self.config.min_hamming_distance,
            self.config.max_similar_strategies,
        )

        # Trim to population size
        if len(self.population) > self.config.population_size:
            self.population = self.population[:self.config.population_size]

    def initialize_population(
        self,
        generator,
        count: int | None = None,
        market_context: dict | None = None,
        regime: str | None = None,
    ):
        """Initialize population with generated strategies."""
        count = count or self.config.population_size

        logger.info(f"[EvolutionEngine] Initializing population with {count} strategies...")
        new_strategies = []

        # Run generation (async if needed)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        batch = loop.run_until_complete(generator.generate_batch(count, market_context, regime))
        new_strategies.extend(batch)

        self.add_strategies(new_strategies)
        logger.info(f"[EvolutionEngine] Population initialized: {len(self.population)} strategies")

    def evaluate_population(
        self,
        data: any = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        **backtest_kwargs,
    ) -> list[BacktestResult]:
        """Evaluate entire population via backtest."""
        logger.info(f"[EvolutionEngine] Evaluating generation {self.generation} ({len(self.population)} strategies)...")

        results = self.backtest_engine.run_batch(
            self.population,
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            **backtest_kwargs,
        )

        # Update best ever
        for result in results:
            strategy = next((s for s in self.population if s.name == result.strategy_name), None)
            if strategy and strategy.fitness > self.best_ever_fitness:
                self.best_ever_fitness = strategy.fitness
                import copy
                self.best_ever = copy.deepcopy(strategy)

        return results

    def evolve_generation(
        self,
        data: any = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        **backtest_kwargs,
    ) -> GenerationStats:
        """
        Run one generation of evolution:
        1. Evaluate current population (if not already evaluated)
        2. Select parents
        3. Apply crossover and mutation
        4. Evaluate offspring
        5. Select next generation
        """
        start_time = datetime.now()

        # 1. Evaluate if needed if any strategy has fitness 0
        if any(s.fitness == 0 for s in self.population):
            self.evaluate_population(data, symbol, timeframe, **backtest_kwargs)

        # 2. Selection - keep elite
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        elite = self.population[:self.config.elite_size]

        # 3. Generate offspring
        offspring = []

        while len(offspring) < self.config.population_size - self.config.elite_size:
            # Select parents
            parents = Selection.select_parents(
                self.population,
                self.config.selection_method,
                self.config.tournament_size,
                "fitness",
                n=2,
            )

            # Crossover
            if random.random() < self.config.crossover_rate:
                child = Crossover.uniform(parents[0], parents[1], 0.5)
            else:
                # Clone one parent
                import copy
                child = copy.deepcopy(parents[0])
                child.name = f"{parents[0].name}_clone_{uuid.uuid4().hex[:6]}"
                child.generation = self.generation + 1
                child.fitness = 0.0

            # Mutation
            child = Mutation.mutate_strategy(
                child,
                self.config.mutation_rate,
                self.config.mutation_strength,
            )

            offspring.append(child)

        # 4. New population = elite + offspring
        self.population = elite + offspring

        # 5. Evaluate new population
        self.evaluate_population(data, symbol, timeframe, **backtest_kwargs)

        # 6. Diversity filter and trim
        self._trim_population()

        # 7. Compute stats
        self.generation += 1
        stats = self._compute_generation_stats(start_time)
        self.history.append(stats)

        # 8. Check stagnation
        best_fitness = self.population[0].fitness if self.population else 0
        if best_fitness - self._last_best_fitness < self.config.convergence_threshold:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
        self._last_best_fitness = best_fitness

        # 9. Save checkpoint
        if self.generation % self.config.save_checkpoint_every == 0:
            self.save_checkpoint()

        # 10. Callback
        if self.config.on_generation_complete:
            self.config.on_generation_complete(self.generation, self.population[0], stats)

        logger.info(
            f"[EvolutionEngine] Generation {self.generation} complete | "
            f"Best: {best_fitness:.4f} | Avg: {stats.avg_fitness:.4f} | "
            f"Pop: {len(self.population)} | Stagnation: {self._stagnation_count}"
        )

        return stats

    def _compute_generation_stats(self, start_time: datetime) -> GenerationStats:
        """Compute generation statistics."""
        if not self.population:
            return GenerationStats(
                generation=self.generation,
                population_size=0,
                best_fitness=0, avg_fitness=0, worst_fitness=0, std_fitness=0,
                best_strategy_name="none",
                unique_signatures=0,
                elapsed_ms=0,
            )

        fitnesses = [s.fitness for s in self.population]
        signatures = set(s.get_signature() for s in self.population)

        return GenerationStats(
            generation=self.generation,
            population_size=len(self.population),
            best_fitness=float(np.max(fitnesses)),
            avg_fitness=float(np.mean(fitnesses)),
            worst_fitness=float(np.min(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            best_strategy_name=self.population[0].name,
            unique_signatures=len(signatures),
            elapsed_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )

    def run(
        self,
        max_generations: int | None = None,
        data: any = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        **backtest_kwargs,
    ) -> list[GenerationStats]:
        """
        Run evolution for multiple generations.

        Returns:
            List of GenerationStats for each generation
        """
        max_gen = max_generations or self.config.max_generations

        logger.info(f"[EvolutionEngine] Starting evolution for {max_gen} generations...")

        for gen in range(max_gen):
            if self._stagnation_count >= self.config.stagnation_limit:
                logger.info(f"[EvolutionEngine] Stagnation limit reached ({self._stagnation_count} gens), stopping")
                break

            self.evolve_generation(data, symbol, timeframe, **backtest_kwargs)

        # Save final results
        self.save_checkpoint()
        self.archive_best()

        logger.info(f"[EvolutionEngine] Evolution complete. Best fitness: {self.best_ever_fitness:.4f}")
        return self.history

    def save_checkpoint(self):
        """Save current population and state."""
        checkpoint = {
            "generation": self.generation,
            "population": [s.to_dict() for s in self.population],
            "best_ever": self.best_ever.to_dict() if self.best_ever else None,
            "best_ever_fitness": self.best_ever_fitness,
            "history": [
                {
                    "generation": h.generation,
                    "population_size": h.population_size,
                    "best_fitness": h.best_fitness,
                    "avg_fitness": h.avg_fitness,
                    "worst_fitness": h.worst_fitness,
                    "std_fitness": h.std_fitness,
                    "best_strategy_name": h.best_strategy_name,
                    "unique_signatures": h.unique_signatures,
                    "elapsed_ms": h.elapsed_ms,
                    "timestamp": h.timestamp.isoformat(),
                }
                for h in self.history
            ],
            "stagnation_count": self._stagnation_count,
        }

        path = Path(self.config.checkpoint_dir) / f"checkpoint_gen{self.generation}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        # Also save as latest
        latest_path = Path(self.config.checkpoint_dir) / "checkpoint_latest.json"
        with open(latest_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.debug(f"[EvolutionEngine] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | None = None):
        """Load checkpoint."""
        if path is None:
            path = Path(self.config.checkpoint_dir) / "checkpoint_latest.json"

        if not Path(path).exists():
            logger.warning(f"[EvolutionEngine] Checkpoint not found: {path}")
            return False

        with open(path) as f:
            checkpoint = json.load(f)

        self.generation = checkpoint["generation"]
        self.population = [StrategyDSL.from_dict(s) for s in checkpoint["population"]]
        if checkpoint["best_ever"]:
            self.best_ever = StrategyDSL.from_dict(checkpoint["best_ever"])
        self.best_ever_fitness = checkpoint["best_ever_fitness"]
        self._stagnation_count = checkpoint.get("stagnation_count", 0)

        # Rebuild history
        self.history = [
            GenerationStats(**{**h, "timestamp": datetime.fromisoformat(h["timestamp"])})
            for h in checkpoint["history"]
        ]

        logger.info(f"[EvolutionEngine] Loaded checkpoint from gen {self.generation}, pop={len(self.population)}")
        return True

    def archive_best(self):
        """Archive the best strategy."""
        if self.best_ever:
            path = Path(self.config.archive_dir) / f"best_{self.best_ever.name}_gen{self.generation}.json"
            with open(path, "w") as f:
                json.dump(self.best_ever.to_dict(), f, indent=2)
            logger.info(f"[EvolutionEngine] Archived best strategy to {path}")

    def get_best_strategies(self, n: int = 10) -> list[StrategyDSL]:
        """Get top N strategies from current population."""
        return sorted(self.population, key=lambda s: s.fitness, reverse=True)[:n]

    def get_status(self) -> dict:
        """Get current evolution status."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.population[0].fitness if self.population else 0,
            "best_ever_fitness": self.best_ever_fitness,
            "stagnation_count": self._stagnation_count,
            "history_length": len(self.history),
        }


# Factory functions
def create_evolution_config(**kwargs) -> EvolutionConfig:
    """Factory for EvolutionConfig."""
    return EvolutionConfig(**kwargs)


def create_evolution_engine(
    config: EvolutionConfig | None = None,
    backtest_engine: BacktestEngine | None = None,
    initial_population: list[StrategyDSL] | None = None,
) -> EvolutionEngine:
    """Factory for EvolutionEngine."""
    return EvolutionEngine(config, backtest_engine, initial_population)


__all__ = [
    "EvolutionConfig",
    "GenerationStats",
    "EvolutionEngine",
    "Selection",
    "Crossover",
    "Mutation",
    "DiversityManager",
    "create_evolution_config",
    "create_evolution_engine",
]