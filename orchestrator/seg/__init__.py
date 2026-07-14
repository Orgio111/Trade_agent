"""
Self-Evolving Strategy Generator (SEG v1) Package.

Components:
- seg_generator: Strategy DSL + LLM Generator
- backtest_engine: Historical replay evaluation
- evolution_engine: Genetic algorithm + RL selection
"""

from .seg_generator import (
    IndicatorConfig,
    EntryRule,
    ExitRule,
    RiskConfig,
    StrategyDSL,
    GeneratorConfig,
    StrategyGenerator,
    create_strategy_dsl,
    create_generator,
)
from .backtest_engine import (
    BacktestConfig,
    BacktestResult,
    BacktestEngine,
    FeatureEngine,
    SignalEvaluator,
    create_backtest_engine,
    create_backtest_config,
)
from .evolution_engine import (
    EvolutionConfig,
    GenerationStats,
    EvolutionEngine,
    Selection,
    Crossover,
    Mutation,
    DiversityManager,
    create_evolution_config,
    create_evolution_engine,
)

__all__ = [
    # DSL
    "IndicatorConfig",
    "EntryRule",
    "ExitRule",
    "RiskConfig",
    "StrategyDSL",
    # Generator
    "GeneratorConfig",
    "StrategyGenerator",
    # Backtest
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "FeatureEngine",
    "SignalEvaluator",
    # Evolution
    "EvolutionConfig",
    "GenerationStats",
    "EvolutionEngine",
    "Selection",
    "Crossover",
    "Mutation",
    "DiversityManager",
    # Factories
    "create_strategy_dsl",
    "create_generator",
    "create_backtest_engine",
    "create_backtest_config",
    "create_evolution_config",
    "create_evolution_engine",
]