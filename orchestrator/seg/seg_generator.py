"""
Self-Evolving Strategy Generator (SEG v1) — DSL + Backtest + Evolution.

Core concept: Continuous strategy invention, testing, and evolution.
LLM generates strategies → Backtest evaluates → Evolution selects/mutates → Deploy best.

Architecture:
  Strategy Generator (LLM) → Strategy Pool (DSL JSON)
                                    ↓
                          Backtest Engine (Historical Replay)
                                    ↓
                          Performance Scorer (RL Reward)
                                    ↓
                          Evolution Engine (Mutation + Selection)
                                    ↓
                          Updated Strategy Pool → Deploy
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# Import existing reward function
from orchestrator.rl_memory.reward_function import RewardConfig, compute_reward
from orchestrator.rl_memory.memory_store import TradeOutcome, create_memory_manager, RLMemoryManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY DSL (Domain Specific Language)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorConfig:
    """Technical indicator configuration."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    atr_period: int = 14
    volume_period: int = 20


@dataclass
class EntryRule:
    """Single entry condition."""
    indicator: str  # e.g., "rsi", "macd_hist", "bb_position", "ema_cross", "volume_spike"
    operator: str   # ">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"
    value: float    # threshold value
    weight: float = 1.0  # rule importance


@dataclass
class ExitRule:
    """Exit condition."""
    type: str  # "tp", "sl", "trailing_sl", "time", "signal_reverse"
    value: float  # percentage for tp/sl, bars for time
    weight: float = 1.0


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_drawdown: float = 0.03      # 3% max drawdown per trade
    max_position_pct: float = 0.05  # 5% of portfolio per position
    leverage: int = 3               # max leverage
    stop_loss_pct: float = 0.01     # 1% stop loss
    take_profit_pct: float = 0.02   # 2% take profit
    max_hold_bars: int = 100        # max bars to hold
    correlation_limit: float = 0.7  # max correlation with open positions


@dataclass
class StrategyDSL:
    """
    Strategy Domain Specific Language — structured, testable, evolvable.

    JSON-serializable format for LLM generation and genetic operations.
    """
    name: str
    version: str = "1.0"
    timeframe: str = "1m"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])

    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    entry_rules: list[EntryRule] = field(default_factory=list)
    exit_rules: list[ExitRule] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_strategy: str | None = None  # For tracking lineage
    generation: int = 0
    fitness: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_pnl_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary (JSON serializable)."""
        d = asdict(self)
        # Convert dataclasses to dicts
        d["indicators"] = asdict(self.indicators)
        d["entry_rules"] = [asdict(r) for r in self.entry_rules]
        d["exit_rules"] = [asdict(r) for r in self.exit_rules]
        d["risk"] = asdict(self.risk)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyDSL":
        """Create from dictionary."""
        d = d.copy()
        d["indicators"] = IndicatorConfig(**d.get("indicators", {}))
        d["entry_rules"] = [EntryRule(**r) for r in d.get("entry_rules", [])]
        d["exit_rules"] = [ExitRule(**r) for r in d.get("exit_rules", [])]
        d["risk"] = RiskConfig(**d.get("risk", {}))
        return cls(**d)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyDSL":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_signature(self) -> str:
        """Unique signature for deduplication."""
        key_parts = [
            self.timeframe,
            str(sorted([r.indicator for r in self.entry_rules])),
            str(sorted([r.type for r in self.exit_rules])),
            str(self.risk.leverage),
        ]
        return "_".join(key_parts)

    def mutate(self, mutation_rate: float = 0.3) -> "StrategyDSL":
        """Create mutated copy of this strategy."""
        import copy
        new = copy.deepcopy(self)
        new.name = f"{self.name}_mut{new.generation + 1}"
        new.version = f"{int(float(self.version)) + 1}.0"
        new.parent_strategy = self.name
        new.generation += 1
        new.fitness = 0.0
        new.trades_count = 0
        new.win_rate = 0.0
        new.sharpe = 0.0
        new.max_drawdown = 0.0
        new.total_pnl_pct = 0.0

        # Mutate indicators
        if random.random() < mutation_rate:
            new.indicators.rsi_period = max(5, min(30, new.indicators.rsi_period + random.randint(-3, 3)))
        if random.random() < mutation_rate:
            new.indicators.bb_period = max(10, min(50, new.indicators.bb_period + random.randint(-5, 5)))
        if random.random() < mutation_rate:
            new.indicators.ema_fast = max(3, min(20, new.indicators.ema_fast + random.randint(-3, 3)))
        if random.random() < mutation_rate:
            new.indicators.ema_slow = max(10, min(60, new.indicators.ema_slow + random.randint(-5, 5)))

        # Mutate entry rules
        if random.random() < mutation_rate and new.entry_rules:
            rule = random.choice(new.entry_rules)
            if random.random() < 0.5:
                rule.value += random.uniform(-0.1, 0.1) * abs(rule.value + 1)
            else:
                rule.operator = random.choice([">", "<", ">=", "<="])

        # Add/remove entry rule
        if random.random() < mutation_rate * 0.5:
            if new.entry_rules and random.random() < 0.5:
                new.entry_rules.pop(random.randrange(len(new.entry_rules)))
            else:
                new.entry_rules.append(EntryRule(
                    indicator=random.choice(["rsi", "macd_hist", "bb_position", "ema_cross", "volume_spike"]),
                    operator=random.choice([">", "<", ">=", "<="]),
                    value=random.uniform(0.2, 0.8),
                    weight=random.uniform(0.5, 1.5),
                ))

        # Mutate exit rules
        if random.random() < mutation_rate and new.exit_rules:
            rule = random.choice(new.exit_rules)
            if rule.type in ("tp", "sl"):
                rule.value *= random.uniform(0.8, 1.3)

        # Mutate risk
        if random.random() < mutation_rate:
            new.risk.leverage = max(1, min(10, new.risk.leverage + random.randint(-1, 1)))
        if random.random() < mutation_rate:
            new.risk.stop_loss_pct *= random.uniform(0.8, 1.2)
        if random.random() < mutation_rate:
            new.risk.take_profit_pct *= random.uniform(0.8, 1.2)

        return new


# ═══════════════════════════════════════════════════════════════════
# STRATEGY GENERATOR (LLM / MoE)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GeneratorConfig:
    """Configuration for strategy generator."""
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # Generation
    strategies_per_batch: int = 10
    max_strategies_in_pool: int = 100
    regime_aware: bool = True

    # Persistence
    pool_dir: str = "strategies/pool"
    archive_dir: str = "strategies/archive"


class StrategyGenerator:
    """
    LLM-based strategy generator.

    Creates new StrategyDSL instances using LLM, guided by:
    - Market context (regime, volatility, recent performance)
    - Memory of successful/failed patterns
    - Diversity constraints
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        memory_manager: RLMemoryManager | None = None,
    ):
        self.config = config or GeneratorConfig()
        self.memory_manager = memory_manager or create_memory_manager()

        # Strategy pool
        self.pool: list[StrategyDSL] = []
        self._load_pool()

        # Ensure directories
        Path(self.config.pool_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.archive_dir).mkdir(parents=True, exist_ok=True)

    def _load_pool(self):
        """Load existing strategy pool from disk."""
        pool_dir = Path(self.config.pool_dir)
        for file in pool_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                strategy = StrategyDSL.from_dict(data)
                self.pool.append(strategy)
            except Exception as e:
                logger.warning(f"Failed to load strategy {file}: {e}")

        logger.info(f"[StrategyGenerator] Loaded {len(self.pool)} strategies from pool")

    def _save_pool(self):
        """Save strategy pool to disk."""
        pool_dir = Path(self.config.pool_dir)
        for strategy in self.pool:
            path = pool_dir / f"{strategy.name}_v{strategy.version}.json"
            with open(path, "w") as f:
                json.dump(strategy.to_dict(), f, indent=2)

    async def generate_batch(
        self,
        count: int | None = None,
        market_context: dict | None = None,
        regime: str | None = None,
    ) -> list[StrategyDSL]:
        """
        Generate a batch of new strategies using LLM.

        Args:
            count: Number of strategies to generate
            market_context: Current market data (price, indicators, etc.)
            regime: Current market regime (trending/ranging/volatile)

        Returns:
            List of new StrategyDSL instances
        """
        count = count or self.config.strategies_per_batch
        new_strategies = []

        for i in range(count):
            try:
                strategy = await self._generate_single(market_context, regime)
                if strategy:
                    new_strategies.append(strategy)
            except Exception as e:
                logger.warning(f"[StrategyGenerator] Generation {i} failed: {e}")

        # Add to pool
        self.pool.extend(new_strategies)
        self._trim_pool()
        self._save_pool()

        logger.info(f"[StrategyGenerator] Generated {len(new_strategies)} new strategies")
        return new_strategies

    async def _generate_single(
        self,
        market_context: dict | None = None,
        regime: str | None = None,
    ) -> StrategyDSL | None:
        """Generate a single strategy using LLM."""
        prompt = self._build_generation_prompt(market_context, regime)

        try:
            import openai
            client = openai.OpenAI(
                base_url=self.config.llm_base_url,
                api_key="ollama",
            )

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

            content = response.choices[0].message.content.strip()
            strategy_data = self._parse_llm_response(content)

            if strategy_data:
                strategy = StrategyDSL.from_dict(strategy_data)
                strategy.name = f"llm_strat_{uuid.uuid4().hex[:8]}"
                return strategy

        except Exception as e:
            logger.error(f"[StrategyGenerator] LLM generation failed: {e}")

        # Fallback: generate random strategy
        return self._generate_random_strategy(regime)

    def _get_system_prompt(self) -> str:
        return """You are a quantitative trading strategy designer.
Generate a complete trading strategy in the exact JSON format specified.
Output ONLY valid JSON. No markdown, no explanation.

The strategy must include:
- name: unique identifier
- timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
- indicators: {rsi_period, macd_fast, macd_slow, macd_signal, bb_period, bb_std, ema_fast, ema_slow, ema_trend, atr_period, volume_period}
- entry_rules: array of {indicator, operator, value, weight}
  indicator ∈ ["rsi", "macd_hist", "bb_position", "ema_cross", "volume_spike", "atr"]
  operator ∈ [">", "<", ">=", "<=", "crosses_above", "crosses_below"]
- exit_rules: array of {type, value, weight}
  type ∈ ["tp", "sl", "trailing_sl", "time", "signal_reverse"]
- risk: {max_drawdown, max_position_pct, leverage, stop_loss_pct, take_profit_pct, max_hold_bars, correlation_limit}

Make strategies diverse and regime-appropriate."""

    def _build_generation_prompt(
        self,
        market_context: dict | None = None,
        regime: str | None = None,
    ) -> str:
        context_str = ""
        if market_context:
            context_str += f"Current market: {json.dumps(market_context, default=str)}\n"
        if regime:
            context_str += f"Market regime: {regime}\n"

        # Add memory of successful patterns
        if self.memory_manager and self.memory_manager.is_available("postgres"):
            try:
                stats = self.memory_manager.get_performance_stats(30)
                context_str += f"Recent 30d performance: {json.dumps(stats)}\n"
            except Exception:
                pass

        # Add top strategies from pool
        if self.pool:
            top = sorted(self.pool, key=lambda s: s.fitness, reverse=True)[:3]
            context_str += "Top existing strategies:\n"
            for s in top:
                context_str += f"  {s.name}: fitness={s.fitness:.3f}, win_rate={s.win_rate:.2f}, sharpe={s.sharpe:.2f}\n"

        return f"""{context_str}

Generate a NEW, DIFFERENT trading strategy in JSON format.
Focus on: regime={regime or 'unknown'} conditions.
Ensure it's testable and has clear entry/exit rules.

Output JSON only:"""

    def _parse_llm_response(self, content: str) -> dict | None:
        """Parse LLM response to extract strategy JSON."""
        try:
            # Try direct JSON parse
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON block
            import re
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _generate_random_strategy(self, regime: str | None = None) -> StrategyDSL:
        """Generate a random strategy as fallback."""
        indicators = IndicatorConfig(
            rsi_period=random.randint(10, 20),
            macd_fast=random.randint(8, 15),
            macd_slow=random.randint(20, 30),
            macd_signal=random.randint(7, 12),
            bb_period=random.randint(15, 25),
            bb_std=random.uniform(1.5, 2.5),
            ema_fast=random.randint(5, 15),
            ema_slow=random.randint(18, 30),
            ema_trend=random.randint(40, 80),
            atr_period=random.randint(10, 20),
            volume_period=random.randint(15, 25),
        )

        # Regime-aware entry rules
        entry_rules = []
        if regime == "trending":
            entry_rules.append(EntryRule("ema_cross", "crosses_above", 0, 1.2))
            entry_rules.append(EntryRule("macd_hist", ">", 0, 1.0))
        elif regime == "ranging":
            entry_rules.append(EntryRule("rsi", "<", 30, 1.2))
            entry_rules.append(EntryRule("bb_position", "<", 0.2, 1.0))
        elif regime == "volatile":
            entry_rules.append(EntryRule("atr", ">", 0.02, 1.0))
            entry_rules.append(EntryRule("volume_spike", ">", 1.5, 1.0))
        else:
            # Random
            for _ in range(random.randint(1, 3)):
                entry_rules.append(EntryRule(
                    indicator=random.choice(["rsi", "macd_hist", "bb_position", "ema_cross", "volume_spike"]),
                    operator=random.choice([">", "<", ">=", "<="]),
                    value=random.uniform(0.1, 0.9),
                    weight=random.uniform(0.5, 1.5),
                ))

        exit_rules = [
            ExitRule("tp", random.uniform(0.01, 0.03), 1.0),
            ExitRule("sl", random.uniform(0.005, 0.02), 1.0),
        ]

        risk = RiskConfig(
            max_drawdown=random.uniform(0.02, 0.05),
            max_position_pct=random.uniform(0.02, 0.08),
            leverage=random.randint(1, 5),
            stop_loss_pct=random.uniform(0.005, 0.02),
            take_profit_pct=random.uniform(0.01, 0.04),
            max_hold_bars=random.randint(20, 200),
        )

        strategy = StrategyDSL(
            name=f"random_{uuid.uuid4().hex[:8]}",
            timeframe=random.choice(["1m", "5m", "15m", "1h"]),
            indicators=indicators,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            risk=risk,
        )

        return strategy

    def _trim_pool(self):
        """Keep pool size within limits, removing worst strategies."""
        if len(self.pool) > self.config.max_strategies_in_pool:
            # Sort by fitness, keep best
            self.pool.sort(key=lambda s: s.fitness, reverse=True)
            # Archive removed strategies
            removed = self.pool[self.config.max_strategies_in_pool:]
            for s in removed:
                archive_path = Path(self.config.archive_dir) / f"{s.name}_v{s.version}.json"
                with open(archive_path, "w") as f:
                    json.dump(s.to_dict(), f, indent=2)
            self.pool = self.pool[: self.config.max_strategies_in_pool]

    def get_pool_stats(self) -> dict:
        """Get pool statistics."""
        if not self.pool:
            return {"count": 0}

        fitnesses = [s.fitness for s in self.pool]
        return {
            "count": len(self.pool),
            "avg_fitness": float(np.mean(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "generations": len(set(s.generation for s in self.pool)),
            "unique_signatures": len(set(s.get_signature() for s in self.pool)),
        }

    def get_best_strategies(self, n: int = 10) -> list[StrategyDSL]:
        """Get top N strategies by fitness."""
        return sorted(self.pool, key=lambda s: s.fitness, reverse=True)[:n]


# ═══════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def create_strategy_dsl(
    name: str,
    timeframe: str = "1m",
    indicators: IndicatorConfig | None = None,
    entry_rules: list[EntryRule] | None = None,
    exit_rules: list[ExitRule] | None = None,
    risk: RiskConfig | None = None,
) -> StrategyDSL:
    """Factory for creating StrategyDSL."""
    return StrategyDSL(
        name=name,
        timeframe=timeframe,
        indicators=indicators or IndicatorConfig(),
        entry_rules=entry_rules or [],
        exit_rules=exit_rules or [],
        risk=risk or RiskConfig(),
    )


def create_generator(
    config: GeneratorConfig | None = None,
    memory_manager: RLMemoryManager | None = None,
) -> StrategyGenerator:
    """Factory for StrategyGenerator."""
    return StrategyGenerator(config, memory_manager)


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
    # Factories
    "create_strategy_dsl",
    "create_generator",
]