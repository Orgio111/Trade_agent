"""
CLI Commands for Real-Time RL Streaming + Meta-Learning + SEG v1.

Commands:
- rl_streaming: Start streaming reward engine daemon
- meta_learner: Run meta-learning analysis cycle
- seg_evolution: Run strategy evolution cycle
- seg_generate: Generate new strategies
- seg_backtest: Backtest strategies
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Real-Time RL Streaming + Meta-Learning + SEG v1 CLI."""
    pass


# ═══════════════════════════════════════════════════════════════════
# RL STREAMING COMMANDS
# ═══════════════════════════════════════════════════════════════════

@cli.group()
def rl_streaming():
    """Streaming RL commands."""
    pass


@rl_streaming.command("daemon")
@click.option("--redis-url", default=None, help="Redis URL")
@click.option("--postgres-dsn", default=None, help="Postgres DSN")
@click.option("--qdrant-url", default=None, help="Qdrant URL")
@click.option("--update-interval", default=100, type=int, help="Policy update interval (ms)")
@click.option("--buffer-size", default=10000, type=int, help="Reward buffer size")
def rl_streaming_daemon(redis_url, postgres_dsn, qdrant_url, update_interval, buffer_size):
    """Run streaming reward engine as daemon."""
    from orchestrator.rl_streaming import (
        StreamingMemoryManager,
        StreamingMemoryConfig,
        StreamingRewardConfig,
        OnlinePolicyConfig,
    )
    from orchestrator.rl_memory.memory_store import create_memory_manager

    config = StreamingMemoryConfig(
        redis_url=redis_url or os.getenv("REDIS_URL"),
        postgres_dsn=postgres_dsn or os.getenv("POSTGRES_DSN"),
        qdrant_url=qdrant_url or os.getenv("QDRANT_URL"),
        streaming_reward_config=StreamingRewardConfig(
            update_interval_ms=update_interval,
            buffer_max_size=buffer_size,
        ),
        online_policy_config=OnlinePolicyConfig(),
    )

    manager = StreamingMemoryManager(config)
    logger.info("[RL Streaming] Daemon started. Press Ctrl+C to stop.")

    try:
        import time
        while True:
            time.sleep(60)
            stats = manager.get_status()
            logger.info(f"[RL Streaming] Status: events={stats['total_events']}, updates={stats['total_updates']}")
    except KeyboardInterrupt:
        logger.info("[RL Streaming] Shutting down...")
        manager.shutdown()


@rl_streaming.command("test")
@click.option("--events", default=100, type=int, help="Number of test events")
def rl_streaming_test(events):
    """Test streaming reward engine with synthetic events."""
    from orchestrator.rl_streaming import (
        StreamingRewardEngine,
        StreamingRewardConfig,
        RewardEvent,
        RewardEventType,
    )
    import numpy as np

    engine = StreamingRewardEngine(StreamingRewardConfig())

    for i in range(events):
        event = RewardEvent(
            event_type=RewardEventType.TRADE_CLOSE if i % 2 == 0 else RewardEventType.TRADE_OPEN,
            trade_id=f"test_{i}",
            symbol="BTCUSDT",
            action=np.random.choice(["BUY", "SELL", "SELL", "HOLD"]).strip(),
            entry_price=50000 + np.random.randn() * 100,
            exit_price=50000 + np.random.randn() * 100,
            quantity=0.01,
            filled_qty=0.01,
            avg_price=50000 + np.random.randn() * 100,
            realized_pnl_pct=np.random.randn() * 0.02,
            unrealized_pnl_pct=np.random.randn() * 0.01,
            confidence=np.random.uniform(0.5, 0.95),
            regime=np.random.choice(["trending", "ranging", "volatile"]),
            indicators={"rsi": np.random.uniform(20, 80), "macd_hist": np.random.randn()},
        )
        reward, breakdown = engine.process_event(event)
        if i % 20 == 0:
            logger.info(f"Event {i}: reward={reward:.4f}, action={event.action}")

    logger.info(f"Test complete. Buffer stats: {engine.get_buffer_stats()}")


# ═══════════════════════════════════════════════════════════════════
# META LEARNER COMMANDS
# ═══════════════════════════════════════════════════════════════════

@cli.group()
def meta_learner():
    """Meta-learning commands."""
    pass


@meta_learner.command("run")
@click.option("--lookback-days", default=7, type=int, help="Days of trades to analyze")
@click.option("--min-trades", default=20, type=int, help="Minimum trades for analysis")
@click.option("--llm-model", default="qwen2.5:3b", help="LLM model for rule generation")
@click.option("--llm-url", default="http://localhost:11434/v1", help="LLM base URL")
@click.option("--auto-deploy/--no-auto-deploy", default=False, help="Auto-deploy improved rules")
@click.option("--force", is_flag=True, help="Force run even if interval not reached")
def meta_learner_run(lookback_days, min_trades, llm_model, llm_url, auto_deploy, force):
    """Run meta-learning analysis cycle."""
    from orchestrator.rl_streaming import MetaLearner, MetaLearningConfig
    from orchestrator.rl_memory.memory_store import create_memory_manager

    config = MetaLearningConfig(
        lookback_days=lookback_days,
        min_trades_for_analysis=min_trades,
        llm_model=llm_model,
        llm_base_url=llm_url,
        auto_deploy=auto_deploy,
    )

    memory = create_memory_manager()
    learner = MetaLearner(config, memory)

    result = asyncio.run(learner.run_analysis(force=force))

    if result:
        click.echo(json.dumps({
            "timestamp": result.timestamp.isoformat(),
            "trades_analyzed": result.trades_analyzed,
            "confidence": result.confidence,
            "deployed": result.deployed,
            "winning_patterns": result.winning_patterns,
            "losing_patterns": result.losing_patterns,
            "regime_insights": result.regime_insights,
            "suggested_rules": result.suggested_rules,
            "backtest_validation": result.backtest_validation,
        }, indent=2))
    else:
        click.echo("No analysis performed (insufficient trades or interval not reached)")


@meta_learner.command("status")
def meta_learner_status():
    """Show meta-learner status."""
    from orchestrator.rl_streaming import MetaLearner, MetaLearningConfig
    from orchestrator.rl_memory.memory_store import create_memory_manager

    config = MetaLearningConfig()
    memory = create_memory_manager()
    learner = MetaLearner(config, memory)

    status = learner.get_status()
    click.echo(json.dumps(status, indent=2, default=str))


@meta_learner.command("history")
@click.option("--limit", default=10, type=int, help="Number of recent analyses")
def meta_learner_history(limit):
    """Show meta-learning analysis history."""
    from orchestrator.rl_streaming import MetaLearner, MetaLearningConfig
    from orchestrator.rl_memory.memory_store import create_memory_manager

    config = MetaLearningConfig()
    memory = create_memory_manager()
    learner = MetaLearner(config, memory)

    history = learner.get_analysis_history(limit)
    click.echo(json.dumps(history, indent=2))


# ═══════════════════════════════════════════════════════════════════
# SEG COMMANDS
# ═══════════════════════════════════════════════════════════════════

@cli.group()
def seg():
    """Self-Evolving Strategy Generator (SEG v1) commands."""
    pass


@seg.command("generate")
@click.option("--count", default=10, type=int, help="Number of strategies to generate")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--regime", default="unknown", help="Market regime")
@click.option("--llm-model", default="qwen2.5:3b", help="LLM model")
@click.option("--llm-url", default="http://localhost:11434/v1", help="LLM base URL")
def seg_generate(count, symbol, regime, llm_model, llm_url):
    """Generate new strategies using LLM."""
    from orchestrator.seg import StrategyGenerator, GeneratorConfig
    from orchestrator.rl_memory.memory_store import create_memory_manager

    config = GeneratorConfig(
        llm_model=llm_model,
        llm_base_url=llm_url,
        strategies_per_batch=count,
    )

    memory = create_memory_manager()
    generator = StrategyGenerator(config, memory)

    market_context = {"symbol": symbol, "regime": regime}

    async def run():
        strategies = await generator.generate_batch(count, market_context, regime)
        for s in strategies:
            click.echo(f"Generated: {s.name} (timeframe={s.timeframe}, rules={len(s.entry_rules)})")
        click.echo(f"Pool stats: {generator.get_pool_stats()}")

    asyncio.run(run())


@seg.command("backtest")
@click.option("--strategy-file", required=True, help="Path to strategy JSON file")
@click.option("--data-file", default=None, help="Path to OHLCV CSV data")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="1m", help="Timeframe")
@click.option("--start", default=None, help="Start date (ISO format)")
@click.option("--end", default=None, help="End date (ISO format)")
@click.option("--capital", default=10000.0, type=float, help="Initial capital")
def seg_backtest(strategy_file, data_file, symbol, timeframe, start, end, capital):
    """Backtest a strategy."""
    from orchestrator.seg import (
        StrategyDSL,
        BacktestEngine,
        BacktestConfig,
        create_backtest_engine,
    )

    with open(strategy_file) as f:
        strategy_data = json.load(f)
    strategy = StrategyDSL.from_dict(strategy_data)

    config = BacktestConfig(
        initial_capital=capital,
        default_symbol=symbol,
        default_timeframe=timeframe,
    )
    engine = create_backtest_engine(config)

    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    result = engine.run(
        strategy,
        data_path=data_file,
        symbol=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
    )

    click.echo(json.dumps({
        "strategy": result.strategy_name,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_pnl_pct": result.total_pnl_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "profit_factor": result.profit_factor,
        "expectancy": result.expectancy,
        "rl_fitness": result.rl_fitness,
        "avg_reward": result.avg_reward,
    }, indent=2))


@seg.command("evolve")
@click.option("--generations", default=20, type=int, help="Max generations")
@click.option("--population", default=50, type=int, help="Population size")
@click.option("--data-file", default=None, help="Path to OHLCV CSV data")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="1m", help="Timeframe")
@click.option("--llm-model", default="qwen2.5:3b", help="LLM model for generation")
@click.option("--llm-url", default="http://localhost:11434/v1", help="LLM base URL")
@click.option("--checkpoint", default=None, help="Resume from checkpoint")
def seg_evolve(generations, population, data_file, symbol, timeframe, llm_model, llm_url, checkpoint):
    """Run strategy evolution."""
    from orchestrator.seg import (
        EvolutionEngine,
        EvolutionConfig,
        StrategyGenerator,
        GeneratorConfig,
        BacktestEngine,
        BacktestConfig,
        create_evolution_engine,
        create_backtest_engine,
        create_generator,
    )
    from orchestrator.rl_memory.memory_store import create_memory_manager

    generator_config = GeneratorConfig(
        llm_model=llm_model,
        llm_base_url=llm_url,
        strategies_per_batch=population,
    )
    backtest_config = BacktestConfig(default_symbol=symbol, default_timeframe=timeframe)
    evolution_config = EvolutionConfig(
        population_size=population,
        max_generations=generations,
        backtest_config=backtest_config,
    )

    memory = create_memory_manager()
    generator = create_generator(generator_config, memory)
    backtest = create_backtest_engine(backtest_config)
    engine = create_evolution_engine(evolution_config, backtest, [])

    if checkpoint:
        engine.load_checkpoint(checkpoint)
    else:
        import asyncio
        asyncio.run(generator.generate_batch(population, {"symbol": symbol}, "unknown"))
        engine.add_strategies(generator.pool)

    data = backtest.load_data(data_path=data_file)

    history = engine.run(
        max_generations=generations,
        data=data,
        symbol=symbol,
        timeframe=timeframe,
    )

    best = engine.get_best_strategies(5)
    click.echo("\n=== Top 5 Strategies ===")
    for i, s in enumerate(best):
        click.echo(f"{i+1}. {s.name}: fitness={s.fitness:.4f}, gen={s.generation}, "
                   f"win_rate={s.win_rate:.2f}, sharpe={s.sharpe:.2f}, dd={s.max_drawdown:.2f}")

    click.echo(f"\nEvolution complete. Best ever fitness: {engine.best_ever_fitness:.4f}")


@seg.command("pool")
@click.option("--limit", default=20, type=int, help="Number of strategies to show")
def seg_pool(limit):
    """Show strategy pool."""
    from orchestrator.seg import StrategyGenerator, GeneratorConfig
    from orchestrator.rl_memory.memory_store import create_memory_manager

    config = GeneratorConfig()
    memory = create_memory_manager()
    generator = StrategyGenerator(config, memory)

    stats = generator.get_pool_stats()
    click.echo(f"Pool stats: {json.dumps(stats, indent=2)}")

    best = generator.get_best_strategies(limit)
    click.echo(f"\nTop {limit} strategies:")
    for i, s in enumerate(best):
        click.echo(f"{i+1}. {s.name}: fitness={s.fitness:.4f}, gen={s.generation}, "
                   f"win_rate={s.win_rate:.2f}, sharpe={s.sharpe:.2f}, "
                   f"trades={s.trades_count}, pnl={s.total_pnl_pct:.2f}%")


if __name__ == "__main__":
    cli()