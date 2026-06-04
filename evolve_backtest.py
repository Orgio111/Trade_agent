"""
QUANTEX Strategy Evolution - Real Binance Data Backtest Optimization

Evolves strategy parameters (EMA, RSI, ATR) using a genetic algorithm
where fitness = backtest Sharpe ratio x profit factor on REAL Binance data.

Usage:
    python evolve_backtest.py              # Quick: 90d data, 10 gen
    python evolve_backtest.py --days 180   # More data
    python evolve_backtest.py --gen 20     # More generations
"""
import sys, os, asyncio, json, time, random, argparse
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator.backtest import DataLoader, BacktestEngine, BacktestResult
from orchestrator.strategy import StrategyEngine, Signal
from orchestrator.rl.strategy_evolver import StrategyEvolver, Strategy

# -- Config -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=90, help="Days of Binance data")
parser.add_argument("--gen", type=int, default=10, help="Evolution generations")
parser.add_argument("--pop", type=int, default=25, help="Population size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--interval", type=str, default="1h", help="Binance interval")
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

# -- Data ---------------------------------------------------
async def load_data():
    print(f"\n {'='*55}")
    print(f"  STEP 1: Fetching {args.days}d BTCUSDT {args.interval} data from Binance...")
    end = datetime.now()
    start = end - timedelta(days=args.days)
    df = await DataLoader.from_binance_api(
        symbol="BTCUSDT", interval=args.interval,
        start_time=start, end_time=end,
    )
    if df.empty:
        print("  [ERR] Binance API unavailable - cannot evolve without data")
        sys.exit(1)
    print(f"  [OK] {len(df)} candles from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"     Price range: ${df.close.min():.2f} -> ${df.close.max():.2f}")
    # 80/20 train/val split
    split = int(len(df) * 0.8)
    train, val = df.iloc[:split].copy(), df.iloc[split:].copy()
    print(f"     Train: {len(train)} candles | Val: {len(val)} candles")
    return train, val

# -- Fitness: backtest params on real Binance data ----------
backtest = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, slippage_bps=0.5)

async def evaluate_params(params: dict, data, label="?") -> BacktestResult:
    """Run backtest with given strategy params, return result."""
    strat = StrategyEngine(
        ema_fast=int(params["ema_fast"]),
        ema_slow=int(params["ema_slow"]),
        rsi_period=int(params.get("rsi_period", 14)),
        rsi_overbought=params.get("rsi_overbought", 70.0),
        rsi_oversold=params.get("rsi_oversold", 30.0),
        atr_multiplier_sl=params["atr_mult_sl"],
        atr_multiplier_tp1=params["atr_mult_tp"] * 0.67,
        atr_multiplier_tp2=params["atr_mult_tp"],
    )
    return await backtest.run(strat, data)

def make_fitness(train_data):
    """Create async fitness function for the evolver."""
    async def fitness_async(strat: Strategy) -> float:
        result = await evaluate_params(strat.params, train_data, strat.name)
        if result.total_trades < 3:
            return -1.0  # Penalize strategies that don't trade
        # Combined fitness: Sharpe x min(profit_factor, 3) x sqrt(trades) / 10
        pf = min(result.profit_factor if result.profit_factor != float("inf") else 3.0, 3.0)
        sharpe = max(result.sharpe_ratio, -2.0)
        score = sharpe * pf * (result.total_trades ** 0.3) / 10
        return max(score, -5.0)  # Clamp
    return fitness_async

# -- Evolution ----------------------------------------------
async def run_evolution(train_data):
    print(f"\n {'='*55}")
    print(f"  STEP 2: Running Genetic Evolution")
    print(f"     Population: {args.pop} | Generations: {args.gen} | Fitness: Backtest on real data")
    print(f"     Each generation = {args.pop} backtests x {args.gen} gen = {args.pop * args.gen} total")
    print(f" {'='*55}\n")

    # Default strategy fitness
    print("  Benchmarking default strategy (EMA 9/21, RSI 14)...")
    default_params = StrategyEvolver.default_strategy_params()
    default_result = await evaluate_params(default_params, train_data, "default")
    print(f"     Default on train: Sharpe={default_result.sharpe_ratio:.2f} PF={default_result.profit_factor:.2f} "
          f"Trades={default_result.total_trades} Win={default_result.win_rate:.1%}")

    # Build fitness wrapper that runs async inside sync evolver
    fit_fn = make_fitness(train_data)

    class AsyncEvolver(StrategyEvolver):
        """Override fitness evaluation to run async."""
        def initialize_population(self):
            self.population = []
            for i in range(self.pop_size):
                s = Strategy(name=f"gen0_strat_{i}", params=self.random_params(), generation=0)
                self.population.append(s)

        async def eval_population(self):
            """Evaluate all strategies in population."""
            for s in self.population:
                s.fitness = await fit_fn(s)
            self.population.sort(key=lambda s: s.fitness, reverse=True)
            self.best_ever = self.population[0]

        async def eval_and_report(self, gen_label: str):
            await self.eval_population()
            best = self.population[0]
            print(f"     {gen_label}: best={best.fitness:.4f} "
                  f"params={{ema_fast={best.params['ema_fast']}, ema_slow={best.params['ema_slow']}, "
                  f"rsi_period={best.params.get('rsi_period', '?')}, "
                  f"atr_mult_sl={best.params['atr_mult_sl']:.2f}, "
                  f"atr_mult_tp={best.params['atr_mult_tp']:.2f}}}")

    evolver = AsyncEvolver(fit_fn, population_size=args.pop, mutation_rate=0.15, elitism_pct=0.15)
    evolver.initialize_population()

    # Gen 0
    t0 = time.time()
    await evolver.eval_and_report("Gen 0/0")
    gen_time = time.time() - t0
    print(f"       [t={gen_time:.1f}s] (est. remaining: {gen_time * args.gen:.0f}s)")
    best_each_gen = [(0, evolver.population[0].fitness, dict(evolver.population[0].params))]

    # Generations 1..N
    for gen in range(1, args.gen + 1):
        t0 = time.time()

        # Create next generation
        new_pop = evolver.population[:evolver.elite_count]
        while len(new_pop) < evolver.pop_size:
            p1 = evolver._tournament_select()
            p2 = evolver._tournament_select()
            child_params = evolver._crossover(p1.params, p2.params)
            child_params = evolver._mutate(child_params)
            new_pop.append(Strategy(
                name=f"gen{gen}_strat_{len(new_pop)}", params=child_params, generation=gen,
            ))
        evolver.population = new_pop

        await evolver.eval_and_report(f"Gen {gen}/{args.gen}")
        best_each_gen.append((gen, evolver.population[0].fitness, dict(evolver.population[0].params)))
        gen_time = time.time() - t0
        remaining = gen_time * (args.gen - gen)
        print(f"       [t={gen_time:.1f}s] (est. remaining: {remaining:.0f}s)")

    # Track best ever
    for s in evolver.population:
        if s.fitness > (evolver.best_ever.fitness if evolver.best_ever else -999):
            evolver.best_ever = s

    return evolver, default_result, best_each_gen

# -- Validation ---------------------------------------------
async def validate(evolver, train_data, val_data, default_result):
    print(f"\n {'='*55}")
    print(f"  STEP 3: Validation on Out-of-Sample Data")
    print(f" {'='*55}")

    best = evolver.best_ever
    if not best:
        print("  [ERR] No best strategy found!")
        return

    print(f"\n  Best evolved params:")
    for k, v in best.params.items():
        print(f"     {k}: {v}")

    print(f"\n  Running backtests on VALIDATION data ({len(val_data)} candles)...")
    print(f"  --------------------------------------------------")

    # Default strategy on val
    def_params = StrategyEvolver.default_strategy_params()
    val_default = await evaluate_params(def_params, val_data, "default (val)")
    print(f"\n  [Default Strategy (EMA 9/21)]:")
    print(f"     Sharpe={val_default.sharpe_ratio:.2f} PF={val_default.profit_factor:.2f} "
          f"Trades={val_default.total_trades} Win={val_default.win_rate:.1%} "
          f"PnL=${val_default.total_pnl:.2f} DD={val_default.max_drawdown_pct:.1%}")

    # Best evolved on val
    val_best = await evaluate_params(best.params, val_data, "evolved (val)")
    print(f"\n  [Evolved Strategy (EMA {int(best.params['ema_fast'])}/{int(best.params['ema_slow'])}]):")
    print(f"     Sharpe={val_best.sharpe_ratio:.2f} PF={val_best.profit_factor:.2f} "
          f"Trades={val_best.total_trades} Win={val_best.win_rate:.1%} "
          f"PnL=${val_best.total_pnl:.2f} DD={val_best.max_drawdown_pct:.1%}")

    # Default on train
    train_default = await evaluate_params(def_params, train_data, "default (train)")
    train_best = await evaluate_params(best.params, train_data, "evolved (train)")

    # Summary table
    print(f"\n")
    print(f"  +==============================================================+")
    print(f"  |         STRATEGY COMPARISON (Real Binance Data)             |")
    print(f"  +--------------------+----------+----------+------------------+")
    print(f"  | Metric             | Default  | Evolved  | Evolved (val)    |")
    print(f"  +--------------------+----------+----------+------------------+")
    print(f"  | Sharpe             | {train_default.sharpe_ratio:>7.2f} | {train_best.sharpe_ratio:>7.2f} | {val_best.sharpe_ratio:>16.2f} |")
    print(f"  | Profit Factor      | {train_default.profit_factor:>7.2f} | {train_best.profit_factor:>7.2f} | {val_best.profit_factor:>16.2f} |")
    print(f"  | Win Rate           | {train_default.win_rate:>7.1%} | {train_best.win_rate:>7.1%} | {val_best.win_rate:>16.1%} |")
    print(f"  | Total PnL          | ${train_default.total_pnl:>5.2f}   | ${train_best.total_pnl:>5.2f}    | ${val_best.total_pnl:>12.2f}  |")
    print(f"  | Max Drawdown       | {train_default.max_drawdown_pct:>7.1%} | {train_best.max_drawdown_pct:>7.1%} | {val_best.max_drawdown_pct:>16.1%} |")
    print(f"  | Total Trades       | {train_default.total_trades:>7} | {train_best.total_trades:>7} | {val_best.total_trades:>16} |")
    print(f"  +--------------------+----------+----------+------------------+")

# -- Main ---------------------------------------------------
async def main():
    print(f"  {'='*55}")
    print(f"  QUANTEX Strategy Evolution - Real Binance Data Backtest")
    print(f"  {'='*55}")

    train_data, val_data = await load_data()
    evolver, default_result, gen_history = await run_evolution(train_data)
    await validate(evolver, train_data, val_data, default_result)

    # Print evolution progress
    print(f"\n  Evolution progress:")
    print(f"  Gen | Fitness | Params")
    print(f"  ---------------------------")
    for gen, fit, params in gen_history:
        print(f"  {gen:3d} | {fit:+.4f} | ema=({params['ema_fast']},{params['ema_slow']}) "
              f"rsi={params.get('rsi_period','?')} sl={params['atr_mult_sl']:.1f} tp={params['atr_mult_tp']:.1f}")

    print(f"\n  [OK] Evolution complete! Best params saved in evolver.best_ever")
    print(f"  To use these params in StrategyEngine:")
    print(f'     StrategyEngine(ema_fast={int(evolver.best_ever.params["ema_fast"])}, '
          f'ema_slow={int(evolver.best_ever.params["ema_slow"])}, '
          f'atr_multiplier_sl={evolver.best_ever.params["atr_mult_sl"]}, ...)')

if __name__ == "__main__":
    asyncio.run(main())
