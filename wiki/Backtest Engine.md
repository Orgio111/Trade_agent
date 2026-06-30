# Backtest Engine

## Overview
Replays historical candles through the full pipeline: AutoGen debate → EnhancedRisk → Paper execution → Metrics.

## Data Format (CSV)
```csv
timestamp,open,high,low,close,volume,rsi_14,atr_14,ema_9,ema_21,bb_width,atr_pct,atr_mean_pct,atr_std_pct
2024-01-01 00:00:00,60000,60500,59800,60200,100,55,500,60100,59900,0.02,0.008,0.008,0.001
```

Optional columns: `atr_pct`, `atr_mean_pct`, `atr_std_pct` for ATR filter.

## Usage
```bash
# From repo root
PYTHONPATH=. python -m orchestrator.autogen_trader backtest --data data/btc_1m.csv --output report.json
```

## Pipeline
```
CSV Row → Candle Dict → AutoGen Team → autogen_to_langgraph()
         → EnhancedRiskEngine.validate(signal, account, indicators)
         → PaperClient.place_order() → CycleResult
         → Accumulate → BacktestReport
```

## BacktestReport
```python
@dataclass
class BacktestReport:
    total_cycles: int
    completed: int
    rejected: int
    errors: int
    buys: int
    sells: int
    holds: int
    initial_equity: float
    final_equity: float
    total_pnl: float
    total_pnl_pct: float
    win_count: int
    loss_count: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_latency_ms: float
    trades: List[dict]  # per-cycle details
    equity_curve: List[float]
```

## Metrics
| Metric | Formula |
|--------|---------|
| Win Rate | wins / (wins + losses) |
| Sharpe | mean(pnl_pct) / std(pnl_pct) * sqrt(252) |
| Max DD | max(peak - trough) / peak |
| Avg Latency | mean(cycle.total_latency_ms) |

## Sample Report
```json
{
  "total_cycles": 1440,
  "completed": 892,
  "rejected": 548,
  "final_equity": 12450.0,
  "total_pnl_pct": 24.5,
  "win_rate": 0.62,
  "max_drawdown_pct": 8.3,
  "sharpe_ratio": 1.84
}
```

## Related
- [[AutoGen Trading Pipeline]]
- [[Enhanced Risk Engine]]
- [[CLI Reference]]