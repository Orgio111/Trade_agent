# Trade Agent Wiki — Index

## Core Documentation
- [[Index]] — Main project overview (index.md)
- [[AutoGen Trading Pipeline]] — 5-agent flow diagram
- [[Enhanced Risk Engine]] — ATR, Kelly, circuit breaker
- [[Model Loader VRAM Strategy]] — Sequential GPU loading
- [[CLI Reference]] — All commands and flags
- [[Backtest Engine]] — Historical replay & metrics
- [[Troubleshooting]] — Common issues & fixes

## Quick Links
```bash
# Daily trading
PYTHONPATH=. python -m orchestrator.autogen_trader single --paper

# Continuous daemon
PYTHONPATH=. python -m orchestrator.autogen_trader daemon --interval 60 --paper

# Backtest
PYTHONPATH=. python -m orchestrator.autogen_trader backtest --data data/btc_1m.csv

# Status check
PYTHONPATH=. python -m orchestrator.autogen_trader status
```

## Key Files
| File | Purpose |
|------|---------|
| `orchestrator/autogen_team.py` | AutoGen GroupChat setup |
| `orchestrator/autogen_executor.py` | Full pipeline: AutoGen → Risk → Execution |
| `orchestrator/autogen_trader.py` | CLI entry point |
| `orchestrator/enhanced_risk.py` | Hard filter + ATR + Kelly + circuit breaker |
| `orchestrator/backtest_engine.py` | CSV/JSON replay |
| `orchestrator/model_loader.py` | VRAM-aware sequential loader |

## Config Reference
```yaml
model:
  warm_model: phi3:mini
  reasoning_model: qwen2.5:3b
  deep_model: qwen2.5:3b
  execution_model: qwen2:1.5b
  ollama_host: http://localhost:11434

risk:
  max_position_pct: 0.05
  max_daily_drawdown: 0.03
  kill_streak: 5
  min_confidence: 0.6
  atr_filter:
    enabled: true
    atr_threshold_mult: 2.5
```

## Related Concepts (from ZCode wiki)
- [[model-sequential-loading]] — VRAM management
- [[local-trading-ai-architecture]] — Overall local AI design
- [[incremental-candle-state]] — Memory architecture
- [[multi-agent-pipeline]] — Dual-path LangGraph state machine
- [[backtesting-pipeline]] — Validation infrastructure