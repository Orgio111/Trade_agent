# Trade Agent — Multi-Agent Trading System

## Architecture
- **AutoGen Team** (5 agents): Quant → Pattern → Macro → Risk (filter) → Coordinator (consensus JSON)
- **EnhancedRiskEngine**: ATR volatility filter, Kelly criterion sizing, circuit breaker (kill streak → 5min cooldown)
- **Execution**: PaperClient (default) / BinanceClient (live) via async sync-wrapper
- **ModelLoader**: Sequential GPU loading (RTX 4050 6GB), `keep_alive="5m"`, `gpu_pinned=["phi3:mini"]`

## New Modules (`orchestrator/`)
| Module | Purpose |
|--------|---------|
| `autogen_executor.py` | Full pipeline: AutoGen debate → Risk → Execution |
| `autogen_trader.py` | CLI entry point (Click) |
| `enhanced_risk.py` | Hard filter + ATR + Kelly + circuit breaker |
| `backtest_engine.py` | Historical CSV/JSON replay |
| `model_loader.py` | VRAM-aware sequential loader with pre-warming |

## CLI Usage
```bash
# Single cycle (paper mode default)
python -m orchestrator.autogen_trader single --paper

# Live mode (DANGER)
python -m orchestrator.autogen_trader single --live

# Daemon mode
python -m orchestrator.autogen_trader daemon --interval 60 --paper

# Backtest
python -m orchestrator.autogen_trader backtest --data data/btc_1m.csv

# Status
python -m orchestrator.autogen_trader status
```

## Config (`config.yaml`)
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

## Key Gotchas
- **Ollama models load sequentially** — first cycle ~40s (warmup), subsequent cycles faster
- **PaperClient.place_order is async** — sync wrapper uses `loop.run_in_executor(None, sync_fn)`
- **HOLD signal always passes risk** (no confidence check)
- **ATR block uses `>=`** — exactly at threshold = blocked
- **Circuit breaker**: 5 consecutive losses → 5min cooldown

## Quick Commands
```bash
# Daily Startup
cd /path/to/Trade_agent
PYTHONPATH=. python -m orchestrator.autogen_trader single --paper

# View Logs
tail -f logs/agent.log | grep -E "(BUY|SELL|REJECTED|APPROVED)"

# Check Status
PYTHONPATH=. python -m orchestrator.autogen_trader status
```

## Related Notes
- [[AutoGen Trading Pipeline]]
- [[Enhanced Risk Engine]]
- [[Model Loader VRAM Strategy]]
- [[CLI Reference]]
- [[Backtest Engine]]
- [[Troubleshooting]]